import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from tqdm import tqdm

class GBA(nn.Module):
    def __init__(self, ETC, num_times, temperature, weight_UWE, use_percentage=True, percentage=0.7, num_high_weight_tokens=20, min_tf_threshold=0.0):
        super().__init__()

        self.ETC = ETC
        self.weight_UWE = weight_UWE
        self.num_times = num_times
        self.temperature = temperature
        self.use_percentage = use_percentage  
        self.percentage = percentage
        self.num_high_weight_tokens = num_high_weight_tokens
        self.min_tf_threshold = min_tf_threshold

    def forward(self, doc_tfidf, global_beta, local_beta):
        """Contrastive loss calculation using document-level TF-IDF scores"""
        device = doc_tfidf.device

        if not isinstance(global_beta, torch.Tensor):
            global_beta = torch.tensor(global_beta, device=device)
        elif global_beta.device != device:
            global_beta = global_beta.to(device)

        total_loss = torch.tensor(0.0, device=device)
        valid_times = torch.tensor(0, device=device)

        for t in range(self.num_times):
            time_tfidf = doc_tfidf[t]

            if torch.sum(time_tfidf) == 0:
                continue

            if self.min_tf_threshold > 0:
                valid_tokens = torch.nonzero(time_tfidf > self.min_tf_threshold, as_tuple=True)[0]
                if len(valid_tokens) == 0:
                    valid_tokens = torch.nonzero(time_tfidf > 0, as_tuple=True)[0]
            else:
                valid_tokens = torch.nonzero(time_tfidf > 0, as_tuple=True)[0]

            if len(valid_tokens) == 0:
                continue

            if self.use_percentage:
                num_to_keep = max(1, int(len(valid_tokens) * self.percentage))

                token_scores = time_tfidf[valid_tokens]
                _, top_indices = torch.topk(token_scores, num_to_keep)
                high_weight_tokens = valid_tokens[top_indices]
            else:
                if len(valid_tokens) > self.num_high_weight_tokens:
                    token_scores = time_tfidf[valid_tokens]
                    _, top_indices = torch.topk(token_scores, self.num_high_weight_tokens)
                    high_weight_tokens = valid_tokens[top_indices]
                else:
                    high_weight_tokens = valid_tokens

            mask = torch.zeros(time_tfidf.shape[0], device=device, dtype=torch.float32)
            mask.index_fill_(0, high_weight_tokens, 1.0)

            if global_beta.dim() == 3:
                global_beta_t = global_beta[t]
            elif global_beta.dim() == 2:
                global_beta_t = global_beta
            else:
                raise ValueError(f"global_beta has unexpected shape: {global_beta.shape}")

            local_beta_t = local_beta[t]

            if global_beta_t.shape != local_beta_t.shape:
                raise ValueError(f"Shape mismatch: global_beta_t {global_beta_t.shape}, local_beta_t {local_beta_t.shape} at time {t}")
            if global_beta_t.shape[1] != mask.shape[0]:
                raise ValueError(f"Shape mismatch: beta vocab {global_beta_t.shape[1]}, mask {mask.shape[0]}")

            global_filtered = global_beta_t * mask
            local_filtered = local_beta_t * mask

            global_sum = torch.sum(global_filtered, dim=1, keepdim=True)
            local_sum = torch.sum(local_filtered, dim=1, keepdim=True)

            global_sum = torch.where(global_sum < 1e-10, torch.ones_like(global_sum) * 1e-10, global_sum)
            local_sum = torch.where(local_sum < 1e-10, torch.ones_like(local_sum) * 1e-10, local_sum)

            global_filtered = global_filtered / global_sum
            local_filtered = local_filtered / local_sum

            if torch.isnan(global_filtered).any() or torch.isnan(local_filtered).any():
                continue

            contrastive_loss = self.ETC.compute_loss(
                global_filtered,
                local_beta_t,
                temperature=self.temperature,
                self_contrast=False,
                only_pos=True
            )

            total_loss += contrastive_loss
            valid_times += 1

        if valid_times > 0:
            total_loss = total_loss * (self.weight_UWE / valid_times.float())

        return total_loss

class ETC(nn.Module):
    def __init__(self, num_times, temperature, weight_neg, weight_pos):
        super().__init__()
        self.num_times = num_times
        self.weight_neg = weight_neg
        self.weight_pos = weight_pos
        self.temperature = temperature

    def forward(self, topic_embeddings):
        loss_neg = 0.
        for t in range(self.num_times):
            loss_neg += self.compute_loss(topic_embeddings[t], topic_embeddings[t], self.temperature, self_contrast=True)
        loss_neg *= (self.weight_neg / self.num_times)
        return loss_neg

    def compute_loss(self, anchor_feature, contrast_feature, temperature, self_contrast=False, only_pos=False, all_neg=False):
        anchor_dot_contrast = torch.div(
            torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T),
            temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        pos_mask = torch.eye(anchor_dot_contrast.shape[0]).to(anchor_dot_contrast.device)

        if self_contrast is False:
            if only_pos is False:
                if all_neg is True:
                    exp_logits = torch.exp(logits)
                    sum_exp_logits = exp_logits.sum(1)
                    log_prob = -torch.log(sum_exp_logits + 1e-12)
                    mean_log_prob = -log_prob.sum() / (logits.shape[0] * logits.shape[1])
            else:
                mean_log_prob = -(logits * pos_mask).sum() / pos_mask.sum()
        else:
            exp_logits = torch.exp(logits) * (1 - pos_mask)
            sum_exp_logits = exp_logits.sum(1)
            log_prob = -torch.log(sum_exp_logits + 1e-12)
            mean_log_prob = -log_prob.sum() / (1 - pos_mask).sum()

        return mean_log_prob