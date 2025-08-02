import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .gba import GBA, ETC
from .Encoder import MLPEncoder
from .alpha import AlphaGenerator


class DART(nn.Module):
    def __init__(self,
                 vocab_size,
                 train_time_wordfreq,
                 doc_tfidf,
                 num_times,
                 pretrained_WE=None,
                 num_topics=50,
                 en_units=100,
                 temperature=0.1,
                 beta_temp=0.7,
                 weight_neg=1.0e+7,
                 weight_pos=1.0,
                 weight_beta_align=1.0e+3,
                 dropout=0.,
                 embed_size=200,
                 delta=0.1,
                 beta_warm_up=150
                ):
        super().__init__()

        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.train_time_wordfreq = train_time_wordfreq
        self.doc_tfidf = doc_tfidf
        self.num_times = num_times
        self.delta = delta
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.global_beta = None
        self.beta_history = []
        self.beta_warm_up = beta_warm_up
        self.weight_beta_align = weight_beta_align
        
        # Prior parameters for theta
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))
        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        # Core components
        self.encoder = MLPEncoder(vocab_size, num_topics, en_units, dropout)
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=False)
        
        self.alpha_generator = AlphaGenerator(num_topics, embed_size, num_times, delta)

        # Word embeddings
        if pretrained_WE is None:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size), std=0.1)
            self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))
        else:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())

        # Topic embeddings: Base embedding
        self.topic_embeddings = nn.Parameter(
            nn.init.xavier_normal_(torch.zeros(num_topics, self.word_embeddings.shape[1])).repeat(num_times, 1, 1)
        )

        # ETC and loss components
        self.ETC = ETC(num_times, temperature, weight_neg, weight_pos)
        etc = ETC(num_times, temperature, weight_neg, weight_pos)
        self.beta_loss = GBA(etc, num_times, temperature, self.weight_beta_align)

    def compute_global_beta(self):
        """Compute global beta by averaging stored beta history"""
        if not self.beta_history:
            return None
        stacked_betas = torch.stack(self.beta_history, dim=0)
        global_beta = torch.mean(stacked_betas, dim=0)
        return global_beta

    def get_alpha(self):
        """Get alpha values using the AlphaGenerator"""
        return self.alpha_generator(self.topic_embeddings)

    def get_beta(self):
        """Compute beta values from alpha and word embeddings"""
        alphas, _ = self.get_alpha()
        dist = self.pairwise_euclidean_dist(
            F.normalize(alphas, dim=-1),
            F.normalize(self.word_embeddings, dim=-1)
        )
        beta = F.softmax(-dist / self.beta_temp, dim=1)
        return beta

    def get_theta(self, x, times):
        """Get theta values from encoder"""
        theta, mu, logvar = self.encoder(x)
        if self.training:
            return theta, mu, logvar
        return theta

    def get_KL(self, mu, logvar):
        """Compute KL divergence for theta"""
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        return KLD.mean()

    def get_NLL(self, theta, beta, x, recon_x=None):
        """Compute negative log likelihood"""
        if recon_x is None:
            recon_x = self.decode(theta, beta)
        recon_loss = -(x * recon_x.log()).sum(axis=1)
        return recon_loss

    def decode(self, theta, beta):
        """Decode theta and beta to reconstruct input"""
        d1 = F.softmax(self.decoder_bn(torch.bmm(theta.unsqueeze(1), beta).squeeze(1)), dim=-1)
        return d1

    def pairwise_euclidean_dist(self, x, y):
        """Compute pairwise Euclidean distance"""
        cost = torch.sum(x ** 2, axis=-1, keepdim=True) + torch.sum(y ** 2, axis=-1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, x, times, doc_embedding=None, epoch=None):
        """Forward pass of the DART model"""
        # Get theta and its KL divergence
        theta, mu, logvar = self.get_theta(x, times)
        kl_theta = self.get_KL(mu, logvar)
        
        # Get alpha and its KL divergence
        alphas, kl_alpha = self.get_alpha()
        
        # Get beta values
        beta = self.get_beta()
        global_beta = torch.mean(beta, dim=0)
        time_index_beta = beta[times]
        
        # Decode and compute reconstruction loss
        recon_x = self.decode(theta, time_index_beta)
        NLL = self.get_NLL(theta, time_index_beta, x, recon_x).mean()
        
        # Compute ETC loss
        loss_ETC = self.ETC(alphas)
        
        # Apply beta alignment loss only in phase 2 (after warm-up)
        beta_loss = 0.0
        if epoch is not None and epoch > self.beta_warm_up:
            beta_loss = self.beta_loss(self.doc_tfidf, global_beta, beta)
            loss = NLL + kl_theta + loss_ETC + kl_alpha + beta_loss

            rst_dict = {
                'loss': loss,
                'nll': NLL,
                'kl_theta': kl_theta,
                'kl_alpha': kl_alpha,
                'loss_ETC': loss_ETC,
                'beta_alignment': beta_loss,
            }
        else:
            loss = NLL + kl_theta + loss_ETC + kl_alpha

            rst_dict = {
                'loss': loss,
                'nll': NLL,
                'kl_theta': kl_theta,
                'kl_alpha': kl_alpha,
                'loss_ETC': loss_ETC,
            }
        
        return rst_dict