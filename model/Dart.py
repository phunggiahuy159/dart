# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from .gba import GBA, ETC
# from .Encoder import MLPEncoder

# class DART(nn.Module):
#     def __init__(self,
#                  vocab_size,
#                  train_time_wordfreq,
#                  doc_tfidf,
#                  num_times,
#                  pretrained_WE=None,
#                  num_topics=50,
#                  en_units=100,
#                  temperature=0.1,
#                  beta_temp=0.7,
#                  weight_neg=1.0e+7,
#                  weight_pos=1.0,
#                  weight_beta_align=1.0e+3,
#                  weight_alpha=1.0,
#                  dropout=0.,
#                  embed_size=200,
#                  delta=0.1,
#                  beta_warm_up = 150
#                 ):
#         super().__init__()

#         self.num_topics = num_topics
#         self.beta_temp = beta_temp
#         self.train_time_wordfreq = train_time_wordfreq
#         self.doc_tfidf = doc_tfidf
#         self.num_times = num_times
#         self.delta = delta
#         self.vocab_size = vocab_size
#         self.embed_size = embed_size
#         self.global_beta = None
#         self.beta_history = []
#         self.beta_warm_up = beta_warm_up
#         self.weight_beta_align = weight_beta_align
#         self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
#         self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
#         self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))
#         self.mu2.requires_grad = False
#         self.var2.requires_grad = False

#         self.encoder = MLPEncoder(vocab_size, num_topics, en_units, dropout)
#         self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=False)

#         # Word embeddings
#         if pretrained_WE is None:
#             self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size), std=0.1)
#             self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))
#         else:
#             self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())

#         # Topic embeddings:Base embedding
#         self.topic_embeddings = nn.Parameter(
#             nn.init.xavier_normal_(torch.zeros(num_topics, self.word_embeddings.shape[1])).repeat(num_times, 1, 1)
#         )

#         # Alpha networks
#         self.alpha_fc_mean = nn.Sequential(
#             nn.Linear(self.embed_size, self.embed_size),
#             nn.ReLU(),
#             nn.Linear(self.embed_size, self.embed_size)
#         )
#         self.adaptive_dropout_net = nn.Sequential(
#             nn.Linear(self.embed_size, self.embed_size // 2),
#             nn.ReLU(),
#             nn.Linear(self.embed_size // 2, 1),
#             nn.Sigmoid() 
#         )
#         self.adaptive_dropout_net_mean = nn.Sequential(
#             nn.Linear(self.embed_size, self.embed_size // 2),
#             nn.ReLU(),
#             nn.Linear(self.embed_size // 2, 1),
#             nn.Sigmoid()
#         )
#         for m in self.adaptive_dropout_net_mean:
#           if isinstance(m, nn.Linear):
#               nn.init.normal_(m.weight, mean=0.0, std=0.05)
#               nn.init.constant_(m.bias, 0.0)
#         self.alpha_fc_logvar = nn.Sequential(
#             nn.Linear(self.embed_size, self.embed_size),
#             nn.ReLU(),
#             nn.Linear(self.embed_size, self.embed_size)
#         )

#         self.ETC = ETC(num_times, temperature, weight_neg, weight_pos)

#         self.Linear_Trend = nn.ModuleList(nn.Linear(i, 1) for i in range(1, self.num_times))
#         self.Linear_Seasonal = nn.ModuleList(nn.Linear(i, 1) for i in range(1, self.num_times))

#         self.alpha_mixing_net_seasonal = nn.Sequential(
#             nn.Linear(1, 100),
#             nn.ReLU(),
#             nn.Linear(100, 1),
#             nn.Softmax(dim=-1)
#         )

#         etc = ETC(num_times, temperature, weight_neg, weight_pos)
#         self.beta_loss = GBA(etc, num_times, temperature, self.weight_beta_align)

#         self.decomposition = self.series_decomp(23)

#     def compute_global_beta(self):
#         """Compute global beta by averaging stored beta history"""
#         if not self.beta_history:
#             return None
#         stacked_betas = torch.stack(self.beta_history, dim=0)
#         global_beta = torch.mean(stacked_betas, dim=0)
#         return global_beta

#     def series_decomp(self, kernel_size):
#         def decompose(x):
#             moving_mean = self.moving_avg(x, kernel_size)
#             res = x - moving_mean
#             return res, moving_mean
#         return decompose

#     def moving_avg(self, x, kernel_size, stride=1):
#         avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#         front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x

#     def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
#         if p_mu is not None and p_logsigma is not None:
#             sigma_q_sq = torch.exp(q_logsigma)
#             sigma_p_sq = torch.exp(p_logsigma)
#             kl = (sigma_q_sq + (q_mu - p_mu)**2) / (sigma_p_sq + 1e-6)
#             kl = kl - 1 + p_logsigma - q_logsigma
#             kl = 0.5 * torch.sum(kl, dim=-1)
#         else:
#             kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
#         return kl

#     def adaptive_dropout(self, x):
#         """
#         Apply learnable adaptive dropout to input tensor x
#         Args:
#             x: Input tensor of shape (num_topics, embed_size)
#         Returns:
#             Tensor with adaptive dropout applied
#         """
#         if not self.training:
#             return x
    
#         # Compute adaptive dropout rate for each topic (as variance)
#         dropout_rates = self.adaptive_dropout_net(x)  # (num_topics, 1)
#         mean = self.adaptive_dropout_net_mean(x)
#         dropout_rates = torch.clamp(dropout_rates, 0.0, 0.99)
    
#         # Expand dropout_rates to match x shape
#         dropout_rates_expanded = dropout_rates.expand_as(x)
    
#         denominator = torch.clamp(1.0 - dropout_rates_expanded, min=1e-6)        
#         std = torch.sqrt(dropout_rates_expanded / denominator)
#         noise = torch.normal(mean=mean, std=std)
    
#         dropped_x = x * noise
    
#         return dropped_x


#     def get_alpha(self):
#         device = self.topic_embeddings.device
#         time_scaling = torch.linspace(0, 1, steps=self.num_times, device=device)  # shape (num_times,)

#         kl_alpha = []
#         alphas = torch.zeros((self.num_times, self.num_topics, self.embed_size), device=device)

#         alpha_0 = self.topic_embeddings[0].clone()
#         alphas[0] = alpha_0

#         p_mu_0 = torch.zeros(self.num_topics, self.embed_size, device=device)
#         p_logsigma_0 = torch.zeros(self.num_topics, self.embed_size, device=device)

#         q_mu_0 = self.alpha_fc_mean(alpha_0)
#         q_logsigma_0 = self.alpha_fc_logvar(alpha_0)
#         kl_0 = self.get_kl(q_mu_0, q_logsigma_0, p_mu_0, p_logsigma_0)
#         kl_alpha.append(kl_0)

#         for t in range(1, self.num_times):
#             alpha_series = alphas[:t].permute(1, 0, 2).detach()

#             seas, trend = self.decomposition(alpha_series)  # seas, trend: (num_topics, t, embed_size)
#             # Reshape to apply the corresponding Linear layer in parallel.
#             seas_perm = seas.permute(0, 2, 1)  # shape: (num_topics, embed_size, t)
#             n_topics, emb_dim, seq_len = seas_perm.shape  # seq_len equals current t
#             seas_flat = seas_perm.reshape(-1, seq_len)  # shape: (num_topics*emb_dim, t)
#             l_seas = self.Linear_Seasonal[t-1](seas_flat)  # shape: (num_topics*emb_dim, 1)
#             l_seas = l_seas.reshape(n_topics, emb_dim)   # (num_topics, emb_dim)

#             trend_perm = trend.permute(0, 2, 1)  # shape: (num_topics, embed_size, t)
#             trend_flat = trend_perm.reshape(-1, seq_len)  # shape: (num_topics*emb_dim, t)
#             l_trend = self.Linear_Trend[t-1](trend_flat)  # shape: (num_topics*emb_dim, 1)
#             l_trend = l_trend.reshape(n_topics, emb_dim)  # (num_topics, emb_dim)

#             combined_alpha = l_trend + l_seas
#             dropped_combined_alpha = self.adaptive_dropout(combined_alpha)

#             # Use base topic embeddings and add the dropped combined_alpha
#             base_embedding = self.topic_embeddings[t]
#             current_alpha = base_embedding + dropped_combined_alpha

#             alphas[t] = current_alpha
#             p_mu_t = alphas[t-1].detach()
#             p_logsigma_t = torch.log(self.delta * torch.ones(self.num_topics, self.embed_size, device=device))
#             q_mu_t = self.alpha_fc_mean(current_alpha)
#             q_logsigma_t = self.alpha_fc_logvar(current_alpha)

#             kl_t = self.get_kl(q_mu_t, q_logsigma_t, p_mu_t, p_logsigma_t)
#             kl_alpha.append(kl_t)

#         kl_alpha = torch.stack(kl_alpha).sum()
#         return alphas, kl_alpha
#     def get_beta(self):
#         alphas, _ = self.get_alpha()
#         dist = self.pairwise_euclidean_dist(
#             F.normalize(alphas, dim=-1),
#             F.normalize(self.word_embeddings, dim=-1)
#         )
#         beta = F.softmax(-dist / self.beta_temp, dim=1)
#         return beta

#     def get_theta(self, x, times):
#         theta, mu, logvar = self.encoder(x)
#         if self.training:
#             return theta, mu, logvar
#         return theta

#     def get_KL(self, mu, logvar):
#         var = logvar.exp()
#         var_division = var / self.var2
#         diff = mu - self.mu2
#         diff_term = diff * diff / self.var2
#         logvar_division = self.var2.log() - logvar
#         KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
#         return KLD.mean()

#     def get_NLL(self, theta, beta, x, recon_x=None):
#         if recon_x is None:
#             recon_x = self.decode(theta, beta)
#         recon_loss = -(x * recon_x.log()).sum(axis=1)
#         return recon_loss

#     def decode(self, theta, beta):
#         d1 = F.softmax(self.decoder_bn(torch.bmm(theta.unsqueeze(1), beta).squeeze(1)), dim=-1)
#         return d1

#     def pairwise_euclidean_dist(self, x, y):
#         cost = torch.sum(x ** 2, axis=-1, keepdim=True) + torch.sum(y ** 2, axis=-1) - 2 * torch.matmul(x, y.t())
#         return cost

#     def forward(self, x, times, doc_embedding=None, epoch=None):
#         theta, mu, logvar = self.get_theta(x, times)
#         kl_theta = self.get_KL(mu, logvar)
#         alphas, kl_alpha = self.get_alpha()
#         beta = self.get_beta()
#         global_beta = torch.mean(beta, dim=0)
#         time_index_beta = beta[times]
#         recon_x = self.decode(theta, time_index_beta)
#         NLL = self.get_NLL(theta, time_index_beta, x, recon_x).mean()
#         loss_ETC = self.ETC(alphas)
#         # Only apply beta alignment loss in phase 2
#         beta_loss = 0.0
#         if epoch is not None and epoch > self.beta_warm_up:
#             beta_loss = self.beta_loss(self.doc_tfidf, global_beta, beta)
#             loss = NLL + kl_theta + loss_ETC + kl_alpha + beta_loss

#             rst_dict = {
#                 'loss': loss,
#                 'nll': NLL,
#                 'kl_theta': kl_theta,
#                 'kl_alpha': kl_alpha,
#                 'loss_ETC': loss_ETC,
#                 'beta_alignment': beta_loss,
#             }
#         else:
#             loss = NLL + kl_theta + loss_ETC + kl_alpha

#             rst_dict = {
#                 'loss': loss,
#                 'nll': NLL,
#                 'kl_theta': kl_theta,
#                 'kl_alpha': kl_alpha,
#                 'loss_ETC': loss_ETC,
#             }
#         return rst_dict



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
                 weight_alpha=1.0,
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