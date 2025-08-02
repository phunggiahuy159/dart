import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGenerator(nn.Module):
    def __init__(self, num_topics, embed_size, num_times, delta=0.1):
        super().__init__()
        self.num_topics = num_topics
        self.embed_size = embed_size
        self.num_times = num_times
        self.delta = delta
        
        # Alpha networks
        self.alpha_fc_mean = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
        
        self.alpha_fc_logvar = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
        
        # Adaptive dropout networks
        self.adaptive_dropout_net = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 1),
            nn.Sigmoid() 
        )
        
        self.adaptive_dropout_net_mean = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize adaptive dropout mean network
        for m in self.adaptive_dropout_net_mean:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                nn.init.constant_(m.bias, 0.0)
        
        # Linear layers for trend and seasonal components
        self.Linear_Trend = nn.ModuleList(nn.Linear(i, 1) for i in range(1, num_times))
        self.Linear_Seasonal = nn.ModuleList(nn.Linear(i, 1) for i in range(1, num_times))
        
        # Series decomposition
        self.decomposition = self._series_decomp(23)
    
    def _series_decomp(self, kernel_size):
        """Create series decomposition function"""
        def decompose(x):
            moving_mean = self._moving_avg(x, kernel_size)
            res = x - moving_mean
            return res, moving_mean
        return decompose
    
    def _moving_avg(self, x, kernel_size, stride=1):
        """Compute moving average for series decomposition"""
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        
        front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Compute KL divergence"""
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu)**2) / (sigma_p_sq + 1e-6)
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl
    
    def adaptive_dropout(self, x):
        """Apply learnable adaptive dropout to input tensor x"""
        if not self.training:
            return x
        
        dropout_rates = self.adaptive_dropout_net(x)  # (num_topics, 1)
        mean = self.adaptive_dropout_net_mean(x)
        dropout_rates = torch.clamp(dropout_rates, 0.0, 0.99)
        
        # Expand dropout_rates to match x shape
        dropout_rates_expanded = dropout_rates.expand_as(x)
        
        denominator = torch.clamp(1.0 - dropout_rates_expanded, min=1e-6)        
        std = torch.sqrt(dropout_rates_expanded / denominator)
        noise = torch.normal(mean=mean, std=std)
        
        dropped_x = x * noise
        
        return dropped_x
    
    def forward(self, topic_embeddings):
        """
        Generate alpha values for all time steps
        
        Args:
            topic_embeddings: Tensor of shape (num_times, num_topics, embed_size)
            
        Returns:
            tuple: (alphas, kl_alpha)
                - alphas: Tensor of shape (num_times, num_topics, embed_size)
                - kl_alpha: Scalar tensor representing total KL divergence
        """
        device = topic_embeddings.device
        
        kl_alpha = []
        alphas = torch.zeros((self.num_times, self.num_topics, self.embed_size), device=device)
        
        alpha_0 = topic_embeddings[0].clone()
        alphas[0] = alpha_0
        
        # KL divergence for first time step (prior is standard normal)
        p_mu_0 = torch.zeros(self.num_topics, self.embed_size, device=device)
        p_logsigma_0 = torch.zeros(self.num_topics, self.embed_size, device=device)
        
        q_mu_0 = self.alpha_fc_mean(alpha_0)
        q_logsigma_0 = self.alpha_fc_logvar(alpha_0)
        kl_0 = self.get_kl(q_mu_0, q_logsigma_0, p_mu_0, p_logsigma_0)
        kl_alpha.append(kl_0)
        
        # Generate subsequent time steps
        for t in range(1, self.num_times):
            alpha_series = alphas[:t].permute(1, 0, 2).detach()
            
            # Decompose into seasonal and trend components
            seas, trend = self.decomposition(alpha_series)  # seas, trend: (num_topics, t, embed_size)
            
            # Apply linear transformations to seasonal component
            seas_perm = seas.permute(0, 2, 1)  # shape: (num_topics, embed_size, t)
            n_topics, emb_dim, seq_len = seas_perm.shape  # seq_len equals current t
            seas_flat = seas_perm.reshape(-1, seq_len)  # shape: (num_topics*emb_dim, t)
            l_seas = self.Linear_Seasonal[t-1](seas_flat)  # shape: (num_topics*emb_dim, 1)
            l_seas = l_seas.reshape(n_topics, emb_dim)   # (num_topics, emb_dim)
            
            # Apply linear transformations to trend component
            trend_perm = trend.permute(0, 2, 1)  # shape: (num_topics, embed_size, t)
            trend_flat = trend_perm.reshape(-1, seq_len)  # shape: (num_topics*emb_dim, t)
            l_trend = self.Linear_Trend[t-1](trend_flat)  # shape: (num_topics*emb_dim, 1)
            l_trend = l_trend.reshape(n_topics, emb_dim)  # (num_topics, emb_dim)
            
            # Combine trend and seasonal components
            combined_alpha = l_trend + l_seas
            dropped_combined_alpha = self.adaptive_dropout(combined_alpha)
            
            # Use base topic embeddings and add the dropped combined_alpha
            base_embedding = topic_embeddings[t]
            current_alpha = base_embedding + dropped_combined_alpha
            
            alphas[t] = current_alpha
            
            # Compute KL divergence for current time step
            p_mu_t = alphas[t-1].detach()
            p_logsigma_t = torch.log(self.delta * torch.ones(self.num_topics, self.embed_size, device=device))
            q_mu_t = self.alpha_fc_mean(current_alpha)
            q_logsigma_t = self.alpha_fc_logvar(current_alpha)
            
            kl_t = self.get_kl(q_mu_t, q_logsigma_t, p_mu_t, p_logsigma_t)
            kl_alpha.append(kl_t)
        
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha