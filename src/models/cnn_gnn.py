import torch
import torch.nn as nn
from .common_blocks import PerChannelCNNExtractor
from .cnn_swin_transformer import GradientReversal, DomainDiscriminator, TemporalTransformer

class CNNGNNModel(nn.Module):
    def __init__(self, eeg_channels, latent_dim=64, num_classes=2):
        super().__init__()
        self.node_extractor = PerChannelCNNExtractor(out_features=32)
        # Simple GAT-like fully connected graph
        self.gat = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.proj = nn.Linear(32, 64)
        
        self.input_norm = nn.InstanceNorm2d(23, affine=False)
        self.temporal = TemporalTransformer(input_dim=64 + latent_dim)
        self.domain_head = DomainDiscriminator(input_dim=64 + latent_dim)
        self.feature_norm = nn.LayerNorm(64 + latent_dim)
        self.cls_dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 + latent_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x_eeg_seq, x_latent_seq, return_features=False, xai_mode=False):
        B, S, C, F, T = x_eeg_seq.shape
        _, _, L = x_latent_seq.shape
        x = x_eeg_seq.reshape(B * S, C, F, T)
        latent = x_latent_seq.reshape(B * S, L)
        
        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)
            
        if self.training:
            drop_rates = torch.rand(B * S, 1, 1, 1, device=x.device) * 0.90
            mask = (torch.rand(B * S, C, 1, 1, device=x.device) > drop_rates).float()
            x = x * mask
            
        nodes = self.node_extractor(x)  # (B*S, C, 32)
        gnn_out, _ = self.gat(nodes, nodes, nodes) # (B*S, C, 32)
        gnn_out = gnn_out.mean(dim=1) # Average pool over nodes -> (B*S, 32)
        window_features = self.proj(gnn_out) # (B*S, 64)
        
        window_features = self.feature_norm(torch.cat((window_features, latent), dim=1))
        combined_seq = window_features.view(B, S, -1) 
        
        if xai_mode:
            aggregated_features, temp_feat = self.temporal(combined_seq, return_attn=True)
        else:
            aggregated_features = self.temporal(combined_seq) 
            
        stable_features = self.feature_norm(aggregated_features)
        
        out = self.cls_dropout(stable_features)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out)
        
        if xai_mode:
            return logits, {
                'attn1': None,
                'attn2': None,
                'temp_feat': temp_feat,
                'stable_features': stable_features
            }
            
        if return_features:
            return logits, stable_features
        return logits
        
    def forward_domain(self, aggregated_features, alpha=0.1):
        rev_features = GradientReversal.apply(aggregated_features, alpha)
        return self.domain_head(rev_features)
