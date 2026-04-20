import torch
import torch.nn as nn
from .common_blocks import CNNFeatureExtractor

class WindowAttention1D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_attn=False):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        if return_attn:
            return x, attn
        return x

class SwinBlock1D(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, return_attn=False):
        B, L, C = x.shape
        # Pad to multiple of window_size
        pad_l = (self.window_size - L % self.window_size) % self.window_size
        if pad_l > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_l))
            
        _, L_padded, _ = x.shape
        
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        # Partition windows
        x_windows = shifted_x.view(-1, self.window_size, C)
        
        # Attention
        if return_attn:
            attn_windows, weights = self.attn(x_windows, return_attn=True)
        else:
            attn_windows = self.attn(x_windows)
        
        # Merge windows
        shifted_x = attn_windows.view(B, L_padded, C)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
            
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        if pad_l > 0:
            x = x[:, :L, :]
            
        if return_attn:
            return x, weights
        return x

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(DomainDiscriminator, self). __init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.net(x)

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2):
        super().__init__()
        # Use ModuleList to allow easy attention extraction
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=num_heads, 
                batch_first=True,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, return_attn=False):
        # x: (B, S, D)
        all_attn = []
        for i, layer in enumerate(self.layers):
            # For the last layer, we can record the output to see its saliency
            x = layer(x)
        
        x_norm = self.norm(x)
        out = self.dropout(x_norm.mean(dim=1))
        
        if return_attn:
            # We treat the magnitude of the sequence before pooling as temporal saliency
            return out, x 
        return out

class CNNSwinTransformerModel(nn.Module):
    def __init__(self, eeg_channels, latent_dim=64, num_classes=2):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels=eeg_channels, out_channels=32)
        
        self.proj = nn.Linear(32, 64)
        
        # Two Swin Blocks
        self.swin1 = SwinBlock1D(dim=64, num_heads=4, window_size=8, shift_size=0)
        self.swin2 = SwinBlock1D(dim=64, num_heads=4, window_size=8, shift_size=4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Domain-Invariant Instance Normalization (Hard scrubbing - no affine backdoor)
        self.input_norm = nn.InstanceNorm2d(23, affine=False)
        
        # Temporal Modeling Block (Simple Mean Pooling + Low Dropout)
        self.temporal = TemporalTransformer(input_dim=64 + latent_dim)
        
        # Domain Discriminator (GRL-based)
        self.domain_head = DomainDiscriminator(input_dim=64 + latent_dim)
        
        # Stability Layers
        self.feature_norm = nn.LayerNorm(64 + latent_dim)
        self.cls_dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(64 + latent_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
        # Xavier Initialization for stable start
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x_eeg_seq, x_latent_seq, return_features=False, xai_mode=False):
        B, S, C, F, T = x_eeg_seq.shape
        _, _, L = x_latent_seq.shape
        
        # Flatten sequence into batch for spatial features
        x = x_eeg_seq.reshape(B * S, C, F, T)
        latent = x_latent_seq.reshape(B * S, L)
        
        # Neutralize extreme dataset baseline bias
        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)
            
        # --- SENSOR DROPOUT FOR CROSS-DATASET GENERALIZATION ---
        if self.training:
            # Dynamically drop between 0% and 90% of channels per sample (Simulates SEIZE's 2-channel sparsity on CHB)
            drop_rates = torch.rand(B * S, 1, 1, 1, device=x.device) * 0.90
            mask = (torch.rand(B * S, C, 1, 1, device=x.device) > drop_rates).float()
            x = x * mask
            
        # 1. Feature Extraction
        cnn_out = self.cnn(x)                # (B*S, 32, T')
        x = cnn_out.transpose(1, 2)          # (B*S, T', 32)
        x = self.proj(x)                     # (B*S, T', 64)
        
        if xai_mode:
            f1, attn1 = self.swin1(x, return_attn=True)
            x, attn2 = self.swin2(f1, return_attn=True)
        else:
            f1 = self.swin1(x)                   
            x = self.swin2(f1)                    
        
        x_post_swin = x.transpose(1, 2)                # (B*S, 64, T')
        window_features = self.pool(x_post_swin).squeeze(-1) 
        
        # 2. Deep Feature Neutralization (Stripping bias without destroying signal)
        # Use LayerNorm for stability across all runs
        window_features = self.feature_norm(torch.cat((window_features, latent), dim=1))
        
        # Reshape to sequence
        combined_seq = window_features.view(B, S, -1) 
        
        # Apply Temporal Transformer with Mean Pooling
        if xai_mode:
            aggregated_features, temp_feat = self.temporal(combined_seq, return_attn=True)
        else:
            aggregated_features = self.temporal(combined_seq) 
        
        # Feature Scale Stabilization
        stable_features = self.feature_norm(aggregated_features)
        
        out = self.cls_dropout(stable_features)
        out = self.relu(self.fc1(out))
        logits = self.fc2(out)
        
        if xai_mode:
            return logits, {
                'attn1': attn1,
                'attn2': attn2,
                'temp_feat': temp_feat,
                'stable_features': stable_features
            }
            
        if return_features:
            return logits, stable_features
        return logits
        
    def forward_domain(self, aggregated_features, alpha=0.1):
        """
        Pass features through GRL and then the domain head.
        """
        rev_features = GradientReversal.apply(aggregated_features, alpha)
        domain_logits = self.domain_head(rev_features)
        return domain_logits
