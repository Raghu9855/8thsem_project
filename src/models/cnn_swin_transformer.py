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

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
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

    def forward(self, x):
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
            
        return x

class CNNSwinTransformerModel(nn.Module):
    def __init__(self, eeg_channels, latent_dim=64, num_classes=2):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels=eeg_channels, out_channels=32)
        
        self.proj = nn.Linear(32, 64)
        
        # Two Swin Blocks (one standard, one shifted)
        self.swin1 = SwinBlock1D(dim=64, num_heads=4, window_size=8, shift_size=0)
        self.swin2 = SwinBlock1D(dim=64, num_heads=4, window_size=8, shift_size=4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(64 + latent_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x_eeg, x_latent):
        cnn_out = self.cnn(x_eeg) # (B, 32, T')
        
        x = cnn_out.transpose(1, 2) # (B, T', 32)
        x = self.proj(x)            # (B, T', 64)
        
        x = self.swin1(x)
        x = self.swin2(x)           # (B, T', 64)
        
        x = x.transpose(1, 2)       # (B, 64, T')
        x = self.pool(x).squeeze(-1) # (B, 64)
        
        combined = torch.cat((x, x_latent), dim=1)
        
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
