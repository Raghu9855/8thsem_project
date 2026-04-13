import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(CNNFeatureExtractor, self).__init__()
        # Input shape: (Batch, Channels, FreqBins, TimeSteps)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(5, 5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
    def forward(self, x):
        # x: (B, C, F, T)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.mean(dim=2) # Pool over F dimension
        return x  # (B, out_channels, T_out)

class PerChannelCNNExtractor(nn.Module):
    """
    Applies CNN feature extraction independently per channel.
    Useful for GNNs where each channel is a node.
    """
    def __init__(self, out_features=16):
        super(PerChannelCNNExtractor, self).__init__()
        # Uses 2D conv over the time/freq dimension, treating in_channels=1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv2 = nn.Conv2d(8, out_features, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (Batch, Channels, FreqBins, TimeBins)
        B, C, F, T = x.shape
        # Reshape to (Batch * Channels, 1, F, T)
        x = x.view(B * C, 1, F, T)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # (Batch * Channels, out_features, 1, 1)
        x = x.view(B, C, -1)  # (Batch, Channels, out_features)
        return x
