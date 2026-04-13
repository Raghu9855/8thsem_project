import torch
import torch.nn as nn
from .common_blocks import CNNFeatureExtractor

class CNNLSTMModel(nn.Module):
    def __init__(self, eeg_channels, latent_dim=64, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        
        self.cnn = CNNFeatureExtractor(in_channels=eeg_channels, out_channels=32)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        
        # Classifier
        # Context vector from LSTM (64) + latent dim from Autoencoder (64)
        self.fc1 = nn.Linear(64 + latent_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x_eeg, x_latent):
        # x_eeg: (B, C, T)
        # x_latent: (B, latent_dim)
        
        # 1. Local temporal feature extraction
        cnn_out = self.cnn(x_eeg)  # (B, F, T')
        
        # 2. Sequential learning (LSTM needs (B, T', F))
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, (hn, cn) = self.lstm(cnn_out)
        
        # Grab the last hidden state from the last layer
        lstm_last = hn[-1, :, :]  # (B, 64)
        
        # 3. Combine with Autoencoder features
        combined = torch.cat((lstm_last, x_latent), dim=1)  # (B, 64 + latent)
        
        # 4. Classification
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
