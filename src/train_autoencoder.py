import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from tqdm import tqdm

from dataset_builder import create_temporal_sequences
from data_loader import load_dataset
from segmentation import generate_window_metadata
from labeling import label_windows
from autoencoder_reduction import FeatureAutoencoder
from utils import OUTPUTS_DIR

def train_ae_for_dataset(dataset_type="CHB", epochs=10, batch_size=64, device='cpu'):
    print(f"\n--- Training Autoencoder for {dataset_type} ---")
    
    # 1. Prepare Data
    records = load_dataset(dataset_type)
    window_metadata = generate_window_metadata(records)
    labeled_windows = label_windows(window_metadata)
    sequences = create_temporal_sequences(labeled_windows)
    
    # Extract features from all sequences
    # We'll use a dummy data loading step to get the features efficiently
    from dataset_builder import EEGDataset
    dataset = EEGDataset(sequences, is_train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Get one batch to find input dimension
    _, feats_sample, _ = next(iter(loader))
    input_dim = feats_sample.shape[2] # (B, S, F)
    latent_dim = 64
    
    model = FeatureAutoencoder(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for _, feats, _ in tqdm(loader, desc=f"AE Epoch {epoch+1}"):
            feats = feats.to(device).view(-1, input_dim) # Flatten S dimension into batch
            
            optimizer.zero_grad()
            reconstructed = model(feats)
            loss = criterion(reconstructed, feats)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.6f}")

    # Save
    save_path = os.path.join(OUTPUTS_DIR, 'saved_models', f'ae_{dataset_type}.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Autoencoder for {dataset_type} saved to {save_path}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ae_for_dataset("CHB", epochs=10, device=device)
    train_ae_for_dataset("SEIZE", epochs=10, device=device)
