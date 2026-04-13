import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import SEIZEIT2_DIR, OUTPUTS_DIR, set_seed, setup_logger
from data_loader import get_seizeit2_records
from segmentation import generate_window_metadata
from labeling import label_windows
from dataset_builder import get_dataloaders
from autoencoder_reduction import FeatureAutoencoder, get_latent_dimension
from models.cnn_lstm import CNNLSTMModel
from models.cnn_swin_transformer import CNNSwinTransformerModel
from models.cnn_gnn import CNNGNNModel

def train_autoencoder(autoencoder, train_loader, device, num_epochs=3, logger=None):
    if logger: logger.info("Pre-training Autoencoder...")
    autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0
        # iterate over dataset (note: we only need features. EEGDataset returns signal, features, label)
        for _, features, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}/{num_epochs}"):
            features = features.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(features)
            loss = criterion(reconstructed, features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if logger:
            logger.info(f"AE Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")
            
    autoencoder.eval()
    if logger: logger.info("Autoencoder pre-training complete.")

def train_model(model_name, num_epochs=5, batch_size=32, device='cpu'):
    set_seed(42)
    logger = setup_logger('train', os.path.join(OUTPUTS_DIR, 'reports', 'train.log'))
    device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # 1. Pipeline Setup
    logger.info("Loading dataset metadata...")
    records = get_seizeit2_records(SEIZEIT2_DIR)
    
    logger.info("Generating sliding windows (5s)...")
    # Limiting to a small subset of records for dry-runs via slicing, otherwise full dataset
    window_metadata = generate_window_metadata(records, window_size_sec=5.0, overlap_ratio=0.5)
    
    logger.info("Assigning labels...")
    labeled_windows = label_windows(window_metadata)
    
    logger.info(f"Total extracted windows across all datasets: {len(labeled_windows)}")
    
    logger.info("Splitting dataset and building DataLoaders...")
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(labeled_windows, batch_size=batch_size, num_workers=0)
    
    # Try fetching a single batch to determine input dimensions
    logger.info("Fetching a test batch to initialize dimensions...")
    sig_shape = None
    feat_shape = None
    for sig, feat, label in train_loader:
        sig_shape = sig.shape
        feat_shape = feat.shape
        break
        
    if sig_shape is None:
        logger.error("Dataloader returned no batches!")
        return
        
    eeg_channels = sig_shape[1]
    input_feature_dim = feat_shape[1]
    
    logger.info(f"EEG Channels: {eeg_channels}, Handcrafted Feature Dim: {input_feature_dim}")
    
    # 2. Autoencoder Pretraining
    autoencoder = FeatureAutoencoder(input_dim=input_feature_dim, latent_dim=64)
    train_autoencoder(autoencoder, train_loader, device, num_epochs=2, logger=logger) # low epochs to save time
    
    for param in autoencoder.parameters():
        param.requires_grad = False
        
    # 3. Model Selection
    if model_name == 'cnn_lstm':
        model = CNNLSTMModel(eeg_channels=eeg_channels, latent_dim=64)
    elif model_name == 'cnn_swin':
        model = CNNSwinTransformerModel(eeg_channels=eeg_channels, latent_dim=64)
    elif model_name == 'cnn_gnn':
        model = CNNGNNModel(eeg_channels=eeg_channels, latent_dim=64)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
        
    model.to(device)
    logger.info(f"Initialized {model_name}")

    # Apply 2:1 class weighting to penalize missing seizures (Class 1) more heavily than background (Class 0)
    class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for sigs, feats, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
            sigs, feats, labels = sigs.to(device), feats.to(device), labels.to(device)
            
            latent_feats = autoencoder.encode(feats)
            
            optimizer.zero_grad()
            outputs = model(sigs, latent_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sigs, feats, labels in val_loader:
                sigs, feats, labels = sigs.to(device), feats.to(device), labels.to(device)
                latent_feats = autoencoder.encode(feats)
                outputs = model(sigs, latent_feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(OUTPUTS_DIR, 'saved_models', f"best_{model_name}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'autoencoder_state_dict': autoencoder.state_dict(),
                'eeg_channels': eeg_channels,
                'feature_dim': input_feature_dim
            }, save_path)
            logger.info(f"Saved new best model to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['cnn_lstm', 'cnn_swin', 'cnn_gnn'], default='cnn_swin')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu') # Strict CPU optimization requested
    
    args = parser.parse_args()
    
    # Run from root of src
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    train_model(args.model, args.epochs, args.batch_size, args.device)
