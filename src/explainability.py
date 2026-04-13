import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

from utils import OUTPUTS_DIR, set_seed, SEIZEIT2_DIR
from dataset_builder import get_dataloaders
from data_loader import get_seizeit2_records
from segmentation import generate_window_metadata
from labeling import label_windows
from autoencoder_reduction import FeatureAutoencoder
from models.cnn_swin_transformer import CNNSwinTransformerModel

# To run SHAP on the Autoencoder features
def explain_features_with_shap(device='cpu'):
    set_seed(42)
    device = torch.device(device)
    
    # We will use SWIN as the representative model
    save_path = os.path.join(OUTPUTS_DIR, 'saved_models', "best_cnn_swin.pth")
    if not os.path.exists(save_path):
        print(f"Skipping XAI, cannot find {save_path}")
        return
        
    checkpoint = torch.load(save_path, map_location=device)
    
    autoencoder = FeatureAutoencoder(input_dim=checkpoint['feature_dim']).to(device)
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    autoencoder.eval()
    
    model = CNNSwinTransformerModel(eeg_channels=checkpoint['eeg_channels']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # We construct a wrapper for SHAP that only takes tabular features 
    # to evaluate which handcrafted features influence the Autoencoder -> Dense layers.
    # Because our model requires BOTH EEG and Tabular, we will freeze a sample EEG baseline.
    
    records = get_seizeit2_records(SEIZEIT2_DIR)
    window_metadata = generate_window_metadata(records, window_size_sec=5.0) # Full metadata allows split to work
    labeled_windows = label_windows(window_metadata)
    
    _, _, test_loader, _ = get_dataloaders(labeled_windows, batch_size=5)
    
    baseline_sigs, baseline_feats, _ = next(iter(test_loader))
    baseline_sigs = baseline_sigs.to(device)
    baseline_feats = baseline_feats.to(device)
    
    def model_predict(feats_numpy):
        feats_tensor = torch.tensor(feats_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            latent = autoencoder.encode(feats_tensor)
            # Re-use the first signal for simplicity in viewing Tabular importance independently
            sig_reps = baseline_sigs[0].unsqueeze(0).repeat(feats_tensor.size(0), 1, 1, 1)
            out = model(sig_reps, latent)
            return out[:, 1].cpu().numpy() # Probability of seizure
            
    explainer = shap.KernelExplainer(model_predict, baseline_feats.cpu().numpy())
    shap_values = explainer.shap_values(baseline_feats.cpu().numpy()[:5])
    
    shap.summary_plot(shap_values, baseline_feats.cpu().numpy()[:5], show=False)
    plt.savefig(os.path.join(OUTPUTS_DIR, 'xai', 'shap_tabular_importance.png'))
    print("Saved SHAP tabular importance.")
    
if __name__ == '__main__':
    explain_features_with_shap()
