import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

from utils import OUTPUTS_DIR, set_seed, CHBMIT_DIR
from dataset_builder import get_dataloaders
from data_loader import get_chbmit_records
from segmentation import generate_window_metadata
from labeling import label_windows
from autoencoder_reduction import FeatureAutoencoder
from models.cnn_swin_transformer import CNNSwinTransformerModel

if __name__ == '__main__':
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join('outputs', 'saved_models', 'best_cnn_swin_CHB_to_CHB.pth')
    out_dir = os.path.join(OUTPUTS_DIR, 'paper_xai_figures')
    os.makedirs(out_dir, exist_ok=True)
    set_seed(42)

    print("Loading model and data...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    autoencoder = FeatureAutoencoder(input_dim=checkpoint['feature_dim']).to(device)
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    autoencoder.eval()

    eeg_channels = checkpoint['eeg_channels']
    model = CNNSwinTransformerModel(eeg_channels=eeg_channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    records = get_chbmit_records(CHBMIT_DIR)
    records = records[:2] 
    window_metadata = generate_window_metadata(records, window_size_sec=5.0)
    labeled_windows = label_windows(window_metadata)
    _, _, test_loader, _ = get_dataloaders(labeled_windows, batch_size=16)

    class SHAPWrapper(nn.Module):
        def __init__(self, model, autoencoder):
            super().__init__()
            self.model = model
            self.autoencoder = autoencoder
        def forward(self, sig, feat):
            lat = self.autoencoder.encode(feat)
            return self.model(sig, lat)[:, 1:2]
            
    wrapped_model = SHAPWrapper(model, autoencoder).to(device)
    wrapped_model.eval()

    # Collect a batch with mixed labels
    sigs, feats, labels = next(iter(test_loader))
    bg_sig, bg_feat = sigs.to(device), feats.to(device)

    print(f"Computing SHAP values for batch of size {len(labels)}...")
    explainer = shap.GradientExplainer(wrapped_model, [bg_sig, bg_feat])
    test_sig, test_feat = bg_sig, bg_feat
    shap_values = explainer.shap_values([test_sig, test_feat])

    if isinstance(shap_values, list) and isinstance(shap_values[0], list):
        shap_sig, shap_feat = shap_values[0][0], shap_values[1][0]
    elif isinstance(shap_values, list):
        shap_sig, shap_feat = shap_values[0], shap_values[1]
    else:
        shap_sig, shap_feat = shap_values, shap_values

    # Remove the trailing class dimension if it exists
    if shap_sig.shape[-1] == 1:
        shap_sig = np.squeeze(shap_sig, axis=-1)
    if shap_feat.shape[-1] == 1:
        shap_feat = np.squeeze(shap_feat, axis=-1)

    B, S, F_D = shap_feat.shape
    C = eeg_channels
    feature_names = ['Mean', 'Var', 'RMS', 'Skew', 'Kurt', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'PE', 'HFD']

    # Reshape and mean over S and C
    shap_feat_reshaped = shap_feat.reshape(B, S, C, 12)
    shap_feat_raw_agg = shap_feat_reshaped.mean(axis=(1, 2)) # Shape: (B, 12)

    feat_np = test_feat.cpu().numpy()
    feat_reshaped = feat_np.reshape(B, S, C, 12)
    feat_agg = feat_reshaped.mean(axis=(1, 2)) # Shape: (B, 12)

    # --- 1. GLOBAL EXPLANATIONS ---
    print("Generating Global Explanations (Beeswarm & Bar)...")
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_feat_raw_agg, features=feat_agg, feature_names=feature_names, show=False)
    plt.title("SHAP Summary (Global Feature Importance)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_beeswarm.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_feat_raw_agg, features=feat_agg, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Mean |SHAP| (Global Feature Importance)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_bar.png'))
    plt.close()

    # --- 2. LOCAL EXPLANATIONS ---
    print("Generating Local Explanations (Waterfall)...")
    with torch.no_grad():
        lat = autoencoder.encode(test_feat)
        logits, xai = model(test_sig, lat, xai_mode=True)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
    labels_np = labels.numpy()
    correct_idx = np.where(preds == labels_np)[0]
    incorrect_idx = np.where(preds != labels_np)[0]

    def plot_waterfall(idx, name):
        if len(idx) == 0: return
        i = idx[0]
        try:
            expected_value = explainer.expected_value
            if isinstance(expected_value, list): expected_value = expected_value[-1]
            if isinstance(expected_value, np.ndarray): expected_value = expected_value[0]
            if isinstance(expected_value, torch.Tensor): expected_value = expected_value.item()
        except AttributeError:
            with torch.no_grad():
                expected_value = wrapped_model(bg_sig, bg_feat).mean().item()
            
        exp = shap.Explanation(values=shap_feat_raw_agg[i], 
                               base_values=float(expected_value), 
                               data=feat_agg[i], 
                               feature_names=feature_names)
        plt.figure(figsize=(8, 5))
        shap.waterfall_plot(exp, show=False)
        plt.title(f"Local Explanation: {name} (True: {labels_np[i]}, Pred: {preds[i]})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'shap_waterfall_{name}.png'))
        plt.close()

    plot_waterfall(correct_idx, "Correct")
    plot_waterfall(incorrect_idx, "Incorrect")

    # --- 3. DEPENDENCE PLOT ---
    print("Generating Dependence Plots...")
    delta_idx = feature_names.index('Delta')
    theta_idx = feature_names.index('Theta')
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(delta_idx, shap_feat_raw_agg, feat_agg, feature_names=feature_names, interaction_index=theta_idx, show=False)
    plt.title("SHAP Dependence Plot: Delta vs Theta")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_dependence.png'))
    plt.close()

    # --- 4. COMPARATIVE ANALYSIS (SHAP vs ATTENTION) ---
    print("Generating Comparative Analysis...")
    attn_temporal = xai['temp_feat'].norm(dim=-1).cpu().numpy() # Shape: (B, S)
    shap_temporal = np.abs(shap_sig).mean(axis=(2, 3, 4)) # Shape: (B, S)

    correlations = []
    for i in range(B):
        # Handle zero variance or NaN issues
        if np.var(attn_temporal[i]) == 0 or np.var(shap_temporal[i]) == 0:
            continue
        corr, _ = pearsonr(attn_temporal[i], shap_temporal[i])
        if not np.isnan(corr):
            correlations.append(corr)

    avg_corr = np.nanmean(correlations) if len(correlations) > 0 else 0.0

    with open(os.path.join(out_dir, 'quantitative_metrics.txt'), 'w') as f:
        f.write("=== Quantitative XAI Metrics ===\n")
        f.write(f"Mean |SHAP| per feature:\n")
        mean_abs_shap = np.abs(shap_feat_raw_agg).mean(axis=0)
        for name, val in zip(feature_names, mean_abs_shap):
            f.write(f"  {name}: {val:.5f}\n")
        f.write(f"\nComparative Analysis (SHAP Temporal vs Attention Temporal):\n")
        f.write(f"  Mean Pearson Correlation across {len(correlations)} valid sequences: {avg_corr:.4f}\n")
        f.write(f"  (High correlation means SHAP signal importance aligns with Transformer Attention)\n")

    # Plot overlay for the first correct sample
    if len(correct_idx) > 0:
        i = correct_idx[0]
        sig_1d = test_sig[i].cpu().numpy().reshape(-1) # flattened
        s_len = len(shap_temporal[i])
        attn_up = np.repeat(attn_temporal[i], len(sig_1d) // s_len)
        shap_up = np.repeat(shap_temporal[i], len(sig_1d) // s_len)
        
        # pad remaining if length mismatch
        rem = len(sig_1d) - len(attn_up)
        if rem > 0:
            attn_up = np.pad(attn_up, (0, rem), 'edge')
            shap_up = np.pad(shap_up, (0, rem), 'edge')
        elif rem < 0:
            attn_up = attn_up[:len(sig_1d)]
            shap_up = shap_up[:len(sig_1d)]
            
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(sig_1d, color='black', alpha=0.5, label='EEG Signal')
        ax2 = ax1.twinx()
        ax2.plot(attn_up, color='orange', alpha=0.8, label='Attention Saliency', linestyle='--')
        ax2.plot(shap_up, color='purple', alpha=0.8, label='SHAP Saliency')
        
        # Add legend combining both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        
        plt.title("Comparative Overlay: Signal vs Attention vs SHAP")
        plt.savefig(os.path.join(out_dir, 'comparative_overlay.png'))
        plt.close()

    print("All paper figures generated in:", out_dir)
