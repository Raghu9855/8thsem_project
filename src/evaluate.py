import os
import argparse
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import OUTPUTS_DIR, setup_logger, set_seed
from segmentation import generate_window_metadata
from labeling import label_windows
from dataset_builder import get_dataloaders
from autoencoder_reduction import FeatureAutoencoder
from models.cnn_lstm import CNNLSTMModel
from models.cnn_swin_transformer import CNNSwinTransformerModel
from models.cnn_gnn import CNNGNNModel

def evaluate_model(model_name, exp_name, device='cpu'):
    logger = setup_logger(f'eval_{model_name}_{exp_name}', os.path.join(OUTPUTS_DIR, 'reports', f'eval_{model_name}_{exp_name}.log'))
    device = torch.device(device)
    set_seed(42)
    
    # Load dataset
    logger.info(f"Initializing Test Dataloader for {exp_name}...")
    source, target = exp_name.split("_to_")
    
    # Load weights first
    save_path = os.path.join(OUTPUTS_DIR, 'saved_models', f"best_{model_name}_{exp_name}.pth")
    if not os.path.exists(save_path):
        logger.error(f"Cannot find weights at {save_path}")
        return
        
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    
    # Strictly load the TARGET evaluation subset to avoid cross-domain testing overlap
    from data_loader import load_dataset
    if target == 'CHB': test_records = load_dataset('CHB')
    else: test_records = load_dataset('SEIZE')
    
    window_metadata = generate_window_metadata(test_records, window_size_sec=5.0, overlap_ratio=0.5)
    labeled_windows = label_windows(window_metadata)
    
    # Split validation exactly as train.py does for target evaluation
    from dataset_builder import get_dataloaders
    test_loader = get_dataloaders(labeled_windows, batch_size=32, num_workers=0)[2] 
        
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    eeg_channels = checkpoint['eeg_channels']
    feature_dim = checkpoint['feature_dim']
    
    # Init Models
    autoencoder = FeatureAutoencoder(input_dim=feature_dim)
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    autoencoder.eval()
    autoencoder.to(device)
    
    if model_name == 'cnn_lstm':
        model = CNNLSTMModel(eeg_channels=eeg_channels)
    elif model_name == 'cnn_swin':
        model = CNNSwinTransformerModel(eeg_channels=eeg_channels)
    elif model_name == 'cnn_gnn':
        model = CNNGNNModel(eeg_channels=eeg_channels)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for sigs, feats, labels in test_loader:
            sigs, feats, labels = sigs.to(device), feats.to(device), labels.to(device)
            latent_feats = autoencoder.encode(feats)
            outputs = model(sigs, latent_feats)
            
            if outputs.shape[1] > 1: outputs = outputs[:, 1:2]
            probs = torch.sigmoid(outputs).squeeze(-1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    inference_time = time.time() - start_time
    avg_inference_ms = (inference_time / len(test_loader.dataset)) * 1000
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate mathematically optimal threshold on the evaluated subset (Youden's J statistic)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    all_preds = (all_probs >= optimal_threshold).astype(int)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0) # Sensitivity
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    try:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        specificity = 0.0
        
    logger.info(f"Results for {model_name}:")
    logger.info(f"Accuracy:    {acc:.4f}")
    logger.info(f"Precision:   {prec:.4f}")
    logger.info(f"Recall:      {rec:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"F1-Score:    {f1:.4f}")
    logger.info(f"ROC-AUC:     {roc_auc:.4f}")
    logger.info(f"Inference ms/window: {avg_inference_ms:.2f} ms")
    
    # Save Confusion Matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} ({exp_name})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'cm_{model_name}_{exp_name}.png'))
    plt.close()

    # Save ROC Curve plot
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name} ({exp_name})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', f'roc_{model_name}_{exp_name}.png'))
    plt.close()
    
    return {
        'Experiment': exp_name,
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Sensitivity (Recall)': rec,
        'Specificity': specificity,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Inference (ms)': avg_inference_ms
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['cnn_swin'])
    args = parser.parse_args()
    
    experiments = [
        "CHB_to_CHB",
        "CHB_to_SEIZE",
        "SEIZE_to_CHB",
        "SEIZE_to_SEIZE"
    ]
    
    results = []
    for m in args.models:
        for exp in experiments:
            res = evaluate_model(m, exp_name=exp, device='cpu')
            if res:
                results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(OUTPUTS_DIR, 'reports', 'final_metrics_comparison.csv'), index=False)
        print("\n--- Final Model Metrics Synthesis ---")
        print(df.to_string(index=False))

        # Master Comparison Chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Experiment', y='ROC-AUC', hue='Model', palette='viridis')
        plt.axhline(0.60, color='red', linestyle='--', alpha=0.6, label='Cross-Domain Goal (0.60)')
        plt.axhline(0.80, color='green', linestyle='--', alpha=0.6, label='Intra-Domain Goal (0.80)')
        plt.title('Final Model Performance Comparison (ROC-AUC)')
        plt.ylim(0.4, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', 'master_performance_comparison.png'))
        plt.close()
        print(f"\n[SUCCESS] Master performance chart saved to: outputs/plots/master_performance_comparison.png")
