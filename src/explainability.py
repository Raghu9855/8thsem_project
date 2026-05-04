import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import shap

# Import project-specific modules
from utils import OUTPUTS_DIR, set_seed, SEIZEIT2_DIR, CHBMIT_DIR
from dataset_builder import get_dataloaders
from data_loader import get_seizeit2_records, get_chbmit_records
from segmentation import generate_window_metadata
from labeling import label_windows
from autoencoder_reduction import FeatureAutoencoder
from models.cnn_swin_transformer import CNNSwinTransformerModel
from models.cnn_lstm import CNNLSTMModel
from models.cnn_gnn import CNNGNNModel

class UltimateXAIReseacher:
    """
    Consolidated, research-grade XAI pipeline for the CNN-Swin Transformer model.
    Focuses on interpreting existing trained models for CHB and SEIZE datasets.
    """
    def __init__(self, model_name, exp_name, device='cpu'):
        self.device = torch.device(device)
        self.exp_name = exp_name
        self.dataset_name = exp_name.split('_to_')[-1]
        self.model_name = model_name
        self.output_dir = os.path.join(OUTPUTS_DIR, f'xai_results_{model_name}_{exp_name}')
        os.makedirs(self.output_dir, exist_ok=True)
        set_seed(42)
        
        model_path = os.path.join(OUTPUTS_DIR, 'saved_models', f'best_{model_name}_{exp_name}.pth')
        print(f"\n[PHASE 1] Analyzing Model: {os.path.basename(model_path)} for {self.exp_name}")
        
        # Load Architecture & Weights
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.autoencoder = FeatureAutoencoder(input_dim=self.checkpoint['feature_dim']).to(self.device)
        self.autoencoder.load_state_dict(self.checkpoint['autoencoder_state_dict'])
        self.autoencoder.eval()
        
        if model_name == 'cnn_lstm':
            self.model = CNNLSTMModel(eeg_channels=self.checkpoint['eeg_channels']).to(self.device)
        elif model_name == 'cnn_gnn':
            self.model = CNNGNNModel(eeg_channels=self.checkpoint['eeg_channels']).to(self.device)
        else:
            self.model = CNNSwinTransformerModel(eeg_channels=self.checkpoint['eeg_channels']).to(self.device)
            
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        self._load_data()

    def _load_data(self):
        """Prepare domain-specific test data."""
        if self.dataset_name == "CHB":
            records = get_chbmit_records(CHBMIT_DIR)
        else:
            records = get_seizeit2_records(SEIZEIT2_DIR)
        
        # Take a subset if necessary, but here we use the test loader logic
        window_metadata = generate_window_metadata(records, window_size_sec=5.0)
        labeled_windows = label_windows(window_metadata)
        _, _, self.test_loader, _ = get_dataloaders(labeled_windows, batch_size=32)
        
        # Sample for visualization
        self.sigs, self.feats, self.labels = next(iter(self.test_loader))
        self.sigs, self.feats = self.sigs.to(self.device), self.feats.to(self.device)

    # --- 1. PERFORMANCE AUDIT ---
    def run_performance_audit(self):
        print("Running Performance Metrics Audit...")
        all_probs, all_labels = [], []
        with torch.no_grad():
            for s, f, l in tqdm(self.test_loader, desc=f"Evaluating {self.dataset_name}"):
                s, f = s.to(self.device), f.to(self.device)
                lat = self.autoencoder.encode(f)
                logits = self.model(s, lat)
                if logits.shape[1] > 1: logits_cls1 = logits[:, 1:2]
                else: logits_cls1 = logits
                probs = torch.sigmoid(logits_cls1).squeeze(-1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(l.numpy())
        
        all_probs, all_labels = np.array(all_probs), np.array(all_labels)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        optimal_idx = np.argmax(tpr - fpr)
        self.optimal_thresh = thresholds[optimal_idx]
        preds = (all_probs >= self.optimal_thresh).astype(int)
        
        # Metrics
        acc = accuracy_score(all_labels, preds)
        rec = recall_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f"Results [{self.dataset_name}] - Acc: {acc:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {self.dataset_name}")
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC Curve - {self.dataset_name}")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()

    # --- 2. ATTENTION INTERPRETABILITY ---
    def run_attention_analysis(self, sample_idx=0):
        print(f"Generating Attention Maps for Sample {sample_idx}...")
        sig = self.sigs[sample_idx:sample_idx+1]
        feat = self.feats[sample_idx:sample_idx+1]
        
        with torch.no_grad():
            lat = self.autoencoder.encode(feat)
            _, xai = self.model(sig, lat, xai_mode=True)
            
        if xai.get('attn1') is None:
            print(f"Skipping detailed attention plots (not supported by {self.model_name}).")
            return
        
        # Layer & Head Wise Map (using Swin Block 1)
        attn1 = xai['attn1'][0].cpu().numpy() # (H, W, W)
        num_heads = attn1.shape[0]
        
        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
        for h in range(num_heads):
            sns.heatmap(attn1[h], ax=axes[h], cmap='viridis', cbar=False)
            axes[h].set_title(f"Head {h}")
        plt.suptitle(f"Layer 1 Attention Maps - {self.dataset_name}")
        plt.savefig(os.path.join(self.output_dir, 'attention_layerwise.png'))
        plt.close()

        # Attention Rollout (Simple implementation for 2 Swin blocks)
        a1 = xai['attn1'].mean(dim=1) # (B*S, W, W)
        a2 = xai['attn2'].mean(dim=1)
        rollout = torch.matmul(a2, a1)[0].cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(rollout, cmap='magma')
        plt.title("Attention Rollout (Aggregated Importance)")
        plt.savefig(os.path.join(self.output_dir, 'attention_rollout.png'))
        plt.close()

    # --- 3. FAITHFULNESS (Deletion/Insertion) ---
    def run_faithfulness_test(self, sample_idx=0):
        print("Evaluating Explanation Faithfulness...")
        sig = self.sigs[sample_idx:sample_idx+1]
        feat = self.feats[sample_idx:sample_idx+1]
        
        with torch.no_grad():
            lat = self.autoencoder.encode(feat)
            logits, xai = self.model(sig, lat, xai_mode=True)
            saliency = torch.norm(xai['temp_feat'][0], dim=1).cpu().numpy()
            base_prob = torch.softmax(logits, dim=1)[0, 1].item()
            
        indices = np.argsort(saliency)[::-1]
        del_probs = [base_prob]
        ins_probs = []
        
        # Deletion
        for k in range(1, len(saliency)+1):
            m_sig = sig.clone(); m_feat = feat.clone()
            for i in range(k):
                m_sig[0, indices[i]] = 0; m_feat[0, indices[i]] = 0
            with torch.no_grad():
                l_m = self.autoencoder.encode(m_feat)
                p = torch.softmax(self.model(m_sig, l_m), dim=1)[0, 1].item()
                del_probs.append(p)
                
        # Insertion
        m_sig = torch.zeros_like(sig); m_feat = torch.zeros_like(feat)
        with torch.no_grad():
            p_initial = torch.softmax(self.model(m_sig, self.autoencoder.encode(m_feat)), dim=1)[0, 1].item()
            ins_probs.append(p_initial)
            
        for k in range(1, len(saliency)+1):
            m_sig[0, indices[k-1]] = sig[0, indices[k-1]]
            m_feat[0, indices[k-1]] = feat[0, indices[k-1]]
            with torch.no_grad():
                p = torch.softmax(self.model(m_sig, self.autoencoder.encode(m_feat)), dim=1)[0, 1].item()
                ins_probs.append(p)
                
        plt.figure(figsize=(10, 5))
        plt.plot(del_probs, label='Deletion (Drop)', color='red', marker='o')
        plt.plot(ins_probs, label='Insertion (Gain)', color='green', marker='s')
        plt.title("Faithfulness Audit: Perturbation Curves")
        plt.xlabel("Segments Perturbed")
        plt.ylabel("Probability")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'faithfulness_curve.png'))
        plt.close()

    # --- 4. ERROR ANALYSIS ---
    def run_error_analysis(self, num_cases=3):
        print("Identifying Error Cases for Deep Dive...")
        errors = []
        with torch.no_grad():
            for s, f, l in self.test_loader:
                s, f = s.to(self.device), f.to(self.device)
                lat = self.autoencoder.encode(f)
                logits, xai = self.model(s, lat, xai_mode=True)
                if logits.shape[1] > 1: logits_cls1 = logits[:, 1:2]
                else: logits_cls1 = logits
                probs = torch.sigmoid(logits_cls1).squeeze(-1).cpu().numpy()
                
                thresh_to_use = getattr(self, 'optimal_thresh', 0.5)
                preds = (probs >= thresh_to_use).astype(int)
                
                for i in range(len(l)):
                    if preds[i] != l[i].item():
                        errors.append({
                            'sig': s[i], 'pred': preds[i], 'true': l[i].item(), 
                            'prob': probs[i], 'saliency': torch.norm(xai['temp_feat'][i], dim=1).cpu().numpy()
                        })
                    if len(errors) >= num_cases: break
                if len(errors) >= num_cases: break
        
        for idx, err in enumerate(errors):
            plt.figure(figsize=(12, 6))
            signal = err['sig'].view(err['sig'].size(0), -1).mean(dim=1).cpu().numpy()
            plt.subplot(2, 1, 1)
            plt.plot(signal, color='black', alpha=0.7)
            plt.title(f"Error Case {idx}: True={err['true']}, Pred={err['pred']} (Prob={err['prob']:.2f})")
            plt.subplot(2, 1, 2)
            plt.fill_between(range(len(err['saliency'])), err['saliency'], color='orange', alpha=0.5)
            plt.title("Attention Saliency Overlay")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'error_analysis_{idx}.png'))
            plt.close()

    # --- 5. STABILITY ---
    def run_stability_check(self, sample_idx=0):
        print("Checking Explanation Stability under Noise...")
        sig = self.sigs[sample_idx:sample_idx+1]
        feat = self.feats[sample_idx:sample_idx+1]
        
        with torch.no_grad():
            lat = self.autoencoder.encode(feat)
            _, xai_orig = self.model(sig, lat, xai_mode=True)
            s_orig = torch.norm(xai_orig['temp_feat'][0], dim=1).cpu().numpy()
            
            # Add Gaussian Noise (5% of signal range)
            noisy_sig = sig + torch.randn_like(sig) * 0.05
            _, xai_noisy = self.model(noisy_sig, lat, xai_mode=True)
            s_noisy = torch.norm(xai_noisy['temp_feat'][0], dim=1).cpu().numpy()
            
        stab_corr, _ = pearsonr(s_orig, s_noisy)
        print(f"Stability Correlation: {stab_corr:.4f}")
        with open(os.path.join(self.output_dir, 'stability_report.txt'), 'w') as f:
            f.write(f"Explanation Stability Correlation (Pearson): {stab_corr:.4f}\n")

    # --- 6. SHAP ANALYSIS ---
    def run_shap_analysis(self, sample_idx=0):
        print("Running SHAP Analysis...")
        
        class SHAPWrapper(nn.Module):
            def __init__(self, model, autoencoder):
                super().__init__()
                self.model = model
                self.autoencoder = autoencoder
            def forward(self, sig, feat):
                lat = self.autoencoder.encode(feat)
                # Return only the logit for class 1 (seizure)
                return self.model(sig, lat)[:, 1:2]
                
        wrapped_model = SHAPWrapper(self.model, self.autoencoder).to(self.device)
        wrapped_model.eval()

        # Background data for SHAP (use a small batch to save memory/compute)
        bg_sigs, bg_feats = [], []
        for i, (s, f, _) in enumerate(self.test_loader):
            bg_sigs.append(s)
            bg_feats.append(f)
            if i >= 0: # just 1 batch (32 samples) is usually enough for GradientExplainer
                break
        bg_sig = torch.cat(bg_sigs, dim=0).to(self.device)
        bg_feat = torch.cat(bg_feats, dim=0).to(self.device)

        test_sig = self.sigs[sample_idx:sample_idx+1].to(self.device)
        test_feat = self.feats[sample_idx:sample_idx+1].to(self.device)

        # GradientExplainer for PyTorch models
        explainer = shap.GradientExplainer(wrapped_model, [bg_sig, bg_feat])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values([test_sig, test_feat])
        
        # shap_values is a list of arrays corresponding to the inputs
        # For single output (class 1 logit), shap_values will be a list [shap_sig, shap_feat]
        if isinstance(shap_values, list) and isinstance(shap_values[0], list):
            # If it's still multi-class format for some reason
            shap_sig = shap_values[0][0]
        elif isinstance(shap_values, list):
            shap_sig = shap_values[0]
        else:
            shap_sig = shap_values

        # shap_sig shape: (1, S, C, F, T)
        # Aggregate over Channels, FreqBins, TimeSteps to get importance per Sequence Segment
        saliency_seq = np.abs(shap_sig[0]).mean(axis=(1, 2, 3))
        
        plt.figure(figsize=(10, 4))
        plt.plot(saliency_seq, marker='o', color='purple')
        plt.title(f"SHAP Feature Importance (EEG Signal) - Sample {sample_idx}")
        plt.xlabel("Sequence Segment")
        plt.ylabel("Mean Absolute SHAP Value")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'shap_importance.png'))
        plt.close()
        print("SHAP Analysis complete.")

    def run_all(self):
        self.run_performance_audit()
        self.run_attention_analysis()
        self.run_faithfulness_test()
        self.run_error_analysis()
        self.run_stability_check()
        self.run_shap_analysis()
        print(f"DONE: Research XAI Results for {self.dataset_name} saved in {self.output_dir}")

def run_multi_dataset_research():
    experiments = [
        "CHB_to_CHB",
        "CHB_to_SEIZE",
        "SEIZE_to_CHB",
        "SEIZE_to_SEIZE"
    ]
    models = ['cnn_swin', 'cnn_lstm', 'cnn_gnn']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for m in models:
        for exp in experiments:
            model_path = os.path.join('outputs', 'saved_models', f'best_{m}_{exp}.pth')
            if os.path.exists(model_path):
                researcher = UltimateXAIReseacher(m, exp, device=device)
                try:
                    researcher.run_all()
                except Exception as e:
                    print(f"Failed XAI for {m} {exp}: {e}")

if __name__ == "__main__":
    run_multi_dataset_research()
