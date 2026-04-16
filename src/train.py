import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
import os
import logging
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix

# Global Constants
OUTPUTS_DIR = 'outputs'
CHBMIT_DIR = r'c:\Users\HP\8th sem\chbmit\physionet.org\files\chbmit\1.0.0'
SEIZEIT2_DIR = r'c:\Users\HP\8th sem\seizeit2'

logger = logging.getLogger(__name__)

# --- UTILITY CLASSES & FUNCTIONS ---

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach())
    def apply_shadow(self):
        self.backup = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])
    def restore(self):
        for name, param in self.model.named_parameters():
            param.data.copy_(self.backup[name])

def calibrate_distribution(logits):
    """Standardizes logits with a stability floor and magnitude clamp."""
    logits = torch.clamp(logits, min=-10.0, max=10.0) 
    mean = logits.mean()
    std = logits.std() + 1e-4
    return (logits - mean) / std

def check_signal_direction(logits, labels):
    """Returns True if seizure mean > background mean."""
    pos_mask = (labels == 1); neg_mask = (labels == 0)
    if pos_mask.sum() == 0 or neg_mask.sum() == 0: return True
    return logits[pos_mask].mean() > logits[neg_mask].mean()

def find_best_threshold(probs, labels):
    """Quantile-aware threshold search for clinical metrics."""
    q_start = np.quantile(probs, 0.05); q_end = np.quantile(probs, 0.95)
    thresholds = np.linspace(q_start, q_end, 50)
    best_f1 = -1; best_t = (q_start + q_end) / 2
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        spec = recall_score(labels, preds, pos_label=0, zero_division=0)
        if f > best_f1 and rec > 0.3 and spec > 0.3:
            best_f1 = f; best_t = t
    return best_t

def compute_entropy(logits):
    probs = torch.sigmoid(logits)
    entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
    return entropy.mean()

def compute_coral_loss(source, target):
    """Correlation Alignment (CORAL) Loss with numerical stability guards."""
    if source.size(0) <= 1 or target.size(0) <= 1:
        return torch.tensor(0.0).to(source.device)
        
    d = source.size(1)
    # Source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm) / (source.size(0) - 1)
    # Target covariance
    ym = torch.mean(target, 0, keepdim=True) - target
    yc = torch.matmul(torch.transpose(ym, 0, 1), ym) / (target.size(0) - 1)
    
    # Stability: Clamp covariance values to prevent explosion
    loss = torch.mean(torch.mul(torch.clamp(xc - yc, -5.0, 5.0), torch.clamp(xc - yc, -5.0, 5.0)))
    return loss

# --- LOSS FUNCTIONS ---

def compute_margin_ranking_loss(logits, labels, margin=3.0):
    pos_mask = (labels == 1); neg_mask = (labels == 0)
    if pos_mask.sum() == 0 or neg_mask.sum() == 0: return torch.tensor(0.0).to(logits.device)
    pos_logits = logits[pos_mask].unsqueeze(1)
    neg_logits = logits[neg_mask].unsqueeze(0)
    loss = torch.clamp(margin - (pos_logits - neg_logits), min=0.0).mean()
    return loss

def apply_dual_anchoring(logits, labels, weight=1.0):
    """Enforces Huber-based anchors at +/- 1.5."""
    if weight == 0: return torch.tensor(0.0).to(logits.device)
    pos_mask = (labels == 1); neg_mask = (labels == 0)
    pos_loss = F.smooth_l1_loss(logits[pos_mask], torch.full_like(logits[pos_mask], 1.5)) if pos_mask.any() else 0
    neg_loss = F.smooth_l1_loss(logits[neg_mask], torch.full_like(logits[neg_mask], -1.5)) if neg_mask.any() else 0
    return weight * (pos_loss + neg_loss)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# --- MAIN TRAINING PIPELINE ---

def train_model(model_name, train_dataset_type='CHB', test_dataset_type='CHB', num_epochs=5, batch_size=32, device='cpu'):
    from data_loader import load_dataset
    from dataset_builder import get_dataloaders, get_cross_dataset_loaders
    from segmentation import generate_window_metadata
    from labeling import label_windows
    from models.cnn_swin_transformer import CNNSwinTransformerModel
    from autoencoder_reduction import FeatureAutoencoder

    logger.info(f"Trajectory Lock: Train on {train_dataset_type}, Test on {test_dataset_type}")
    is_cross = (train_dataset_type != test_dataset_type)
    exp_name = f"{train_dataset_type}_to_{test_dataset_type}"
    
    source_records = load_dataset(train_dataset_type)
    source_windows = label_windows(generate_window_metadata(source_records))
    target_records = load_dataset(test_dataset_type)
    target_windows = label_windows(generate_window_metadata(target_records))

    if is_cross:
        train_loader, val_loader, test_loader, train_windows = get_cross_dataset_loaders(source_windows, target_windows, batch_size=batch_size)
    else:
        train_loader, val_loader, test_loader, train_windows = get_dataloaders(source_windows, batch_size=batch_size)

    # Model & Optimization Initialization
    sig_sample, feat_sample, _ = next(iter(test_loader))
    eeg_channels = sig_sample.shape[2]
    input_feat_dim = feat_sample.shape[2]
    latent_dim = 64
    
    autoencoder = FeatureAutoencoder(input_feat_dim, latent_dim).to(device)
    ae_path = os.path.join('outputs', 'saved_models', f'ae_{train_dataset_type}.pth')
    if os.path.exists(ae_path):
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
        logger.info(f"Autoencoder strictly loaded from: {ae_path}")
    else:
        logger.warning(f"CRITICAL: Autoencoder not found at {ae_path}. Using random untrained projections.")
        
    autoencoder.eval()
    for p in autoencoder.parameters(): p.requires_grad = False
    
    model = CNNSwinTransformerModel(eeg_channels, latent_dim=latent_dim).to(device)
    ema = EMA(model) 
    
    # STANDARD OPTIMIZATION STARTING A BIT HIGHER TO PREVENT EARLY UNDERFITTING
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-2)
    criterion = FocalLoss()
    
    # Balanced DataLoader
    train_labels = [w[-1]['label'] for w in train_windows]
    counts = np.bincount(train_labels)
    sample_w = [1.0/(counts[l]+1e-8) for l in train_labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    train_loader_balanced = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=sampler, num_workers=2, persistent_workers=True)
    
    best_val_auc = 0.0
    total_steps = num_epochs * len(train_loader_balanced)
    current_step = 0
    target_loader = val_loader if is_cross else None
    target_iter = iter(target_loader) if is_cross else None
    current_alpha_damp = 1.0
    last_loss_d_tgt = 0.69

    # 3. Training Curriculum
    for epoch in range(num_epochs):
        # STAGE SELECTION: 0=Orientation, 1=LOCKDOWN (Alignment), 2=Refinement
        stage = 0 if epoch == 0 else (1 if epoch == 1 else 2)
        
        # Domain-Conditional Constants
        c_dann = 0.0 if (stage == 0 or not is_cross) else 0.5 
        c_rank = 0.1 if stage == 0 else (0.0 if stage == 1 else 1.0)
        
        # STAGE-AWARE LR: Start higher, taper later
        current_lr = 2e-4 if stage == 0 else (5e-5 if stage == 1 else 1e-5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        model.train()
        for batch_idx, (sigs_src, feats_src, labels_src) in enumerate(tqdm(train_loader_balanced, desc=f"Stage {stage} Ep {epoch+1}")):
            sigs_src, feats_src, labels_src = sigs_src.to(device), feats_src.to(device), labels_src.to(device)
            labels_f = labels_src.float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward Source (Raw Logits with channel safety)
            l_src, f_src = model(sigs_src, autoencoder.encode(feats_src), return_features=True)
            if l_src.shape[1] > 1: l_src = l_src[:, 1:2]
            l_src_s = l_src
            
            # POLARITY GUARD: Ensure Seizure > Background
            loss = criterion(l_src_s, labels_f)
            
            if c_rank > 0:
                loss += (c_rank * compute_margin_ranking_loss(l_src_s, labels_src, margin=2.0))
                loss += apply_dual_anchoring(l_src_s, labels_src, weight=c_rank)
            
            if is_cross and c_dann > 0:
                try: s_tgt, f_tgt, _ = next(target_iter)
                except StopIteration: target_iter = iter(target_loader); s_tgt, f_tgt, _ = next(target_iter)
                
                l_tgt, f_tgt = model(s_tgt.to(device), autoencoder.encode(f_tgt.to(device)), return_features=True)
                if l_tgt.shape[1] > 1: l_tgt = l_tgt[:, 1:2]
                l_tgt_s = torch.clamp(l_tgt / 2.0, -10.0, 10.0)
                
                # EQUILIBRIUM GUARD: Monitor gaps
                p = float(current_step) / total_steps
                base_alpha = (2. / (1. + np.exp(-1.0 * (3.0 if stage == 1 else 1.5) * p)) - 1.0) * c_dann 
                
                # ADVERSARIAL SAFETY BRAKE:
                # If the classification gap on source data has collapsed (Discriminator is too strong), 
                # we force alpha to zero to let the feature extractor recover its seizure-detection capability.
                p_s = torch.sigmoid(l_src_s)
                current_gap = p_s[labels_src==1].mean() - p_s[labels_src==0].mean()
                
                alpha = base_alpha
                if current_gap < 0.1 and stage > 0:
                    alpha = 0.0
                    if batch_idx % 20 == 0: logger.warning(f"Adversarial Brake Applied | Gap: {current_gap.item():.4f}")
                    
                if not check_signal_direction(l_src_s, labels_src) and stage > 0: alpha = 0.0
                
                # Leverage the strong zero-shot baseline (Epoch 1) to condition the domain alignment!
                prob_src = torch.sigmoid(l_src_s).detach()
                prob_tgt = torch.sigmoid(l_tgt_s).detach()
                f_src_c = f_src * (0.5 + prob_src)
                f_tgt_c = f_tgt * (0.5 + prob_tgt)
                
                d_src = model.forward_domain(f_src_c, alpha=alpha)
                d_tgt = model.forward_domain(f_tgt_c, alpha=alpha)
                
                loss_d_src = F.binary_cross_entropy_with_logits(d_src[:,1], torch.full((d_src.size(0),), 0.1).to(device))
                loss_d_tgt = F.binary_cross_entropy_with_logits(d_tgt[:,1], torch.full((d_tgt.size(0),), 0.9).to(device))
                
                last_loss_d_tgt = loss_d_tgt.item()
                
                loss += (0.5 * (loss_d_src + loss_d_tgt))
                loss += 0.1 * compute_entropy(l_tgt_s) 
                
                if batch_idx % 100 == 0:
                    logger.info(f"Lockdown Audit | Div: {loss_d_tgt.item():.4f} | Alpha: {alpha:.3f}")
                    
            # L1 Structural Sparsity: Erode dead spatial filters (solves SEIZE->CHB random weight collision)
            if hasattr(model, 'cnn') and hasattr(model.cnn, 'conv_block1'):
                l1_penalty = torch.norm(model.cnn.conv_block1[0].weight, p=1)
                loss += 0.002 * l1_penalty
                
            # --- FINAL LOSS STABILITY CHECK ---
            if not torch.isfinite(loss):
                logger.warning(f"NaN Loss detected at step {current_step}. Skipping batch.")
                optimizer.zero_grad()
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            ema.update()
            current_step += 1
            
            if batch_idx == 0:
                p_s = torch.sigmoid(l_src_s)
                gap = p_s[labels_src==1].mean() - p_s[labels_src==0].mean()
                print(f"\n[AUDIT] Stage {stage} | Prob Gap: {gap.item():.4f}")

        # 4. Validation (Ensemble EMA Consensus)
        model.eval()
        v_probs, v_y = [], []
        with torch.no_grad():
            # Consensus: EMA weights
            ema.apply_shadow()
            for s, f, l in val_loader:
                out_ema = model(s.to(device), autoencoder.encode(f.to(device)))
                if out_ema.shape[1] > 1: out_ema = out_ema[:, 1:2]
                v_probs.append(np.nan_to_num(torch.sigmoid(out_ema).cpu().numpy(), nan=0.5))
                v_y.append(l.cpu().numpy())
            
            v_probs, v_y = np.concatenate(v_probs), np.concatenate(v_y)
            val_auc = roc_auc_score(v_y, v_probs)
            thresh = find_best_threshold(v_probs, v_y)
            logger.info(f"Epoch {epoch+1} Results | EMA AUC: {val_auc:.4f} | Thresh: {thresh:.4f}")
            
            # SAVE ONLY BEST EMA WEIGHTS
            if val_auc > best_val_auc + 0.005:
                best_val_auc = val_auc
                save_dict = {
                    'model_state_dict': model.state_dict(), 
                    'thresh': thresh, 
                    'auc': val_auc,
                    'eeg_channels': eeg_channels,
                    'feature_dim': input_feat_dim,
                    'autoencoder_state_dict': autoencoder.state_dict()
                }
                torch.save(save_dict, os.path.join(OUTPUTS_DIR, 'saved_models', f'best_{model_name}_{exp_name}.pth'))
                logger.info(f"New BEST AUC: {val_auc:.4f}")
            
            # NOW restore weights for the next training epoch
            ema.restore()
        
    return model
