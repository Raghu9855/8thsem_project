import torch
from torch.utils.data import Dataset, DataLoader
import mne
import numpy as np
import scipy.signal as signal
import random
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from preprocessing import preprocess_eeg_window
from augmentation import apply_transforms
from feature_extraction import extract_features
from functools import lru_cache
from tqdm import tqdm

@lru_cache(maxsize=100)
def get_cached_raw_edf(edf_path):
    return mne.io.read_raw_edf(edf_path, preload=False, verbose='error')

class EEGDataset(Dataset):
    def __init__(self, sequences, is_train=False, augment_prob=0.5):
        """
        sequences: List of lists of window dicts. Each sequence has seq_len windows.
        """
        self.sequences = sequences
        self.is_train = is_train
        self.augment_prob = augment_prob
        # Weak caching for disk streaming
        self.current_edf_path = None
        self.current_raw = None
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq_windows = self.sequences[idx]
        
        all_sig = []
        all_feat = []
        # The label for the sequence is the label of the LAST window
        final_label = seq_windows[-1]['label']
        
        for win_info in seq_windows:
            record = win_info['record']
            edf_path = record['path']
            start_sec = win_info['start']
            end_sec = win_info['end']
            label = win_info['label']
            
            raw = get_cached_raw_edf(edf_path)
            sfreq = raw.info['sfreq']
            start_samp = int(start_sec * sfreq)
            end_samp = int(end_sec * sfreq)
            
            data, _ = raw[:, start_samp:end_samp]
            
            # --- Robust per-window normalization ---
            # Use a slightly larger epsilon for better CPU stability
            data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
            data = np.nan_to_num(data) # Catch any division errors
            
            data = preprocess_eeg_window(data, sfreq=sfreq, target_sfreq=256.0)
            
            if self.is_train and label == 1 and np.random.rand() < self.augment_prob:
                data = apply_transforms(data, sfreq=256.0)
                
            features = extract_features(data, sfreq=256.0)
            
            # Pad/Standardize channels
            MAX_CHANNELS = 23
            FEATS_PER_CHANNEL = 12
            if data.shape[0] < MAX_CHANNELS:
                data = np.pad(data, ((0, MAX_CHANNELS - data.shape[0]), (0, 0)), mode='constant')
            elif data.shape[0] > MAX_CHANNELS:
                data = data[:MAX_CHANNELS, :]
                
            if len(features) < MAX_CHANNELS * FEATS_PER_CHANNEL:
                features = np.pad(features, (0, MAX_CHANNELS * FEATS_PER_CHANNEL - len(features)), mode='constant')
            elif len(features) > MAX_CHANNELS * FEATS_PER_CHANNEL:
                features = features[:MAX_CHANNELS * FEATS_PER_CHANNEL]
            
            f, t, Sxx = signal.spectrogram(data, fs=256.0, nperseg=64, noverlap=32)
            Sxx = Sxx[:, f <= 40.0, :]
            
            # --- Log-power stabilization ---
            Sxx = np.log(Sxx + 1e-8)
            
            # Final standardization with safety gate (per-channel to prevent padded channels from distorting the whole tensor)
            Sxx = (Sxx - Sxx.mean(axis=(1, 2), keepdims=True)) / (Sxx.std(axis=(1, 2), keepdims=True) + 1e-8)
            Sxx = np.nan_to_num(Sxx)
            
            all_sig.append(torch.tensor(Sxx, dtype=torch.float32))
            all_feat.append(torch.tensor(np.nan_to_num(features), dtype=torch.float32))
            
        return torch.stack(all_sig), torch.stack(all_feat), torch.tensor(final_label, dtype=torch.long)

class CachedEEGDataset(Dataset):
    def __init__(self, sequences, cache_name=None, is_train=False):
        self.sequences = sequences
        self.is_train = is_train
        
        # Load or create cache
        self.cached_sigs = None
        self.cached_feats = None
        self.cached_labels = None
        
        if cache_name:
            cache_dir = os.path.join('outputs', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            sig_path = os.path.join(cache_dir, f"{cache_name}_sigs.pt")
            feat_path = os.path.join(cache_dir, f"{cache_name}_feats.pt")
            label_path = os.path.join(cache_dir, f"{cache_name}_labels.pt")
            
            if os.path.exists(sig_path) and os.path.exists(feat_path):
                print(f"Loading {cache_name} from cache...")
                self.cached_sigs = torch.load(sig_path, weights_only=False)
                self.cached_feats = torch.load(feat_path, weights_only=False)
                self.cached_labels = torch.load(label_path, weights_only=False)
                
                # Security Check: Assure cache matches requested dataset size to prevent index out of bounds
                if len(self.cached_sigs) != len(self.sequences) or len(self.cached_feats) != len(self.sequences) or len(self.cached_labels) != len(self.sequences):
                    print(f"Cache size mismatch for {cache_name}! Wanted {len(self.sequences)} but got {len(self.cached_sigs)}. Forcing rebuild...")
                    self.cached_sigs = None
            
            if self.cached_sigs is None:
                print(f"Building parallel cache for {cache_name}... (Using 4 CPU cores)")
                base_dataset = EEGDataset(sequences, is_train=False)
                # Use a DataLoader with multiple workers to build the cache in parallel
                cache_loader = DataLoader(
                    base_dataset, 
                    batch_size=1, 
                    num_workers=4, 
                    shuffle=False, 
                    pin_memory=False
                )
                
                all_sigs = []
                all_feats = []
                all_labels = []
                
                for s, f, l in tqdm(cache_loader, desc=f"Caching {cache_name}"):
                    # s, f are (1, ...) due to batch_size=1
                    all_sigs.append(s.squeeze(0))
                    all_feats.append(f.squeeze(0))
                    all_labels.append(l.squeeze(0))
                
                self.cached_sigs = torch.stack(all_sigs)
                self.cached_feats = torch.stack(all_feats)
                self.cached_labels = torch.stack(all_labels)
                
                torch.save(self.cached_sigs, sig_path)
                torch.save(self.cached_feats, feat_path)
                torch.save(self.cached_labels, label_path)
                print(f"Cache saved for {cache_name}")

    def __len__(self):
        return len(self.cached_sigs)

    def __getitem__(self, idx):
        # We don't apply augmentation to cached data to keep it deterministic and fast
        return self.cached_sigs[idx], self.cached_feats[idx], self.cached_labels[idx]

def create_temporal_sequences(windows_list, seq_len=5):
    """
    Group windows by record and create sliding window sequences of size seq_len.
    """
    # Group by record_id
    record_groups = {}
    for w in windows_list:
        rid = w['record']['record_id']
        if rid not in record_groups:
            record_groups[rid] = []
        record_groups[rid].append(w)
        
    sequences = []
    for rid in record_groups:
        # Sort windows by start time within record
        wins = sorted(record_groups[rid], key=lambda x: x['start'])
        if len(wins) < seq_len:
            continue
            
        for i in range(len(wins) - seq_len + 1):
            seq = wins[i : i + seq_len]
            sequences.append(seq)
            
    return sequences

def calculate_dataset_stats(sequences, num_samples=50):
    """
    Stats on sequences. Returns mean/std of windows within sequences.
    """
    # Not strictly needed anymore with per-window processing, but kept for signature compatibility
    return { 'sig_mean': 0.0, 'sig_std': 1.0, 'feat_mean': 0.0, 'feat_std': 1.0 }

def print_split_distribution(name, sequences):
    labels = [s[-1]['label'] for s in sequences]
    counts = np.bincount(labels, minlength=2)
    print(f"--- {name} (Sequences) Distribution ---")
    print(f"  Total: {len(labels)}")
    print(f"  Seizure: {counts[1]}")
    print(f"  Background: {counts[0]}")

def get_dataloaders(windows_list, batch_size=16, num_workers=0, seq_len=5):
    """
    Create sequences then split.
    """
    sequences = create_temporal_sequences(windows_list, seq_len=seq_len)
    labels = [s[-1]['label'] for s in sequences]
    
    train_seq, temp_seq = train_test_split(
        sequences, test_size=0.30, stratify=labels, random_state=42
    )
    
    temp_labels = [s[-1]['label'] for s in temp_seq]
    val_seq, test_seq = train_test_split(
        temp_seq, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    # Train set balancing (on sequences)
    random.seed(42)
    seiz_seq = [s for s in train_seq if s[-1]['label'] == 1]
    bg_seq = [s for s in train_seq if s[-1]['label'] == 0]
    target_bg = len(seiz_seq) * 5
    if target_bg > 0 and len(bg_seq) > target_bg:
        train_seq = seiz_seq + random.sample(bg_seq, target_bg)
        random.shuffle(train_seq)
    
    print_split_distribution("Train", train_seq)
    print_split_distribution("Validation", val_seq)
    print_split_distribution("Test", test_seq)
    
    # 50/50 Weighted Sampling for training
    train_labels = [s[-1]['label'] for s in train_seq]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[l] for l in train_labels]
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Use 2 workers for CPU efficiency, pin memory for faster transfer
    loader_args = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True}
    
    return (
        DataLoader(EEGDataset(train_seq, is_train=True), **loader_args, sampler=sampler),
        DataLoader(EEGDataset(val_seq, is_train=False), **loader_args, shuffle=False),
        DataLoader(EEGDataset(test_seq, is_train=False), **loader_args, shuffle=False),
        train_seq
    )

def get_cross_dataset_loaders(source_windows, target_windows, batch_size=16, num_workers=0, seq_len=5):
    """
    Train: All Source (100%)
    Val: Target subset (20% by patient)
    Test: Target subset (80% by patient)
    """
    # 1. Prepare Source Train
    train_seq = create_temporal_sequences(source_windows, seq_len=seq_len)
    
    # Balance source training data
    random.seed(42)
    seiz_seq = [s for s in train_seq if s[-1]['label'] == 1]
    bg_seq = [s for s in train_seq if s[-1]['label'] == 0]
    target_bg = len(seiz_seq) * 5
    if target_bg > 0 and len(bg_seq) > target_bg:
        train_seq = seiz_seq + random.sample(bg_seq, target_bg)
        random.shuffle(train_seq)
        
    # 2. Prepare Target Val/Test (STRICT Human-level split)
    target_seq = create_temporal_sequences(target_windows, seq_len=seq_len)
    
    # Extract Human ID precisely: e.g., 'chb01' or 'pat001'
    def extract_human_id(s):
        p_id = s[0]['record']['patient_id']
        # Use first part of underscore or hyphen, otherwise use raw ID
        if '_' in p_id: return p_id.split('_')[0]
        if '-' in p_id: return p_id.split('-')[0]
        return p_id
        
    target_human_groups = [extract_human_id(s) for s in target_seq]
    unique_groups = len(set(target_human_groups))
    
    # SAFETY GUARD: Ensure we have enough groups for a patient-level split
    if unique_groups >= 5:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
        val_idx, test_idx = next(gss.split(target_seq, groups=target_human_groups))
    else:
        # Fallback to Stratified Window Split if patient grouping is impossible
        from sklearn.model_selection import train_test_split
        target_labels = [s[-1]['label'] for s in target_seq]
        val_idx, test_idx = train_test_split(
            np.arange(len(target_seq)), test_size=0.8, stratify=target_labels, random_state=42
        )
    
    val_seq = [target_seq[i] for i in val_idx]
    test_seq = [target_seq[i] for i in test_idx]
    
    # Clear old caches to prevent stale evaluation
    v_cache = f"val_tgt_v2_{len(val_seq)}"
    t_cache = f"test_tgt_v2_{len(test_seq)}"
    
    print_split_distribution("Source Train", train_seq)
    print_split_distribution("Target Validation", val_seq)
    print_split_distribution("Target Test", test_seq)
    
    # 50/50 Weighted Sampling for cross-dataset training
    train_labels = [s[-1]['label'] for s in train_seq]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[l] for l in train_labels]
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    loader_args = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True}
    
    # Note: We enable caching for cross-dataset as it's the longest run
    return (
        DataLoader(CachedEEGDataset(train_seq, cache_name=f"train_src", is_train=True), **loader_args, sampler=sampler),
        DataLoader(CachedEEGDataset(val_seq, cache_name=v_cache, is_train=False), **loader_args, shuffle=False),
        DataLoader(CachedEEGDataset(test_seq, cache_name=t_cache, is_train=False), **loader_args, shuffle=False),
        train_seq
    )

def get_full_test_loader(windows_list, batch_size=16, num_workers=0, seq_len=5):
    """
    Return all available sliding sequences for the test set.
    """
    sequences = create_temporal_sequences(windows_list, seq_len=seq_len)
    dataset = EEGDataset(sequences, is_train=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

def balance_dataset(windows):
    """
    Apply 5:1 background-to-seizure balancing.
    """
    random.seed(42)
    seizure_wins = [w for w in windows if w['label'] == 1]
    bg_wins = [w for w in windows if w['label'] == 0]
    
    target_bg = len(seizure_wins) * 5
    if target_bg > 0 and len(bg_wins) > target_bg:
        bg_wins = random.sample(bg_wins, target_bg)
        balanced = seizure_wins + bg_wins
        random.shuffle(balanced)
        return balanced
    return windows
