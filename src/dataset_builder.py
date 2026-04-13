import torch
from torch.utils.data import Dataset, DataLoader
import mne
import numpy as np
import scipy.signal as signal
import random
from sklearn.model_selection import GroupShuffleSplit
from preprocessing import preprocess_eeg_window
from augmentation import apply_transforms
from feature_extraction import extract_features
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_raw_edf(edf_path):
    return mne.io.read_raw_edf(edf_path, preload=False, verbose='error')

class EEGDataset(Dataset):
    def __init__(self, windows, is_train=False, augment_prob=0.5):
        """
        windows: List of dicts {'record': dict, 'start': float, 'end': float, 'label': int}
        """
        self.windows = windows
        self.is_train = is_train
        self.augment_prob = augment_prob
        # Weak caching for disk streaming
        self.current_edf_path = None
        self.current_raw = None
        
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        win_info = self.windows[idx]
        record = win_info['record']
        edf_path = record['path']
        start_sec = win_info['start']
        end_sec = win_info['end']
        label = win_info['label']
        
        # Fast loading raw EDF from LRU cache
        raw = get_cached_raw_edf(edf_path)
            
        sfreq = raw.info['sfreq']
        start_samp = int(start_sec * sfreq)
        end_samp = int(end_sec * sfreq)
        
        data, _ = raw[:, start_samp:end_samp]
        
        # 1. Preprocess
        data = preprocess_eeg_window(data, sfreq=sfreq, target_sfreq=256.0)
        
        # 2. Augment (only training, mainly seizures)
        if self.is_train and label == 1 and np.random.rand() < self.augment_prob:
            data = apply_transforms(data, sfreq=256.0)
            
        # 3. Extract Features
        features = extract_features(data, sfreq=256.0)
        
        # 4. Standardize Channels for Cross-Dataset training to avoid variable sized tensors
        MAX_CHANNELS = 23
        FEATS_PER_CHANNEL = 12
        
        if data.shape[0] < MAX_CHANNELS:
            pad_width = MAX_CHANNELS - data.shape[0]
            data = np.pad(data, ((0, pad_width), (0, 0)), mode='constant')
        elif data.shape[0] > MAX_CHANNELS:
            data = data[:MAX_CHANNELS, :]
            
        if len(features) < MAX_CHANNELS * FEATS_PER_CHANNEL:
            pad_width = (MAX_CHANNELS * FEATS_PER_CHANNEL) - len(features)
            features = np.pad(features, (0, pad_width), mode='constant')
        elif len(features) > MAX_CHANNELS * FEATS_PER_CHANNEL:
            features = features[:MAX_CHANNELS * FEATS_PER_CHANNEL]
        
        # 5. Generate Spectrogram
        f, t, Sxx = signal.spectrogram(data, fs=256.0, nperseg=64, noverlap=32)
        idx = f <= 40.0
        Sxx = Sxx[:, idx, :]
        
        signal_tensor = torch.tensor(Sxx, dtype=torch.float32)
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, feature_tensor, label_tensor

def get_dataloaders(windows_list, batch_size=32, num_workers=0):
    """
    Patient-wise split to prevent leakage using GroupShuffleSplit.
    windows_list: list of window dicts.
    """
    # Extract patient groups
    groups = [w['record']['patient_id'] for w in windows_list]
    
    # We need Train, Val, Test split (70-15-15)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss1.split(windows_list, groups=groups))
    
    train_windows = [windows_list[i] for i in train_idx]
    
    # 5:1 Data Balancing for train_windows
    random.seed(42)
    seizure_wins = [w for w in train_windows if w['label'] == 1]
    bg_wins = [w for w in train_windows if w['label'] == 0]
    target_bg = len(seizure_wins) * 5
    if target_bg > 0 and len(bg_wins) > target_bg:
        bg_wins = random.sample(bg_wins, target_bg)
        train_windows = seizure_wins + bg_wins
        random.shuffle(train_windows)
        
    temp_windows = [windows_list[i] for i in temp_idx]
    temp_groups = [groups[i] for i in temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_windows, groups=temp_groups))
    
    val_windows = [temp_windows[i] for i in val_idx]
    test_windows = [temp_windows[i] for i in test_idx]
    
    train_dataset = EEGDataset(train_windows, is_train=True)
    val_dataset = EEGDataset(val_windows, is_train=False)
    test_dataset = EEGDataset(test_windows, is_train=False)
    
    # Strict CPU usage - num_workers should be appropriately set or 0 to avoid IPC issues on some OS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset
