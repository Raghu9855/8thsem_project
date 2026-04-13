import numpy as np
from scipy import stats, signal

def higuchi_fd(x, kmax=10):
    """
    Compute Higuchi Fractal Dimension of a 1D array.
    """
    N = len(x)
    L = np.zeros((kmax,))
    for k in range(1, kmax + 1):
        Lk = np.zeros((k,))
        for m in range(0, k):
            idx = np.arange(m, N, k)
            if len(idx) > 1:
                diff = np.abs(np.diff(x[idx]))
                Lk[m] = np.sum(diff) * (N - 1) / (((len(idx) - 1) * k) * k)
        L[k - 1] = np.mean(Lk)
    
    # Fit line to log-log plot
    log_k = np.log(1.0 / np.arange(1, kmax + 1))
    log_L = np.log(L)
    log_L = log_L[~np.isnan(log_L) & ~np.isinf(log_L)]
    log_k = log_k[:len(log_L)]
    
    if len(log_k) > 1:
        slope, _, _, _, _ = stats.linregress(log_k, log_L)
        return slope
    return 0.0

def permutation_entropy(x, order=3, delay=1):
    """
    Compute Permutation Entropy of a 1D array.
    """
    N = len(x)
    if N < order * delay:
        return 0.0
    
    # Create phase space reconstruction
    mat = np.zeros((N - (order - 1) * delay, order))
    for i in range(order):
        mat[:, i] = x[i * delay : i * delay + mat.shape[0]]
        
    # Get permutations
    sort_indices = np.argsort(mat, axis=1)
    
    # Hash permutations to count frequencies
    hash_mults = np.power(order, np.arange(order))
    hashed = np.sum(sort_indices * hash_mults, axis=1)
    
    _, counts = np.unique(hashed, return_counts=True)
    p = counts / counts.sum()
    pe = -np.sum(p * np.log2(p))
    
    # Normalize
    pe = pe / np.log2(math.factorial(order)) if order > 1 else 0.0
    return pe

import math # required for permutation_entropy

def extract_features(data, sfreq=256):
    """
    Extracts time, frequency, and nonlinear features from an EEG window.
    data format: (channels, samples)
    Returns a 1D feature vector.
    """
    n_channels, n_samples = data.shape
    features = []
    
    # Frequencies for Welch
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    for ch in range(n_channels):
        sig = data[ch, :]
        
        # 1. Time-domain
        mean = np.mean(sig)
        var = np.var(sig)
        rms = np.sqrt(np.mean(sig**2))
        skew = stats.skew(sig)
        kurt = stats.kurtosis(sig)
        time_feats = [mean, var, rms, skew, kurt]
        
        # 2. Frequency-domain (Welch)
        freqs, psd = signal.welch(sig, fs=sfreq, nperseg=min(sfreq*2, n_samples))
        freq_feats = []
        for band, (low, high) in freq_bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.sum(psd[idx])
            freq_feats.append(band_power)
            
        # 3. Nonlinear
        pe = permutation_entropy(sig)
        hfd = higuchi_fd(sig)
        nl_feats = [pe, hfd]
        
        # Combine per channel
        ch_features = time_feats + freq_feats + nl_feats
        features.extend(ch_features)
        
    return np.array(features, dtype=np.float32)
