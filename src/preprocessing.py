import numpy as np
import mne
from scipy import signal

def preprocess_eeg_window(data, sfreq, target_sfreq=256.0):
    """
    Applies preprocessing to an EEG window (NumPy array of shape (channels, samples)).
    
    Processing Steps:
    1. Resample to 256 Hz if necessary.
    2. Bandpass filter 0.5 - 40 Hz.
    3. Notch filter at 50 Hz.
    4. Z-score normalization per channel.
    """
    # 1. Resample
    if sfreq != target_sfreq:
        n_samples = data.shape[1]
        target_n_samples = int(n_samples * target_sfreq / sfreq)
        data = signal.resample(data, target_n_samples, axis=1)
        current_sfreq = target_sfreq
    else:
        current_sfreq = float(sfreq)
        
    # Determine nyquist for filter design
    nyq = current_sfreq / 2.0
    
    # 2. Bandpass filter 0.5 - 40 Hz using Butterworth filter
    low = 0.5 / nyq
    high = 40.0 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    data = signal.filtfilt(b, a, data, axis=1)
    
    # 3. Notch filter at 50 Hz
    if nyq > 50.0:
        b_notch, a_notch = signal.iirnotch(50.0, 30.0, current_sfreq)
        data = signal.filtfilt(b_notch, a_notch, data, axis=1)
        
    # 4. Robust scaling (Per-channel, per-window)
    # X = (X - median) / IQR
    median = np.median(data, axis=1, keepdims=True)
    q75, q25 = np.percentile(data, [75, 25], axis=1)
    iqr = (q75 - q25).reshape(-1, 1)
    
    iqr[iqr == 0] = 1.0 # avoid division by zero
    data = (data - median) / iqr
    
    # 5. Clip extreme amplitudes
    data = np.clip(data, -5, 5)
    
    return data
