import numpy as np
import random

def add_gaussian_noise(data, snr_db=10):
    """
    Adds Gaussian noise to the EEG window based on a desired Target SNR.
    """
    signal_power = np.mean(data ** 2)
    signal_power_db = 10 * np.log10(signal_power if signal_power > 0 else 1e-10)
    noise_power_db = signal_power_db - snr_db
    noise_power = 10 ** (noise_power_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

def time_shift(data, shift_max=0.1, sfreq=256):
    """
    Shifts the EEG signal in time.
    shift_max: max shift in seconds (e.g., 0.1s -> 25 samples at 256Hz)
    """
    shift_samples = int(shift_max * sfreq)
    shift = random.randint(-shift_samples, shift_samples)
    if shift == 0:
        return data
    
    shifted_data = np.zeros_like(data)
    if shift > 0:
        shifted_data[:, shift:] = data[:, :-shift]
        shifted_data[:, :shift] = data[:, 0:1] # padding 
    else:
        shift_abs = abs(shift)
        shifted_data[:, :-shift_abs] = data[:, shift_abs:]
        shifted_data[:, -shift_abs:] = data[:, -1:] # padding
        
    return shifted_data

def amplitude_scale(data, scale_range=(0.8, 1.2)):
    """
    Scales the amplitude of the EEG signal.
    """
    scale = random.uniform(scale_range[0], scale_range[1])
    return data * scale

def co_mixup(batch_x, batch_y, alpha=0.2):
    """
    Applies Co-Mixup to a batch of tensors.
    batch_x: EEG windows (B, C, L)
    batch_y: Labels (B,)
    """
    import torch
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = batch_x.size()[0]
    index = torch.randperm(batch_size).to(batch_x.device)

    mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
    y_a, y_b = batch_y, batch_y[index]
    
    return mixed_x, y_a, y_b, lam

def apply_transforms(data, sfreq=256):
    """
    Applies a random set of transformations for data augmentation.
    To be used only on minority (seizure) windows.
    """
    if random.random() < 0.5:
        data = add_gaussian_noise(data)
    if random.random() < 0.5:
        data = time_shift(data, sfreq=sfreq)
    if random.random() < 0.5:
        data = amplitude_scale(data)
    return data
