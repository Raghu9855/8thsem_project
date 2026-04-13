import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch

from utils import SEIZEIT2_DIR, OUTPUTS_DIR, set_seed
from data_loader import get_seizeit2_records
from segmentation import generate_window_metadata
from labeling import label_windows
from dataset_builder import EEGDataset, get_cached_raw_edf
from preprocessing import preprocess_eeg_window

def plot_visuals():
    set_seed(42)
    os.makedirs(os.path.join(OUTPUTS_DIR, 'plots'), exist_ok=True)
    
    # 1. Fetch a Seizure Window
    print("Loading data for visualization...")
    records = get_seizeit2_records(SEIZEIT2_DIR)
    window_metadata = generate_window_metadata(records, window_size_sec=5.0)
    labeled_windows = label_windows(window_metadata)
    
    seizure_wins = [w for w in labeled_windows if w['label'] == 1]
    if not seizure_wins:
        print("No seizures found! Using background.")
        win_info = labeled_windows[0]
    else:
        win_info = seizure_wins[0]
        
    record = win_info['record']
    edf_path = record['path']
    start_sec = win_info['start']
    end_sec = win_info['end']
    
    raw = get_cached_raw_edf(edf_path)
    sfreq = raw.info['sfreq']
    start_samp = int(start_sec * sfreq)
    end_samp = int(end_sec * sfreq)
    
    data, _ = raw[:, start_samp:end_samp]
    
    # Take a single channel for clear visualization (Channel 0)
    raw_signal = data[0, :]
    
    # Preprocess
    preprocessed_data = preprocess_eeg_window(data, sfreq=sfreq, target_sfreq=256.0)
    clean_signal = preprocessed_data[0, :]
    
    t_raw = np.linspace(0, 5, len(raw_signal))
    t_clean = np.linspace(0, 5, len(clean_signal))
    
    # ----------------------------------------------------
    # Plot 1: Before vs After Preprocessing
    # ----------------------------------------------------
    print("Generating Preprocessing Plot...")
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    
    axs[0].plot(t_raw, raw_signal, color='red', alpha=0.7)
    axs[0].set_title('Raw Noisy EEG Signal (5 Seconds)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    
    axs[1].plot(t_clean, clean_signal, color='blue', alpha=0.9)
    axs[1].set_title('Processed EEG Signal (Resampled, Filtered, Z-Score Normalized)')
    axs[1].set_xlabel('Time (Seconds)')
    axs[1].set_ylabel('Normalized Amplitude')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', 'preprocessing_plot.png'))
    plt.close()
    
    # ----------------------------------------------------
    # Plot 2: 2D Spectrogram Heatmap
    # ----------------------------------------------------
    print("Generating Spectrogram Heatmap...")
    f, t, Sxx = signal.spectrogram(preprocessed_data, fs=256.0, nperseg=64, noverlap=32)
    idx = f <= 40.0
    Sxx_filtered = Sxx[:, idx, :]
    f_filtered = f[idx]
    
    # Mean across all channels for a global representation
    Sxx_mean = np.mean(Sxx_filtered, axis=0)
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f_filtered, 10 * np.log10(Sxx_mean + 1e-10), shading='gouraud', cmap='magma')
    plt.title('2D EEG Spectrogram Heatmap (0-40 Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'plots', 'spectrogram.png'))
    plt.close()
    
    print("Visuals generated successfully at outputs/plots/")

if __name__ == '__main__':
    plot_visuals()
