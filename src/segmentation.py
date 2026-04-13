import math
from data_loader import load_eeg_metadata

def generate_window_metadata(records, window_size_sec=5.0, overlap_ratio=0.5):
    """
    Generates metadata for all possible windows across all given records.
    Returns: List of tuples (record_info, start_time_sec, end_time_sec)
    """
    all_windows = []
    step_sec = window_size_sec * (1.0 - overlap_ratio)
    
    for record in records:
        try:
            metadata = load_eeg_metadata(record['path'])
        except Exception as e:
            print(f"Skipping {record['path']} due to read error: {e}")
            continue
            
        duration = metadata['duration_sec']
        
        # Calculate sliding windows
        current_start = 0.0
        while current_start + window_size_sec <= duration:
            current_end = current_start + window_size_sec
            all_windows.append((record, current_start, current_end))
            current_start += step_sec
            
    return all_windows
