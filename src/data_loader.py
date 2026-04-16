import os
import glob
import pandas as pd
import mne
import re

def parse_chbmit_summary(summary_path):
    """
    Parses a CHB-MIT summary file to extract seizure intervals per EDF.
    Returns a dictionary: {edf_filename: [(start_sec, end_sec), ...]}
    """
    seizures = {}
    current_edf = None
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            current_edf = line.split(":")[1].strip()
            if current_edf not in seizures:
                seizures[current_edf] = []
        elif "Seizure" in line and "Start Time" in line:
            # Extract start time
            try:
                start_sec = int(re.findall(r'\d+', line.split(":")[1])[0])
                # We expect the next line to be End Time
            except IndexError:
                continue
        elif "Seizure" in line and "End Time" in line:
            try:
                end_sec = int(re.findall(r'\d+', line.split(":")[1])[0])
                seizures[current_edf].append((start_sec, end_sec))
            except IndexError:
                continue
                
    return seizures

def get_chbmit_records(data_dir):
    """
    Collects all CHB-MIT records and their ground truth seizure intervals.
    Returns: List of dicts
    """
    records = []
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, 'chb*')))
    
    for pdir in patient_dirs:
        if not os.path.isdir(pdir):
            continue
            
        patient_id = os.path.basename(pdir)
        summary_file = os.path.join(pdir, f"{patient_id}-summary.txt")
        
        seizure_dict = {}
        if os.path.exists(summary_file):
            seizure_dict = parse_chbmit_summary(summary_file)
            
        edf_files = sorted(glob.glob(os.path.join(pdir, '*.edf')))
        for edf_path in edf_files:
            filename = os.path.basename(edf_path)
            intervals = seizure_dict.get(filename, [])
            records.append({
                'dataset': 'chbmit',
                'patient_id': patient_id,
                'record_id': filename.replace('.edf', ''),
                'path': edf_path,
                'seizures': intervals
            })
            
    return records

def get_seizeit2_records(data_dir):
    """
    Collects all SeizeIT2 records and their ground truth seizure intervals from TSV files.
    Returns: List of dicts
    """
    records = []
    # Find all EDF files in the BIDS structure
    edf_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', 'ses-*', 'eeg', '*.edf')))
    
    for edf_path in edf_files:
        filename = os.path.basename(edf_path)
        subject_match = re.search(r'(sub-\d+)', filename)
        patient_id = subject_match.group(1) if subject_match else 'unknown'
        
        # Determine TSV path
        tsv_path = edf_path.replace('_eeg.edf', '_events.tsv')
        intervals = []
        
        if os.path.exists(tsv_path):
            try:
                df = pd.read_csv(tsv_path, sep='\t')
                # seizure events are those where eventType starts with 'sz'
                if 'eventType' in df.columns and 'onset' in df.columns and 'duration' in df.columns:
                    seiz_events = df[df['eventType'].str.startswith('sz', na=False)]
                    for _, row in seiz_events.iterrows():
                        start = row['onset']
                        end = start + row['duration']
                        intervals.append((start, end))
            except Exception as e:
                print(f"Error reading TSV {tsv_path}: {e}")
                
        records.append({
            'dataset': 'seizeit2',
            'patient_id': patient_id,
            'record_id': filename.replace('_eeg.edf', ''),
            'path': edf_path,
            'seizures': intervals
        })
        
    return records

def load_dataset(dataset_type="CHB"):
    """
    Loads EEG records for a specific dataset. 
    dataset_type: "CHB" for CHB-MIT or "SEIZE" for SeizeIT2.
    """
    from utils import CHBMIT_DIR, SEIZEIT2_DIR
    records = []
    
    if dataset_type.upper() == "CHB":
        records = get_chbmit_records(CHBMIT_DIR)
    elif dataset_type.upper() == "SEIZE":
        records = get_seizeit2_records(SEIZEIT2_DIR)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'CHB' or 'SEIZE'.")
        
    return records

def load_eeg_metadata(edf_path):
    """
    Quickly load EDF metadata without loading into memory.
    """
    info = mne.io.read_raw_edf(edf_path, preload=False, verbose='error')
    n_samples = info.n_times
    sfreq = info.info['sfreq']
    ch_names = info.ch_names
    duration_sec = n_samples / sfreq
    return {
        'n_samples': n_samples,
        'sfreq': sfreq,
        'ch_names': ch_names,
        'duration_sec': duration_sec
    }
