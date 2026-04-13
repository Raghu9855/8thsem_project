import os
import random
import numpy as np
import torch
import logging

# Path Configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CHBMIT_DIR = r"d:\8thsem project\physionet.org\files\chbmit\1.0.0"
SEIZEIT2_DIR = r"d:\8thsem project\ds005873"

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(handler)
        
        # Also print to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "saved_models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "xai"), exist_ok=True)
