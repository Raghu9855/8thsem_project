import os
# Force CPU-safe environment variables BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import sys

# Configure logging immediately
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Orchestrator')
logger.info("Initializing Research-Grade EEG Experiment Pipeline...")
sys.stdout.flush()

# Ensure project src is in the system path (handle spaces in paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR) # Insert at beginning to override any shadows

def load_ai_framework():
    """Lazy-load heavy frameworks with tracing."""
    logger.info("Starting AI framework initialization...")
    
    logger.info("Importing PyTorch (torch)...")
    sys.stdout.flush()
    import torch
    logger.info(f"PyTorch {torch.__version__} loaded successfully.")
    
    logger.info("Importing MNE (mne)...")
    sys.stdout.flush()
    import mne
    logger.info(f"MNE {mne.__version__} loaded successfully.")
    
    logger.info("Importing Scikit-Learn...")
    sys.stdout.flush()
    import sklearn
    logger.info("Scikit-Learn loaded successfully.")
    
    from train import train_model
    return train_model

def run_pipeline():
    logger.info("Loading system utilities...")
    sys.stdout.flush()
    try:
        from src.utils import SEIZEIT2_DIR, CHBMIT_DIR
        logger.info(f"CHB-MIT Directory: {CHBMIT_DIR}")
        logger.info(f"SEIZEIT2 Directory: {SEIZEIT2_DIR}")
        
    except Exception as e:
        logger.error(f"Failed to initialize paths: {e}")
        return

    # LAZY LOADING STEP
    try:
        train_model = load_ai_framework()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load AI framework: {e}")
        import traceback
        traceback.print_exc()
        return

    experiments = [
        ("CHB", "CHB"), 
        ("CHB", "SEIZE"),
       
        ("SEIZE", "CHB"),
        ("SEIZE", "SEIZE")
    ]

    for train_ds, test_ds in experiments:
        try:
            train_model(
                model_name='cnn_swin', 
                train_dataset_type=train_ds, 
                test_dataset_type=test_ds, 
                num_epochs=10, 
                batch_size=16, 
                device='cpu'
            )
        except Exception as e:
            logger.error(f"Experiment {train_ds} -> {test_ds} failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("All experiments completed.")
    
    # --- START XAI SUITE ---
    logger.info("Initializing XAI (Explainable AI) analysis...")
    try:
        from src.explainability import UltimateXAIReseacher
        # Run analysis on the two within-dataset models
        models_to_analyze = {
            "CHB": os.path.join(OUTPUTS_DIR, 'saved_models', 'best_cnn_swin_CHB_to_CHB.pth'),
            "SEIZE": os.path.join(OUTPUTS_DIR, 'saved_models', 'best_cnn_swin_SEIZE_to_SEIZE.pth')
        }
        
        for ds_name, m_path in models_to_analyze.items():
            if os.path.exists(m_path):
                researcher = UltimateXAIReseacher(m_path, ds_name, device='cpu')
                researcher.run_all()
    except Exception as e:
        logger.error(f"XAI Suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()
