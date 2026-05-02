import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from src.utils import OUTPUTS_DIR, set_seed, CHBMIT_DIR
from src.dataset_builder import get_dataloaders
from src.data_loader import get_chbmit_records
from src.segmentation import generate_window_metadata
from src.labeling import label_windows
from src.autoencoder_reduction import FeatureAutoencoder
from src.models.cnn_swin_transformer import CNNSwinTransformerModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.path.join('outputs', 'saved_models', 'best_cnn_swin_CHB_to_CHB.pth')

if not os.path.exists(model_path):
    print("Model not found.")
    exit()

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
autoencoder = FeatureAutoencoder(input_dim=checkpoint['feature_dim']).to(device)
autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
autoencoder.eval()

model = CNNSwinTransformerModel(eeg_channels=checkpoint['eeg_channels']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

records = get_chbmit_records(CHBMIT_DIR)
window_metadata = generate_window_metadata(records, window_size_sec=5.0)
labeled_windows = label_windows(window_metadata)
_, _, test_loader, _ = get_dataloaders(labeled_windows, batch_size=8) # smaller batch size

sigs, feats, labels = next(iter(test_loader))

class SHAPWrapper(nn.Module):
    def __init__(self, model, autoencoder):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
    def forward(self, sig, feat):
        lat = self.autoencoder.encode(feat)
        return self.model(sig, lat)[:, 1:2]
        
wrapped_model = SHAPWrapper(model, autoencoder).to(device)
wrapped_model.eval()

bg_sig = sigs.to(device)
bg_feat = feats.to(device)

test_sig = sigs[0:1].to(device)
test_feat = feats[0:1].to(device)

print("Computing SHAP values...")
explainer = shap.GradientExplainer(wrapped_model, [bg_sig, bg_feat])
shap_values = explainer.shap_values([test_sig, test_feat])

if isinstance(shap_values, list) and isinstance(shap_values[0], list):
    shap_sig = shap_values[0][0]
elif isinstance(shap_values, list):
    shap_sig = shap_values[0]
else:
    shap_sig = shap_values

saliency_seq = np.abs(shap_sig[0]).mean(axis=(1, 2, 3))

out_dir = os.path.join(OUTPUTS_DIR, 'xai_results_CHB')
os.makedirs(out_dir, exist_ok=True)
plt.figure(figsize=(10, 4))
plt.plot(saliency_seq, marker='o', color='purple')
plt.title("SHAP Feature Importance (EEG Signal) - Sample 0")
plt.xlabel("Sequence Segment")
plt.ylabel("Mean Absolute SHAP Value")
plt.grid(True)
plt.savefig(os.path.join(out_dir, 'shap_importance_quick.png'))
plt.close()
print("Done. Saved to", os.path.join(out_dir, 'shap_importance_quick.png'))
