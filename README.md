# 🧠 Cross-Domain EEG Seizure Detection: From Clinical Gold-Standard to Wearable Reality

A robust, adversarial domain-adaptation pipeline designed to bridge the gap between high-density clinical EEG (23-channels, CHB-MIT) and sparse wearable sensors (2-channels, SEIZEIT2).

---

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
# Activate the virtual environment first
eeg_env\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Configure Dataset Paths
Edit `src/utils.py` (or `src/train.py`) to point to your local dataset locations:
```python
CHBMIT_DIR  = r"path\to\chbmit\physionet.org\files\chbmit\1.0.0"
SEIZEIT2_DIR = r"path\to\seizeit2"
```

### 3. Train All Models
```bash
python main.py
```
This runs 3 models (`cnn_swin`, `cnn_lstm`, `cnn_gnn`) across all 4 domain experiments (12 total runs). Trained weights are saved to `outputs/saved_models/`.

### 4. Evaluate Models
```bash
python src/evaluate.py
# Optional: specify models
python src/evaluate.py --models cnn_swin cnn_lstm
```
Evaluation outputs (ROC curves, confusion matrices, metrics CSV) are saved to `outputs/plots/` and `outputs/reports/`.

---

## ⚡ 1. The Reality: Why EEG is "Hard Mode" for ML

Seizure detection isn't a traditional classification problem; it's a **non-stationary signal alignment problem**.

Most existing repositories fail in the real world because they ignore:
*   **The Domain Chasm:** Model performance on clinical data (CHB-MIT) rarely transfers to wearables (SEIZEIT2) because of extreme channel count differences (23 vs 2) and signal-to-noise ratio (SNR) shifts.
*   **Patient Specificity:** Seizure morphologies are unique to individuals. A model that "learns the patient" instead of "learning the seizure" will fail 100% of the time on new data.
*   **Temporal Non-Stationarity:** EEG signatures evolve over time, making fixed-feature systems brittle.

---

## 📉 2. Development History & Initial Assumptions

This project began with a CNN-LSTM baseline and evolved into a three-model comparison system (CNN-LSTM, CNN-GNN, CNN-Swin Transformer), each evaluated across all domain transfer scenarios.

*   **Assumption 1:** We assumed that feeding Raw EEG directly into a deep model would allow the model to learn the frequency filters.
*   **Reality:** Raw signals are too noisy on CPU-scale training. We shifted to an STFT-based representation to provide a "spectral roadmap" for the model.
*   **Assumption 2:** We assumed global standardization (StandardScaler) was sufficient.
*   **Reality:** This destroyed the 2-channel data because the padded zero-channels corrupted the variance calculations.

---

## 💥 3. Failure Analysis (Brutal Honesty)

The early iterations of this system failed spectacularly. Here is why:

*   **Metric Slingshotting:** Initial DANN (Adversarial) implementations caused the model to "give up" on classification to satisfy the discriminator. AUC would hit 0.45 (worse than guessing).
*   **The Weight Mirage:** We were incorrectly measuring performance using EMA (Exponential Moving Average) weights but saving **Raw** weights. This caused a 15% drop in performance during actual deployment.
*   **Normalization Corruptions:** Normalizing across the entire 22-channel array meant that if a wearable only had 2 active sensors, the noise floor was amplified 10x, making seizures indistinguishable from background.

---

## 🔬 4. Iteration Timeline

### Phase 1: The Naive System (Naive CNN-LSTM)
*   **Build:** Raw signals + Global Normalization + Softmax Output.
*   **Failure:** ROC-AUC hit a ceiling of 0.64 on CHB and 0.47 on SEIZE. It failed to generalize because it overfit the high-density spatial layout of the clinical cap.

### Phase 2: Structural Fixes (Feature Engineering & AE)
*   **Insight:** The model needed a compressed "latent language" to talk about features.
*   **Change:** Added a **Feature Autoencoder** to reduce manual features into a 64-dim latent vector, fused with a **Swin-Transformer** backbone for the final classification.

### Phase 3: The Generalization System (Current)
*   **Insight:** We had to force the model to *expect* data loss.
*   **Change:** Implemented **Extreme Sensor Dropout** (dropping 0–90% of channels during training). This forced the 23-channel clinical model to learn how to detect seizures using only 2 channels.

---

## 🏗️ 5. Architecture

### 5.1 End-to-End Pipeline

Two parallel streams are combined:

```
Stream A (EEG Signal):
Raw EEG (5s Windows) → Channel-Wise Norm → InstanceNorm2D → CNN Feature Extractor
→ Swin Transformer (2 blocks) → Adaptive Pool → [512-dim feature]

Stream B (Manual Features):
12 features/channel × n_channels → Feature Autoencoder (Encoder only) → [64-dim latent]

Fusion: Concat(Stream A, Stream B) → LayerNorm → Temporal Transformer
→ Adversarial Safety Brake → Youden's J Optimized Sigmoid Output
```

### 5.2 The "Hybrid" Representation

We do not use an end-to-end deep model. Why?

1.  **Manual Features:** We extract **12 features per channel** (Mean, Variance, RMS, Skewness, Kurtosis, Delta/Theta/Alpha/Beta/Gamma band power, Permutation Entropy, Higuchi Fractal Dimension).
    - CHB-MIT (23 channels): **276-dim** feature vector
    - SEIZEIT2 (2 channels): **24-dim** feature vector
2.  **Autoencoder:** A **6-layer autoencoder** (3 encoder + 3 decoder linear layers) compresses these into a 64-dim latent space. Only the encoder half is used at inference time.
3.  **Fusion:** The 64-dim AE latent is **concatenated** with the CNN-Swin pooled features *after* both streams are processed independently. The fused representation then passes through the Temporal Transformer.

*   **Justification:** This hybrid approach provides "domain knowledge" (Manual Features) while allowing the Transformer to discover "hidden patterns" (DL). The concatenation-based fusion avoids interference between the two streams during early training stages.

### 5.3 Training Strategy: Stage-Wise Curriculum (10 epochs)

| Stage | Epochs | CDANN (α) | Margin Ranking | Description |
| :--- | :--- | :--- | :--- | :--- |
| **0 — Orientation** | 0–1 | Off | On (weight 0.1) | Pure classification + light ranking. LR = 2e-4 |
| **1 — Alignment** | 2–4 | On | **Off** | CDANN activates, ranking disabled to avoid competing gradients. LR = 5e-5 |
| **2 — Refinement** | 5–9 | On | On (weight 1.0) | Full loss suite. Adversarial brake monitors class gap. LR = 1e-5 |

**Adversarial Safety Brake:** If the probability gap between seizure and background predictions drops below 0.1, the GRL alpha is forced to zero — preventing the discriminator from destroying the seizure-detection signal.

---

## 🧠 6. Models Compared

Three architectures are trained and evaluated across all domain scenarios:

| Model | Key Component | File |
| :--- | :--- | :--- |
| **CNN-Swin** *(primary)* | Swin Transformer blocks with windowed attention | `src/models/cnn_swin_transformer.py` |
| **CNN-LSTM** | Bidirectional LSTM for temporal modeling | `src/models/cnn_lstm.py` |
| **CNN-GNN** | Graph Neural Network for spatial channel relationships | `src/models/cnn_gnn.py` |

All three use the same CNN feature extractor backbone (`common_blocks.py`) and the same Feature Autoencoder for the manual feature stream.

---

## 🚫 7. Failed Ideas (Deep Technical Reasoning)

| Technique | Status | Reason for Failure |
| :--- | :--- | :--- |
| **CORAL Loss** | Abandoned | Correlation alignment is too "global." It washed out the fine-grained seizure spikes in favor of matching the noise floors. |
| **Global Softmax** | Abandoned | Softmax is too aggressive for imbalanced EEG. Switched to Sigmoid + Youden's J Thresholding for clinical sensitivity. |
| **DANN without Safety Brake** | Abandoned | Unconstrained adversarial training caused the classifier to collapse. The adversarial brake (gap < 0.1 → α = 0) was the key fix. |

---

## ⚙️ 8. Design Justifications (The "Why")

*   **Why Autoencoder before classification?**
    *   *Trade-off:* Adds latent complexity.
    *   *Decision:* It filters out the "feature noise." By forcing the model to reconstruct the features first, we ensure it only uses the most robust 64 dimensions for classification.
*   **Why 5-Second Windows?**
    *   *Decision:* Clinically, seizures take time to manifest in the frequency domain. Anything shorter than 5s fails to capture the evolution of the rhythmic discharge.
*   **Why not Raw Signals?**
    *   *Decision:* On CPU, Raw signals require massive filters that are computationally expensive. STFT spectrograms allow us to use 2D-CNN kernels which are optimized for spatial pattern recognition.
*   **Why concatenate AE latent after Swin (not condition the Swin with it)?**
    *   *Decision:* Injecting the AE latent into the Swin's attention would couple the two streams too tightly. Concatenation post-pooling keeps them independent, preventing the AE's domain-specific statistics from corrupting the spatial feature learning.

---

## 📊 9. Results (Honest)

Results from the CNN-Swin Transformer model (best performer):

| Scenario | ROC-AUC | Note |
| :--- | :--- | :--- |
| **CHB ➔ CHB** | **0.958** | Near perfect intra-domain fidelity. |
| **SEIZE ➔ SEIZE** | **0.860** | Highly robust on wearable data. |
| **SEIZE ➔ CHB** | **0.722** | Successful "Sparse-to-Dense" transfer. |
| **CHB ➔ SEIZE** | **0.546** | The most difficult trajectory (domain shift from 23ch → 2ch). |

> Results are from `outputs/saved_models/best_cnn_swin_*.pth` checkpoints. Performance may vary across runs due to the stochastic sensor dropout augmentation.

---

## 🌍 10. Generalization & Domain Shift

The shift from **Clinical (23ch)** to **Wearable (2ch)** is an asymmetric problem.
*   **What breaks:** The spatial connectivity features completely disappear.
*   **The solution:** Our **L1 Structural Sparsity** on the CNN's first conv layer erodes dead spatial filters, allowing the model to focus only on the longitudinal temporal signatures that exist in both datasets.

---

## 🧠 11. Key Insights

*   **What actually matters:** Normalization. If you normalize incorrectly, you are just training a model to detect the hardware difference between datasets, not the seizures.
*   **What we misunderstood:** We thought more data meant more accuracy. In reality, **diverse** data with sensor dropout is the only way to achieve **>0.60 AUC** in the hardest cross-domain scenarios (CHB→SEIZE).

---

## ⚠️ 12. Limitations
*   **Hardware:** Optimized for CPU; GPU latency would be significantly lower.
*   **Real-world:** SEIZEIT2 is still limited in patient count. Performance may vary on "unseen" wearable hardware.
*   **Hardcoded paths:** Dataset paths in `src/utils.py` and `src/train.py` are currently hardcoded to the development machine. Update them before running on a new system.

---

## 🚀 13. Future Work
*   **Transfer Learning via Masked Autoencoders:** Pre-training on 10,000+ hours of unlabeled EEG.
*   **Quantization:** Reducing the model size for edge deployment on ESP32/ARM-Cortex wearable boards.
*   **Config File:** Replace hardcoded dataset paths with a `config.yaml` for portability.

---

## 📁 Project Structure

```
8thsem_project/
├── main.py                    # Orchestrates all training experiments
├── requirements.txt           # Python dependencies
├── src/
│   ├── train.py               # Training loop with staged curriculum
│   ├── evaluate.py            # Model evaluation & metric generation
│   ├── data_loader.py         # CHB-MIT and SEIZEIT2 data ingestion
│   ├── dataset_builder.py     # Window sampling, DataLoaders
│   ├── feature_extraction.py  # 12 features/channel (time, freq, nonlinear)
│   ├── autoencoder_reduction.py # 6-layer Feature Autoencoder
│   ├── explainability.py      # SHAP, attention rollout, XAI suite
│   ├── train_autoencoder.py   # Standalone AE pre-training script
│   ├── augmentation.py        # Sensor dropout augmentation
│   ├── preprocessing.py       # Channel-wise normalization
│   ├── segmentation.py        # 5s windowing with overlap
│   ├── labeling.py            # Seizure/background labeling
│   ├── plot_visuals.py        # ROC, confusion matrix plotting
│   ├── report_generator.py    # Clinical HTML report generation
│   └── models/
│       ├── cnn_swin_transformer.py  # Primary model (Swin + CDANN)
│       ├── cnn_lstm.py              # Baseline LSTM model
│       ├── cnn_gnn.py               # Graph Neural Network model
│       └── common_blocks.py         # Shared CNN feature extractor
└── outputs/                   # Generated (git-ignored)
    ├── saved_models/          # Best checkpoints (.pth)
    ├── plots/                 # ROC curves, confusion matrices
    ├── reports/               # Evaluation logs, metrics CSV
    └── clinical_reports/      # HTML clinical reports
```

---

*Created as part of the 8th Semester degree project (2026).*
