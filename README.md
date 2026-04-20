# 🧠 Cross-Domain EEG Seizure Detection: From Clinical Gold-Standard to Wearable Reality



A robust, adversarial domain-adaptation pipeline designed to bridge the gap between high-density clinical EEG (23-channels) and sparse wearable sensors (2-channels).

---

## ⚡ 1. The Reality: Why EEG is "Hard Mode" for ML

Seizure detection isn't a traditional classification problem; it's a **non-stationary signal alignment problem**. 

Most existing repositories fail in the real world because they ignore:
*   **The Domain Chasm:** Model performance on clinical data (CHB-MIT) rarely transfers to wearables (SEIZEIT2) because of extreme channel count differences (23 vs 2) and signal-to-noise ratio (SNR) shifts.
*   **Patient Specificity:** Seizure morphologies are unique to individuals. A model that "learns the patient" instead of "learning the seizure" will fail 100% of the time on new data.
*   **Temporal Non-Stationarity:** EEG signatures evolve over time, making fixed-feature systems brittle.

---

## 📉 2. Fork Origin & Initial Assumptions

This project was built upon a standard CNN-LSTM architecture. 
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
*   **Change:** Added a **Feature Autoencoder** to reduce 270+ manual features into a 64-dim latent vector, feeding into a **Swin-Transformer**.

### Phase 3: The Generalization System (Current)
*   **Insight:** We had to force the model to *expect* data loss.
*   **Change:** Implemented **Extreme Sensor Dropout** (dropping 0-90% of channels during training). This forced the 23-channel clinical model to learn how to detect seizures using only 2 channels.

---

## 🏗️ 5. Elite Architecture

### 5.1 End-to-End Pipeline
`Raw EEG (5s Windows)` ➔ `Channel-Wise Normalization` ➔ `STFT Spectrograms` ➔ `Deep Feature Latent Projections` ➔ `CNN-Swin Transformer` ➔ `Adversarial Domain Alignment` ➔ `Youden's J Optimized Output`.

### 5.2 The "Hybrid" Representation
We do not use an end-to-end deep model. Why?
1.  **Manual Features:** We extract 12 features per channel (Skewness, Kurtosis, Band Powers).
2.  **Autoencoder:** These are passed through an 8-layer Autoencoder to create a compressed representation.
3.  **Transformer:** The Swin-Transformer processes the Spectrograms *while* conditioned by the Autoencoder's latent vector. 
*   **Justification:** This hybrid approach provides "domain knowledge" (Manual Features) while allowing the Transformer to discover "hidden patterns" (DL).

### 5.3 Training Strategy: Stage-Wise Curriculum
*   **Stage 0 (Orientation):** Pure classification. No adversarial loss.
*   **Stage 1 (Alignment):** **CDANN** (Conditional DANN) turns on. We align features conditioned on their predicted class to ensure "Seizure" in CHB looks like "Seizure" in SEIZE.
*   **Stage 2 (Refinement):** **Margin Ranking Loss** activates, pushing the probability gap between seizure and background even further apart.

---

## 🚫 6. Failed Ideas (Deep Technical Reasoning)

| Technique | Status | Reason for Failure |
| :--- | :--- | :--- |
| **CORAL Loss** | Abandoned | Correlation alignment is too "global." It washed out the fine-grained seizure spikes in favor of matching the noise floors. |
| **Global Softmax** | Abandoned | Softmax is too aggressive for imbalanced EEG. Switched to Sigmoid + Youden's J Thresholding for clinical sensitivity. |
| **LSTM Temporal Units** | Abandoned | LSTMs suffered from vanishing gradients on long 5-sequence windows. Transformers handled the temporal context far better. |

---

## ⚙️ 7. Design Justifications (The "Why")

*   **Why Autoencoder before classification?**
    *   *Trade-off:* Adds latent complexity. 
    *   *Decision:* It filters out the "feature noise." By forcing the model to reconstruct the features first, we ensure it only uses the most robust 64 dimensions for classification.
*   **Why 5-Second Windows?**
    *   *Decision:* Clinically, seizures take time to manifest in the frequency domain. Anything shorter than 5s fails to capture the evolution of the rhythmic discharge.
*   **Why not Raw Signals?**
    *   *Decision:* On CPU, Raw signals require massive filters that are computationally expensive. STFT spectrograms allow us to use 2D-CNN kernels which are optimized for spatial pattern recognition.

---

## 📊 8. Results (Honest)

| Scenario | ROC-AUC | Note |
| :--- | :--- | :--- |
| **CHB ➔ CHB** | **0.958** | Near perfect intra-domain fidelity. |
| **SEIZE ➔ SEIZE** | **0.860** | Highly robust on wearable data. |
| **SEIZE ➔ CHB** | **0.722** | Successful "Sparse-to-Dense" transfer. |
| **CHB ➔ SEIZE** | **0.546** | The most difficult trajectory (Domain shift). |

---

## 🌍 9. Generalization & Domain Shift

The shift from **Clinical (23ch)** to **Wearable (2ch)** is an asymmetric problem.
*   **What breaks:** The spatial connectivity features completely disappear.
*   **The solution:** Our **L1 Structural Sparsity** erodes the dead spatial filters, allowing the model to focus only on the longitudinal temporal signatures that exist in both datasets.

---

## 🧠 10. Key Insights

*   **What actually matters:** Normalization. If you normalize incorrectly, you are just training a model to detect the hardware difference between datasets, not the seizures.
*   **What we misunderstood:** We thought more data meant more accuracy. In reality, **diverse** data with sensor dropout is the only way to achieve <0.60 AUC in cross-domain scenarios.

---

## ⚠️ 11. Limitations
*   **Hardware:** Optimized for CPU; GPU latency would be significantly lower.
*   **Real-world:** SEIZEIT2 is still limited in patient count. Performance may vary on "unseen" wearable hardware.

---

## 🚀 12. Future Work
*   **Transfer Learning via Masked Autoencoders:** Pre-training on 10,000+ hours of unlabeled EEG.
*   **Quantization:** Reducing the model size for edge deployment on ESP32/ARM-Cortex wearable boards.

---
*Created as part of the 8th Semester degree project (2026).*
