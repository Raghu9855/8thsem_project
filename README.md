# Cross-Dataset EEG Seizure Detection using CNN-Based Hybrid Deep Learning Models

## Overview
This project compares three CNN-based hybrid deep learning models for EEG seizure detection: CNN + LSTM for temporal sequence modeling, CNN + GNN for channel relationship modeling, and CNN + Swin Transformer for hierarchical long-range dependency learning. Autoencoder-based dimensionality reduction is used instead of PCA to preserve nonlinear EEG patterns, and explainability is added through SHAP and attention analysis.

## Models Evaluated
1. **CNN + LSTM**: Baseline deep learning hybrid for time-series EEG.
2. **CNN + Swin Transformer**: Proposed model for hierarchical and long-range EEG dependency learning. Expectation is that this performs best overall.
3. **CNN + GNN**: Graph-based approach to model spatial connectivity and inter-channel dependency.

## Goal
Show the performance differences between the three models and summarize which hybrid model is best across two distinct datasets (CHB-MIT, SeizeIT2).

## Pipeline Methodology
EEG Data Loading → Bandpass and Notch Filtering → Z-score Normalization → 5-Second Overlapping Segmentation → Annotation-Based Labeling → Data Augmentation (Gaussian Noise, Time Shift, Amplitude Scaling, Co-Mixup) → Time, Frequency, and Nonlinear Feature Extraction → Autoencoder-Based Dimensionality Reduction → Model Comparison (CNN+LSTM, CNN+Swin Transformer, CNN+GNN) → Seizure Classification → Explainability and Performance Evaluation.
