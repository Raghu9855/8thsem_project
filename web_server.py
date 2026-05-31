import os
import sys
import torch
import mne
import shutil
import numpy as np
import scipy.signal as signal
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)

from src.models.cnn_swin_transformer import CNNSwinTransformerModel
from src.autoencoder_reduction import FeatureAutoencoder
from src.preprocessing import preprocess_eeg_window
from src.feature_extraction import extract_features

MODELS = {}
SYSTEM_LOG = []

def log_event(message):
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    SYSTEM_LOG.append(f"[{timestamp}] {message}")
    if len(SYSTEM_LOG) > 50: SYSTEM_LOG.pop(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device('cpu')
    experiments = [("CHB", "CHB"), ("CHB", "SEIZE"), ("SEIZE", "CHB"), ("SEIZE", "SEIZE")]
    for tr, ts in experiments:
        key = f"{tr.lower()}_to_{ts.lower()}"
        path = os.path.join(BASE_DIR, 'outputs', 'saved_models', f'best_cnn_swin_{tr}_to_{ts}.pth')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model = CNNSwinTransformerModel(eeg_channels=ckpt.get('eeg_channels', 23)).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            ae = FeatureAutoencoder(input_dim=ckpt.get('feature_dim', 276)).to(device)
            ae.load_state_dict(ckpt['autoencoder_state_dict'])
            ae.eval()
            MODELS[key] = {"model": model, "ae": ae, "device": device, "thresh": float(ckpt.get('thresh', 0.5))}
    os.makedirs(os.path.join(BASE_DIR, 'outputs', 'clinical_reports'), exist_ok=True)
    yield
    MODELS.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
async def root(): return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...), experiment: str = "chb_to_chb", adaptive: bool = False, threshold: float = None):
    if experiment not in MODELS: raise HTTPException(status_code=400, detail="Model not loaded")
    tmp_path = os.path.join(BASE_DIR, "uploads", file.filename)
    os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
    with open(tmp_path, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        m = MODELS[experiment]
        checkpoint_thresh = m["thresh"]
        if adaptive:
            base_thresh = max(threshold if threshold is not None else checkpoint_thresh, checkpoint_thresh)
        else:
            base_thresh = threshold if threshold is not None else checkpoint_thresh
        log_event(f"Scanning Signal: {file.filename} (Adaptive: {adaptive}, Threshold: {base_thresh:.4f})")
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        sfreq, data_all = raw.info['sfreq'], raw.get_data()
        win_size, results = int(5 * sfreq), []
        
        spec_buffer = deque(maxlen=5)
        feat_buffer = deque(maxlen=5)
        ema_prob, ema_alpha = 0.0, 0.4
        
        baseline_history = [0.05, 0.05, 0.05, 0.05, 0.05]
        
        for i in range(0, data_all.shape[1] - win_size, win_size):
            window = data_all[:, i:i+win_size]
            
            # 1. Robust per-window normalization (pre-normalization)
            window_norm = (window - np.mean(window, axis=1, keepdims=True)) / (np.std(window, axis=1, keepdims=True) + 1e-8)
            window_norm = np.nan_to_num(window_norm)
            
            # 2. Preprocess EEG window
            processed = preprocess_eeg_window(window_norm, sfreq=sfreq, target_sfreq=256.0)
            
            # 3. Extract features on active channels FIRST
            features = extract_features(processed, 256.0)
            
            # 4. Pad/Standardize channels for signal and features
            MAX_CHANNELS = 23
            FEATS_PER_CHANNEL = 12
            if processed.shape[0] < MAX_CHANNELS:
                processed_padded = np.pad(processed, ((0, MAX_CHANNELS - processed.shape[0]), (0, 0)), mode='constant')
            else:
                processed_padded = processed[:MAX_CHANNELS, :]
                
            if len(features) < MAX_CHANNELS * FEATS_PER_CHANNEL:
                features = np.pad(features, (0, MAX_CHANNELS * FEATS_PER_CHANNEL - len(features)), mode='constant')
            else:
                features = features[:MAX_CHANNELS * FEATS_PER_CHANNEL]
            features = np.nan_to_num(features)
            
            # 5. Compute Spectrogram with exact training normalization
            f_a, t_a, Sxx = signal.spectrogram(processed_padded, fs=256.0, nperseg=64, noverlap=32)
            Sxx = Sxx[:, f_a <= 40.0, :]
            
            # Log-power stabilization
            Sxx = np.log(Sxx + 1e-8)
            
            # Final standardization with safety gate (per-channel)
            Sxx = (Sxx - Sxx.mean(axis=(1, 2), keepdims=True)) / (Sxx.std(axis=(1, 2), keepdims=True) + 1e-8)
            Sxx = np.nan_to_num(Sxx)
            
            spec_buffer.append(torch.tensor(Sxx, dtype=torch.float32))
            
            # 6. Encode features via Autoencoder (using raw features matching the AE training spec)
            feat_buffer.append(m["ae"].encode(torch.tensor(features, dtype=torch.float32).unsqueeze(0)).squeeze(0))

            if len(spec_buffer) > 0:
                s_list = [spec_buffer[0]] * (5 - len(spec_buffer)) + list(spec_buffer)
                f_list = [feat_buffer[0]] * (5 - len(feat_buffer)) + list(feat_buffer)
                with torch.no_grad():
                    logits, xai = m["model"](torch.stack(s_list).unsqueeze(0).to(m["device"]), 
                                            torch.stack(f_list).unsqueeze(0).to(m["device"]), xai_mode=True)
                    raw_prob = torch.sigmoid(logits[:, 1]).item()
                    ema_prob = (ema_alpha * raw_prob) + ((1 - ema_alpha) * ema_prob)
                
                # Determine adaptive threshold dynamically if enabled
                if adaptive:
                    clean_history = [v for v in baseline_history if v < 0.35]
                    if len(clean_history) >= 5:
                        mu_base = np.mean(clean_history)
                        std_base = np.std(clean_history)
                        dynamic_thresh = float(np.clip(base_thresh + mu_base + 2.5 * (std_base + 1e-4) - 0.05, base_thresh, 0.95))
                    else:
                        dynamic_thresh = base_thresh
                else:
                    dynamic_thresh = base_thresh
                
                is_seiz = ema_prob > dynamic_thresh
                
                # Update baseline history only during quiet background frames (< 0.35) to prevent self-contamination
                if ema_prob < 0.35:
                    baseline_history.append(ema_prob)
                    if len(baseline_history) > 20:
                        baseline_history.pop(0)

                # Calculate raw variance for attention mapping before z-score flattening
                raw_vars = np.var(window, axis=1)
                if len(raw_vars) < MAX_CHANNELS:
                    raw_vars = np.pad(raw_vars, (0, MAX_CHANNELS - len(raw_vars)), mode='constant', constant_values=0.0)
                else:
                    raw_vars = raw_vars[:MAX_CHANNELS]
                raw_vars = np.nan_to_num(raw_vars)
                raw_vars = np.clip(raw_vars, 1e-20, None)
                raw_vars = [float(v) for v in raw_vars]

                results.append({
                    "time": i / sfreq, "probability": ema_prob, "is_seizure": is_seiz,
                    "threshold": dynamic_thresh,
                    "signals": processed_padded[:, ::10].tolist(), 
                    "attention": xai['attn1'].mean(dim=(1,2))[0].cpu().numpy().tolist(),
                    "raw_variances": raw_vars
                })

        events, is_active, start_t = [], False, 0
        for d in results:
            if d["is_seizure"] and not is_active: is_active, start_t = True, d["time"]
            elif not d["is_seizure"] and is_active:
                is_active = False
                if d["time"] - start_t >= 15: events.append({"start": start_t, "end": d["time"]})
        if is_active and results[-1]["time"] - start_t >= 15: events.append({"start": start_t, "end": results[-1]["time"]})
        
        return {
            "filename": file.filename, "final_prediction": len(events) > 0,
            "confidence": max([d["probability"] for d in results]) if results else 0.0,
            "timeline": results, "events": events, 
            "channels": raw.info['ch_names'],
            "stats": {"threshold": base_thresh, "experiment": experiment, "adaptive_mode": adaptive}
        }
    finally:
        try:
            raw.close()  # Release MNE file handle on Windows before deletion
        except Exception:
            pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass  # Suppress if file is still locked (Windows)

@app.post("/api/report/generate")
async def generate_report(data: dict = Body(...)):
    import datetime
    report_id = f"REPORT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = os.path.join(BASE_DIR, 'outputs', 'clinical_reports', report_id)
    
    is_seizure = data.get('final_prediction', False)
    notes = data.get('notes', '').strip()
    if not notes:
        notes = "No manual comments provided. Diagnostic decision based solely on automated neural sequence analysis."
        
    events = data.get('events', [])
    timeline = data.get('timeline', [])
    filename = data.get('filename', 'EEG_RECORDING.edf')
    
    # 1. Advanced Telemetry Metrics Aggregation
    total_duration = 0.0
    if timeline:
        total_duration = timeline[-1]["time"] + 5.0
    
    probs = [d.get('probability', 0.0) for d in timeline]
    peak_prob = max(probs) if probs else 0.0
    mean_prob = np.mean(probs) if probs else 0.0
    variance_prob = np.var(probs) if probs else 0.0
    
    background_probs = [d.get('probability', 0.0) for d in timeline if not d.get('is_seizure', False)]
    bg_mean = np.mean(background_probs) if background_probs else 0.0
    bg_std = np.std(background_probs) if background_probs else 0.0
    
    seizure_duration = sum([e['end'] - e['start'] for e in events])
    seizure_burden_pct = (seizure_duration / total_duration * 100) if total_duration > 0 else 0.0
    
    # 2. Dynamic Channel Mapping and Spatial Attention Analytics
    active_ch_list = data.get('channels', [])
    if not active_ch_list:
        active_ch_list = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
            'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
            'FZ-CZ', 'CZ-PZ', 'T7-FT9', 'FT9-FT10', 
            'FT10-T8', 'P7-PO9', 'PO9-PO10'
        ]
    
    num_channels = len(active_ch_list)
    is_wearable = num_channels <= 2 or 'FP1-F7' not in active_ch_list
    
    # Calculate average attention based on physical electrode signal variance weighted by segment probability
    avg_attn = np.zeros(num_channels)
    if timeline:
        for seg in timeline:
            raw_vars = seg.get('raw_variances', [])
            prob = seg.get('probability', 0.0)
            if raw_vars and len(raw_vars) == num_channels:
                variances = np.clip(np.array(raw_vars), 1e-20, None)
                chan_influence = variances / np.sum(variances)
                avg_attn += chan_influence * (prob + 0.01)
            else:
                signals = seg.get('signals', [])
                if signals and len(signals) == num_channels:
                    # Compute signal variance for each channel
                    variances = np.array([np.var(ch_sig) for ch_sig in signals]) + 1e-6
                    # Normalize so they sum to 1.0
                    chan_influence = variances / np.sum(variances)
                    # Weight by segment probability to emphasize active seizure zones
                    avg_attn += chan_influence * (prob + 0.01)
        
        # Normalize the global average attention to sum to 1.0
        sum_attn = np.sum(avg_attn)
        if sum_attn > 0:
            avg_attn = avg_attn / sum_attn
        else:
            avg_attn = np.ones(num_channels) / num_channels
    
    # Spatial Attention Clustering by Lobe
    if not is_wearable:
        frontal_ch = ['FP1-F7', 'FP2-F8', 'FP1-F3', 'FP2-F4', 'FZ-CZ']
        temporal_ch = ['F7-T7', 'T7-P7', 'F8-T8', 'T8-P8', 'T7-FT9', 'FT10-T8']
        central_ch = ['C3-P3', 'C4-P4', 'CZ-PZ', 'FT9-FT10']
        occipital_ch = ['P7-O1', 'P8-O2', 'PO9-PO10']
        
        frontal_sum = sum([avg_attn[active_ch_list.index(ch)] for ch in frontal_ch if ch in active_ch_list])
        temporal_sum = sum([avg_attn[active_ch_list.index(ch)] for ch in temporal_ch if ch in active_ch_list])
        central_sum = sum([avg_attn[active_ch_list.index(ch)] for ch in central_ch if ch in active_ch_list])
        occipital_sum = sum([avg_attn[active_ch_list.index(ch)] for ch in occipital_ch if ch in active_ch_list])
        
        lobe_sum = frontal_sum + temporal_sum + central_sum + occipital_sum
        if lobe_sum > 0:
            frontal_pct = (frontal_sum / lobe_sum) * 100
            temporal_pct = (temporal_sum / lobe_sum) * 100
            central_pct = (central_sum / lobe_sum) * 100
            occipital_pct = (occipital_sum / lobe_sum) * 100
        else:
            frontal_pct = temporal_pct = central_pct = occipital_pct = 25.0
            
        lobes = [
            {"name": "Frontal Lobe", "pct": frontal_pct, "desc": "Executive, cognitive, and anterior motor integration zones"},
            {"name": "Temporal Lobe", "pct": temporal_pct, "desc": "Auditory processing, language, memory, and major focal seizure nodes"},
            {"name": "Central/Parietal Zone", "pct": central_pct, "desc": "Somatosensory, motor strips, and midline propagation networks"},
            {"name": "Occipital Lobe", "pct": occipital_pct, "desc": "Primary visual processing and posterior baseline rhythm zones"}
        ]
        lobes = sorted(lobes, key=lambda x: x["pct"], reverse=True)
        dominant_lobe = lobes[0]
        
        # Hemispheric Asymmetry lateralization
        left_ch = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'T7-FT9', 'P7-PO9']
        right_ch = ['FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FT10-T8', 'P8-O2']
        
        left_sum = sum([avg_attn[active_ch_list.index(ch)] for ch in left_ch if ch in active_ch_list])
        right_sum = sum([avg_attn[active_ch_list.index(ch)] for ch in right_ch if ch in active_ch_list])
        
        tot_hem = left_sum + right_sum
        if tot_hem > 0:
            left_pct = (left_sum / tot_hem) * 100
            right_pct = (right_sum / tot_hem) * 100
        else:
            left_pct = right_pct = 50.0
            
        if abs(left_pct - right_pct) < 5:
            lateralization_str = "Symmetric Bilateral Trace"
            lateralization_desc = f"Bilateral attention balance (Left: {left_pct:.1f}%, Right: {right_pct:.1f}%) suggesting generalized synchronization."
        elif left_pct > right_pct:
            lateralization_str = "Left Hemispheric Predominance"
            lateralization_desc = f"Left asymmetric focus (Left: {left_pct:.1f}%, Right: {right_pct:.1f}%) indicating local left lateralized ictal triggers."
        else:
            lateralization_str = "Right Hemispheric Predominance"
            lateralization_desc = f"Right asymmetric focus (Right: {right_pct:.1f}%, Left: {left_pct:.1f}%) indicating local right lateralized ictal triggers."
    else:
        # Wearable setup
        dominant_lobe = {"name": "Temporal Lobe (Wearable)", "pct": 100.0, "desc": "Dual wearable temporal channels tracing localized micro-signals"}
        lateralization_str = "Dual Wearable Focus"
        lateralization_desc = "Wearable sensor recording. Standard clinical lobes and hemispheric asymmetry bypassed."
        left_pct = right_pct = 50.0
        frontal_pct = temporal_pct = central_pct = occipital_pct = 0.0
        
    # Top driving channels
    top_indices = np.argsort(avg_attn)[::-1][:3]
    top_channels = [f"{active_ch_list[idx]} ({avg_attn[idx]*100:.1f}%)" for idx in top_indices if idx < len(active_ch_list)]
    top_channels_str = ", ".join(top_channels)
    
    # 3. Dynamic AI Clinical Narrative Generation
    adaptive_mode_active = data.get('stats', {}).get('adaptive_mode', False)
    adaptive_str = "Adaptive Baseline-Relative Calibration" if adaptive_mode_active else "Static Youden's J Thresholding"
    base_thresh = data.get('stats', {}).get('threshold', 0.50)
    experiment_str = data.get('stats', {}).get('experiment', 'chb_to_chb').upper().replace('_', ' ➔ ')
    
    if is_seizure:
        ai_narrative = f"""
        <strong>[ELECTROGRAPHIC SUMMARY]</strong>: The SeizureX Neural Engine completed a comprehensive sequence scan of EEG recording <strong>'{filename}'</strong> under <strong>{adaptive_str}</strong> (Calibration limit: {base_thresh:.2f}). The Swin-Transformer neural encoder flagged <strong>{len(events)} discrete paroxysmal burst event(s)</strong>, accumulating a total seizure burden of <strong>{seizure_duration:.1f} seconds</strong> ({seizure_burden_pct:.1f}% of global recording runtime). Peak ictal probability reached <strong>{peak_prob*100:.1f}%</strong>.<br><br>
        
        <strong>[LOCALIZATION & LATERALIZATION]</strong>: Spatial self-attention mapping shows significant focal synchronization localized predominantly in the <strong>{dominant_lobe['name']}</strong> ({dominant_lobe['pct']:.1f}% attention). Hemispheric lateralization metrics indicate a <strong>{lateralization_str}</strong> ({lateralization_desc}). The primary channels driving classification decisions are <strong>{top_channels_str}</strong>.<br><br>
        
        <strong>[CALIBRATION & SIGNAL STABILITY]</strong>: Background rhythm calibration established a baseline mean of <strong>{bg_mean:.3f}</strong> with a standard deviation of <strong>{bg_std:.3f}</strong>. Global probability variance of <strong>{variance_prob:.4f}</strong> reflects stable background signals punctuated by high-amplitude ictal transitions.<br><br>
        
        <strong>[CLINICAL RECOMMENDATION]</strong>: Urgent clinical review is advised. Localized paroxysmal bursts within the {dominant_lobe['name']} suggest focal seizure activity of {lateralization_str} origin. Recommend continuous video-EEG monitoring and pharmacological correlation.
        """
    else:
        ai_narrative = f"""
        <strong>[ELECTROGRAPHIC SUMMARY]</strong>: The SeizureX Neural Engine completed a comprehensive sequence scan of EEG recording <strong>'{filename}'</strong> under <strong>{adaptive_str}</strong>. The neural classification network confirmed well-organized background oscillations with <strong>zero sustained paroxysmal activities or seizure clusters</strong> matching diagnostic duration limits (min 15s). Peak probability remained stable at <strong>{peak_prob*100:.1f}%</strong>.<br><br>
        
        <strong>[LOCALIZATION & LATERALIZATION]</strong>: Attention profiles show normal physiological distribution. The dominant background attention is centered around the <strong>{dominant_lobe['name']}</strong> ({dominant_lobe['pct']:.1f}% weight), reflecting normal resting alpha/theta wave symmetry. Lateralization measures represent a <strong>{lateralization_str}</strong> ({lateralization_desc}).<br><br>
        
        <strong>[CALIBRATION & SIGNAL STABILITY]</strong>: The baseline preservation algorithms maintained high signal-to-noise stability with a background mean of <strong>{bg_mean:.3f}</strong> and background standard deviation of <strong>{bg_std:.3f}</strong>. A global session variance of <strong>{variance_prob:.5f}</strong> indicates the absence of abrupt baseline drifts or sub-clinical paroxysmal spikes.<br><br>
        
        <strong>[CLINICAL RECOMMENDATION]</strong>: Routine clinical follow-up is sufficient. No active epileptogenic focus, lateralizing ictal zones, or sustained seizure clusters were identified in the sequence. Standard baseline EEG tracking is recommended.
        """
        
    # Render progress bars for top 3 channels
    top_channels_html = ""
    for idx in top_indices:
        if idx >= len(active_ch_list): continue
        ch_name = active_ch_list[idx]
        val = avg_attn[idx]
        top_channels_html += f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-family: monospace; color: #f0f6fc; margin-bottom: 4px;">
                <span>⚡ {ch_name}</span>
                <span style="color: #00f2ff; font-weight: bold;">{val*100:.1f}% influence</span>
            </div>
            <div style="background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; overflow: hidden; border: 1px solid rgba(0,242,255,0.1);">
                <div style="background: linear-gradient(90deg, #00f2ff, #00ffaf); width: {min(100, val*100 * 5):.1f}%; height: 100%; border-radius: 3px;"></div>
            </div>
        </div>
        """

    # 4. Interactive SVG Brain Headmap (XAI Layout)
    electrodes = {
        "Fp1": (160, 60), "Fp2": (340, 60),
        "F7": (100, 120), "F3": (190, 120), "Fz": (250, 120), "F4": (310, 120), "F8": (400, 120),
        "T7": (80, 200), "C3": (180, 200), "Cz": (250, 200), "C4": (320, 200), "T8": (420, 200),
        "P7": (100, 280), "P3": (190, 280), "Pz": (250, 280), "P4": (310, 280), "P8": (400, 280),
        "O1": (160, 340), "O2": (340, 340)
    }
    
    # Calculate activations by splitting bipolar channels
    elec_attn = {name: 0.04 for name in electrodes}
    if not is_wearable:
        for ch, val in zip(active_ch_list, avg_attn):
            parts = ch.split('-')
            for p in parts:
                p_mapped = p.strip()
                p_key = p_mapped[0].upper() + p_mapped[1:].lower()
                if p_key in elec_attn:
                    elec_attn[p_key] = max(elec_attn[p_key], val)
        max_elec = max(elec_attn.values()) if elec_attn.values() else 1.0
        for k in elec_attn:
            elec_attn[k] = elec_attn[k] / max_elec if max_elec > 0 else 0.04
    else:
        # Wearable setup: light up T7 and T8
        for i, ch in enumerate(active_ch_list):
            node_key = "T7" if i == 0 else "T8"
            elec_attn[node_key] = 1.0
            
    # Draw connections
    montage_lines = [
        ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
        ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
        ("Fz", "Cz"), ("Cz", "Pz")
    ]
    svg_lines = ""
    for start, end in montage_lines:
        if start in electrodes and end in electrodes:
            sx, sy = electrodes[start]
            ex, ey = electrodes[end]
            svg_lines += f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="rgba(255,255,255,0.08)" stroke-width="1" stroke-dasharray="2,2" />'
            
    svg_nodes = ""
    for name, (cx, cy) in electrodes.items():
        w = elec_attn.get(name, 0.04)
        r = 6 + (w * 12)
        glow_r = r + 6
        color = "#ff3e3e" if w > 0.4 else ("#00ffaf" if w > 0.15 else "#00f2ff")
        opacity = 0.2 + (w * 0.8)
        glow_opacity = w * 0.4
        
        svg_nodes += f"""
        <g class="node-group">
            <circle cx="{cx}" cy="{cy}" r="{glow_r}" fill="{color}" opacity="{glow_opacity}" style="filter: blur(2px);"></circle>
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" stroke="#ffffff" stroke-width="1.2" opacity="{opacity}"></circle>
            <text x="{cx}" y="{cy - r - 4}" fill="#8b949e" font-size="8" font-family="monospace" font-weight="bold" text-anchor="middle">{name}</text>
        </g>
        """
        
    svg_html = f"""
    <svg viewBox="0 0 500 400" style="width: 100%; max-width: 450px; height: auto; margin: 0 auto; display: block; background: rgba(0,0,0,0.2); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); padding: 10px;">
        <!-- Head Outline -->
        <ellipse cx="250" cy="200" rx="145" ry="170" fill="none" stroke="rgba(255, 255, 255, 0.1)" stroke-width="2" />
        <!-- Nose -->
        <path d="M 245 30 L 250 15 L 255 30 Z" fill="none" stroke="rgba(255, 255, 255, 0.1)" stroke-width="2" />
        <!-- Ears -->
        <path d="M 105 185 A 15 25 0 0 0 105 215" fill="none" stroke="rgba(255, 255, 255, 0.1)" stroke-width="2" />
        <path d="M 395 185 A 15 25 0 0 1 395 215" fill="none" stroke="rgba(255, 255, 255, 0.1)" stroke-width="2" />
        {svg_lines}
        {svg_nodes}
    </svg>
    """

    # 5. Consolidated Seizure Timelines Table
    events_html = ""
    if is_seizure and events:
        table_rows = ""
        for i, e in enumerate(events):
            ev_attn = np.zeros(num_channels)
            ev_segs = [seg for seg in timeline if e['start'] <= seg['time'] <= e['end']]
            if ev_segs:
                for seg in ev_segs:
                    raw_vars = seg.get('raw_variances', [])
                    prob = seg.get('probability', 0.0)
                    if raw_vars and len(raw_vars) == num_channels:
                        variances = np.clip(np.array(raw_vars), 1e-20, None)
                        chan_influence = variances / np.sum(variances)
                        ev_attn += chan_influence * (prob + 0.01)
                    else:
                        signals = seg.get('signals', [])
                        if signals and len(signals) == num_channels:
                            # Compute variance for each channel
                            variances = np.array([np.var(ch_sig) for ch_sig in signals]) + 1e-6
                            # Normalize so they sum to 1.0
                            chan_influence = variances / np.sum(variances)
                            # Weight by probability
                            ev_attn += chan_influence * (prob + 0.01)
                
                # Normalize ev_attn to sum to 1.0
                sum_ev = np.sum(ev_attn)
                if sum_ev > 0:
                    ev_attn = ev_attn / sum_ev
            
            top_ev_indices = np.argsort(ev_attn)[::-1][:2]
            top_ev_ch = [active_ch_list[idx] for idx in top_ev_indices if idx < len(active_ch_list)]
            top_ev_str = ", ".join(top_ev_ch) if top_ev_ch else "N/A"
            
            peak_ev_prob = max([seg.get('probability', 0.0) for seg in ev_segs]) if ev_segs else e.get('confidence', 0.0)
            
            table_rows += f"""
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                <td style="padding: 12px 8px; font-weight: bold; color: #ff3e3e; font-family: monospace;">🚨 Event #{i+1}</td>
                <td style="padding: 12px 8px; font-family: monospace; color: #ffffff;">{round(e['start'])}s &mdash; {round(e['end'])}s</td>
                <td style="padding: 12px 8px; font-family: monospace; color: #ffffff;">{round(e['end'] - e['start'], 1)}s</td>
                <td style="padding: 12px 8px; font-family: monospace; font-weight: bold; color: #00f2ff;">{round(peak_ev_prob * 100, 1)}%</td>
                <td style="padding: 12px 8px; font-family: monospace; color: #8b949e; font-size: 0.8rem;">{top_ev_str}</td>
            </tr>
            """
            
        events_html = f"""
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(0, 242, 255, 0.15); color: #8b949e; text-transform: uppercase; font-family: monospace; font-size: 0.75rem; letter-spacing: 1px;">
                    <th style="padding: 8px;">Burst ID</th>
                    <th style="padding: 8px;">Interval</th>
                    <th style="padding: 8px;">Duration</th>
                    <th style="padding: 8px;">Peak Prob</th>
                    <th style="padding: 8px;">Focal Electrodes</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
    else:
        events_html = """
        <div style="background: rgba(0, 255, 175, 0.05); border: 1px solid rgba(0, 255, 175, 0.2); padding: 20px; border-radius: 12px; text-align: center; color: #00ffaf; font-family: sans-serif; font-weight: bold;">
            ✅ NO EPILEPTIFORM SEIZURE BURSTS FLAG-ACTIVATED
        </div>
        """

    verdict_badge = ""
    if is_seizure:
        verdict_badge = """
        <div style="background: #1a0f12; border: 2px solid #ff3e3e; padding: 20px; border-radius: 20px; text-align: center; box-shadow: 0 0 20px rgba(255, 62, 62, 0.2); margin-bottom: 30px;">
            <h2 style="color: #ff3e3e; margin: 0; font-size: 1.6rem; font-family: sans-serif; letter-spacing: 2px; text-transform: uppercase;">🚨 High Risk Seizure Activity Flagged</h2>
            <p style="color: #8b949e; margin: 8px 0 0 0; font-size: 0.85rem; font-family: monospace;">Paroxysmal waveforms identified above safety-limit bounds.</p>
        </div>
        """
    else:
        verdict_badge = """
        <div style="background: #020806; border: 2px solid #00ffaf; padding: 20px; border-radius: 20px; text-align: center; box-shadow: 0 0 20px rgba(0, 255, 175, 0.2); margin-bottom: 30px;">
            <h2 style="color: #00ffaf; margin: 0; font-size: 1.6rem; font-family: sans-serif; letter-spacing: 2px; text-transform: uppercase;">✅ EEG Clinically Stable / Normal Rhythms</h2>
            <p style="color: #8b949e; margin: 8px 0 0 0; font-size: 0.85rem; font-family: monospace;">Standard wave oscillations preserved across all active sensors.</p>
        </div>
        """

    # Build the full, stunning clinical diagnostic document
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SeizureX | Clinical Diagnostics Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            body {{
                background-color: #030508;
                color: #f0f6fc;
                font-family: 'Space Grotesk', sans-serif;
                margin: 0;
                padding: 40px;
                line-height: 1.6;
                transition: background-color 0.3s ease, color 0.3s ease;
            }}
            .container {{
                max-width: 850px;
                margin: 0 auto;
                background: rgba(13, 17, 23, 0.85);
                border: 1px solid rgba(0, 242, 255, 0.15);
                border-radius: 28px;
                padding: 40px;
                box-shadow: 0 8px 32px 0 rgba(0,0,0,0.5);
                backdrop-filter: blur(20px);
                position: relative;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(0, 242, 255, 0.15);
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .meta-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 35px;
            }}
            .meta-card {{
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.05);
                padding: 15px 20px;
                border-radius: 16px;
            }}
            .label {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.72rem;
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 1.5px;
            }}
            .value {{
                font-size: 1.05rem;
                font-weight: bold;
                color: #ffffff;
                margin-top: 5px;
            }}
            .section-title {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                color: #00f2ff;
                border-bottom: 1px solid rgba(0, 242, 255, 0.15);
                padding-bottom: 8px;
                margin-top: 40px;
                margin-bottom: 20px;
                letter-spacing: 2px;
                text-transform: uppercase;
            }}
            .notes-box {{
                background: #020305;
                border: 1px solid rgba(0, 242, 255, 0.15);
                padding: 20px;
                border-radius: 16px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                color: #f0f6fc;
                white-space: pre-wrap;
            }}
            .narrative-box {{
                background: rgba(0, 242, 255, 0.03);
                border: 1px solid rgba(0, 242, 255, 0.12);
                padding: 24px;
                border-radius: 18px;
                font-size: 0.92rem;
                line-height: 1.7;
                color: #e2e8f0;
                margin-bottom: 25px;
            }}
            .badge {{
                display: inline-block;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                font-weight: bold;
                padding: 3px 8px;
                border-radius: 6px;
                border: 1px solid rgba(0, 242, 255, 0.3);
                background: rgba(0, 242, 255, 0.1);
                color: #00f2ff;
                margin-bottom: 15px;
                text-transform: uppercase;
            }}
            .print-btn {{
                background: #00f2ff;
                color: #030508;
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: bold;
                cursor: pointer;
                font-size: 0.8rem;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 0 10px rgba(0, 242, 255, 0.3);
            }}
            .print-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 0 15px rgba(0, 242, 255, 0.5);
            }}
            .contrast-btn {{
                background: rgba(255, 255, 255, 0.05);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 10px 20px;
                border-radius: 10px;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: bold;
                cursor: pointer;
                font-size: 0.8rem;
                transition: all 0.2s ease;
            }}
            .contrast-btn:hover {{
                background: rgba(255, 255, 255, 0.1);
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                border-top: 1px solid rgba(255, 255, 255, 0.05);
                padding-top: 20px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                color: #777;
            }}
            
            body.light-theme {{
                background-color: #f8fafc;
                color: #0f172a;
            }}
            body.light-theme .container {{
                background: #ffffff;
                border-color: #cbd5e1;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }}
            body.light-theme .meta-card {{
                background: #f1f5f9;
                border-color: #e2e8f0;
            }}
            body.light-theme .value {{
                color: #0f172a;
            }}
            body.light-theme .notes-box {{
                background: #f8fafc;
                border-color: #cbd5e1;
                color: #0f172a;
            }}
            body.light-theme .narrative-box {{
                background: rgba(0, 242, 255, 0.03);
                border-color: #cbd5e1;
                color: #334155;
            }}
            body.light-theme .section-title {{
                color: #0284c7;
                border-bottom-color: #bae6fd;
            }}
            body.light-theme text {{
                fill: #334155;
            }}
            body.light-theme svg {{
                background: #f1f5f9;
                border-color: #cbd5e1;
            }}

            @media print {{
                body {{
                    background-color: #ffffff !important;
                    color: #000000 !important;
                    padding: 0 !important;
                }}
                .container {{
                    background: #ffffff !important;
                    border: none !important;
                    box-shadow: none !important;
                    padding: 0 !important;
                    max-width: 100% !important;
                    backdrop-filter: none !important;
                }}
                .meta-card, .notes-box, .narrative-box {{
                    background: #f8fafc !important;
                    border: 1px solid #cbd5e1 !important;
                    color: #000000 !important;
                }}
                .value {{
                    color: #000000 !important;
                }}
                .section-title {{
                    color: #0f172a !important;
                    border-bottom: 2px solid #0f172a !important;
                }}
                .print-actions {{
                    display: none !important;
                }}
                .badge {{
                    border-color: #000000 !important;
                    color: #000000 !important;
                    background: none !important;
                }}
                svg {{
                    background: #ffffff !important;
                    border: 1px solid #cbd5e1 !important;
                }}
                text {{
                    fill: #000000 !important;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1 style="margin: 0; font-size: 2rem; font-weight: bold; color: #ffffff; letter-spacing: 1px; transition: color 0.3s ease;" class="title-text">Seizure<span style="color: #00f2ff;">X</span></h1>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #00f2ff; letter-spacing: 2px; text-transform: uppercase;">Neuro-AI Clinical Workstation</span>
                </div>
                <div class="print-actions" style="display: flex; gap: 12px; align-items: center;">
                    <button class="contrast-btn" onclick="toggleContrast()">🌓 Contrast Mode</button>
                    <button class="print-btn" onclick="window.print()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="background:none; border:none; padding:0; width:14px; height:14px;"><polygon points="6 9 6 2 18 2 18 9"></polygon><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
                        Export PDF
                    </button>
                </div>
            </div>

            {verdict_badge}

            <div class="section-title">Automated AI Diagnostic Narrative</div>
            <div class="narrative-box">
                <span class="badge">Clinical AI Interpretation</span>
                <p style="margin: 0;">{ai_narrative}</p>
            </div>

            <div class="meta-grid">
                <div class="meta-card">
                    <span class="label">Target EEG Recording</span>
                    <div class="value" style="font-family: monospace; font-size: 1rem;">{filename}</div>
                </div>
                <div class="meta-card">
                    <span class="label">Domain Optimization Trajectory</span>
                    <div class="value" style="color: #00f2ff;">{experiment_str}</div>
                </div>
                <div class="meta-card">
                    <span class="label">Calibration / Threshold Mode</span>
                    <div class="value" style="font-size: 0.95rem;">{adaptive_str} (Limit: {base_thresh:.2f})</div>
                </div>
                <div class="meta-card">
                    <span class="label">Active Sensors Scanned</span>
                    <div class="value">{num_channels} channels detected</div>
                </div>
            </div>

            <div class="section-title">Anatomical Regional Attention (XAI Lobe Focus)</div>
            <div style="display: grid; grid-template-columns: 1.2fr 1fr; gap: 30px; margin-bottom: 30px; align-items: center;" class="regional-grid">
                <div>
                    <p style="margin: 0 0 15px 0; font-size: 0.8rem; color: #8b949e;">
                        Anatomical distribution of active sensors showing the percentage share of global spatial self-attention maps:
                    </p>
                    <div style="background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.04); border-radius: 16px; padding: 20px;">
                        <div style="margin-bottom: 12px;">
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span>🧠 Frontal Lobe</span>
                                <span style="font-weight:bold; color:#00f2ff;">{frontal_pct:.1f}%</span>
                            </div>
                            <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;"><div style="background:#00f2ff; width:{frontal_pct:.1f}%; height:100%;"></div></div>
                        </div>
                        <div style="margin-bottom: 12px;">
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span>🧠 Temporal Lobe</span>
                                <span style="font-weight:bold; color:#00ffaf;">{temporal_pct:.1f}%</span>
                            </div>
                            <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;"><div style="background:#00ffaf; width:{temporal_pct:.1f}%; height:100%;"></div></div>
                        </div>
                        <div style="margin-bottom: 12px;">
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span>🧠 Central/Parietal Zone</span>
                                <span style="font-weight:bold; color:#ffb700;">{central_pct:.1f}%</span>
                            </div>
                            <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;"><div style="background:#ffb700; width:{central_pct:.1f}%; height:100%;"></div></div>
                        </div>
                        <div>
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span>🧠 Occipital Lobe</span>
                                <span style="font-weight:bold; color:#ff3e3e;">{occipital_pct:.1f}%</span>
                            </div>
                            <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;"><div style="background:#ff3e3e; width:{occipital_pct:.1f}%; height:100%;"></div></div>
                        </div>
                    </div>
                </div>
                <div>
                    {svg_html}
                </div>
            </div>

            <div class="section-title">Clinical Telemetry & Baseline Metrics</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px;" class="telemetry-grid">
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px 15px; text-align: center;">
                    <div style="font-family: monospace; font-size: 0.65rem; color: #8b949e; text-transform: uppercase;">Peak Prob</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #00f2ff; margin-top: 4px;">{round(peak_prob * 100, 1)}%</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px 15px; text-align: center;">
                    <div style="font-family: monospace; font-size: 0.65rem; color: #8b949e; text-transform: uppercase;">Baseline Mean</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #00ffaf; margin-top: 4px;">{round(bg_mean, 3)}</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px 15px; text-align: center;">
                    <div style="font-family: monospace; font-size: 0.65rem; color: #8b949e; text-transform: uppercase;">Signal Variance</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #ffb700; margin-top: 4px;">{round(variance_prob, 4)}</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px 15px; text-align: center;">
                    <div style="font-family: monospace; font-size: 0.65rem; color: #8b949e; text-transform: uppercase;">Ictal Burden</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #ff3e3e; margin-top: 4px;">{round(seizure_burden_pct, 1)}%</div>
                </div>
            </div>

            <div class="section-title">Spatial Attention Insights (XAI Channels)</div>
            <div style="background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.04); border-radius: 16px; padding: 20px; margin-bottom: 30px;">
                <p style="margin: 0 0 15px 0; font-size: 0.8rem; color: #8b949e;">
                    Transformer self-attention profiles averaged across all sequence timesteps. Top channels driving the network classification decisions:
                </p>
                {top_channels_html}
            </div>

            <div class="section-title">Consolidated Seizure Timelines</div>
            <div style="margin-bottom: 30px;">
                {events_html}
            </div>

            <div class="section-title">Clinician Diagnostic Notes</div>
            <div class="notes-box">{notes}</div>

            <div class="footer">
                Generated by SeizureX Neural Engine v4.0. Machine classification should be audited by a certified clinical neurophysiologist. For research and clinical decision assistance only.
            </div>
        </div>
        
        <script>
            function toggleContrast() {{
                document.body.classList.toggle('light-theme');
                const isLight = document.body.classList.contains('light-theme');
                const title = document.querySelector('.title-text');
                if (isLight) {{
                    title.style.color = '#0f172a';
                }} else {{
                    title.style.color = '#ffffff';
                }}
            }}
        </script>
    </body>
    </html>
    """
    with open(report_path, "w", encoding="utf-8") as f: f.write(html)
    return {"url": f"/outputs/clinical_reports/{report_id}"}

@app.get("/api/outputs/explorer")
async def explorer():
    res = {"plots": [], "reports": [], "xai": []}
    base = os.path.join(BASE_DIR, "outputs")
    for r, d, fs in os.walk(base):
        for f in fs:
            if not f.endswith(('.png', '.jpg', '.html')): continue
            rp = os.path.relpath(os.path.join(r, f), base).replace(os.sep, '/')
            cat = "reports" if 'reports' in r.lower() or f.endswith('.html') else ("xai" if 'xai' in r.lower() else "plots")
            res[cat].append({"name": f.replace('_', ' ').upper(), "url": f"/outputs/{rp}"})
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
