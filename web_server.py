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
async def predict(file: UploadFile = File(...), experiment: str = "chb_to_chb"):
    if experiment not in MODELS: raise HTTPException(status_code=400, detail="Model not loaded")
    tmp_path = os.path.join(BASE_DIR, "uploads", file.filename)
    os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
    with open(tmp_path, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        m = MODELS[experiment]
        log_event(f"Scanning Signal: {file.filename}")
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        sfreq, data_all = raw.info['sfreq'], raw.get_data()
        win_size, results = int(5 * sfreq), []
        
        spec_buffer = deque(maxlen=5)
        feat_buffer = deque(maxlen=5)
        ema_prob, ema_alpha = 0.0, 0.4
        
        for i in range(0, data_all.shape[1] - win_size, win_size):
            window = data_all[:, i:i+win_size]
            processed = preprocess_eeg_window(window, sfreq=sfreq, target_sfreq=256.0)
            if processed.shape[0] < 23: processed = np.pad(processed, ((0, 23 - processed.shape[0]), (0, 0)))
            else: processed = processed[:23, :]
            
            f_a, t_a, Sxx = signal.spectrogram(processed, fs=256.0, nperseg=64, noverlap=32)
            Sxx = (np.log(Sxx[:, f_a <= 40.0, :] + 1e-8) + 8) / 10 # Scaled
            spec_buffer.append(torch.tensor(Sxx, dtype=torch.float32))
            
            raw_feats = extract_features(processed, 256.0)
            norm_feats = (raw_feats - np.mean(raw_feats)) / (np.std(raw_feats) + 1e-8)
            feat_buffer.append(m["ae"].encode(torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0)).squeeze(0))

            if len(spec_buffer) > 0:
                s_list = [spec_buffer[0]] * (5 - len(spec_buffer)) + list(spec_buffer)
                f_list = [feat_buffer[0]] * (5 - len(feat_buffer)) + list(feat_buffer)
                with torch.no_grad():
                    logits, xai = m["model"](torch.stack(s_list).unsqueeze(0).to(m["device"]), 
                                            torch.stack(f_list).unsqueeze(0).to(m["device"]), xai_mode=True)
                    raw_prob = torch.sigmoid(logits[:, 1]).item()
                    ema_prob = (ema_alpha * raw_prob) + ((1 - ema_alpha) * ema_prob)
                results.append({
                    "time": i / sfreq, "probability": ema_prob, "is_seizure": ema_prob > m["thresh"],
                    "signal": processed[0, ::10].tolist(), 
                    "attention": xai['attn1'].mean(dim=(1,2))[0].cpu().numpy().tolist()
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
            "timeline": results, "events": events, "stats": {"threshold": m["thresh"], "experiment": experiment}
        }
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

@app.post("/api/report/generate")
async def generate_report(data: dict = Body(...)):
    import datetime
    report_id = f"REPORT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = os.path.join(BASE_DIR, 'outputs', 'clinical_reports', report_id)
    html = f"""
    <html><body style="font-family: sans-serif; padding: 40px; color: #333;">
        <h1 style="color: #004a99;">Clinical Seizure Analysis Report</h1>
        <hr>
        <p><b>Target File:</b> {data['filename']}</p>
        <p><b>Analysis Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p><b>Experiment Trajectory:</b> {data['stats']['experiment'].upper()}</p>
        <h2 style="color: #d32f2f;">Verdict: {'SEIZURE DETECTED' if data['final_prediction'] else 'NO SEIZURE DETECTED'}</h2>
        <h3>Consolidated Events:</h3>
        <ul>{' '.join([f"<li>Seizure from {round(e['start'])}s to {round(e['end'])}s</li>" for e in data['events']]) if data['events'] else '<li>No clinical seizures identified</li>'}</ul>
        <hr>
        <p style="font-size: 0.8rem; color: #777;">Generated by NeuroGuardian Neural Engine v4.0</p>
    </body></html>
    """
    with open(report_path, "w") as f: f.write(html)
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
