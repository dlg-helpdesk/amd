# api.py
import glob
import os
import datetime

import librosa
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from feature_extractor import extract_features, TARGET_DURATION, TARGET_SR
import io
import soundfile as sf
import pickle
import uuid

# -----------------------------
# Load model once on startup
# -----------------------------
MODEL_PATH = "model/audio_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

app = FastAPI(title="Audio Classification API")


# -----------------------------
# REST API Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file bytes
        file_bytes = await file.read()
        features = extract_features(file_bytes, mode="bytes").reshape(1, -1)

        # Prediction
        pred_encoded = model.predict(features)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]


        return {
            "filename": file.filename,
            "prediction": pred_label.lower()
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Constants ---
VICIDIAL_SR = 8000       # PCM rate from VICIdial
BYTES_PER_SAMPLE = 2
SAMPLE_LIMIT = int(VICIDIAL_SR * TARGET_DURATION * BYTES_PER_SAMPLE)
DEBUG_DIR = "debug_wavs"
DEBUG_RECORDING_LIMIT = 26000

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] WebSocket connection accepted.")
    buffer_bytes = bytearray()
    os.makedirs(DEBUG_DIR, exist_ok=True)

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer_bytes.extend(data)
            print(f"[INFO] Received {len(data)} bytes, buffer size: {len(buffer_bytes)}")

            # Process only full 3-second chunks
            num_full_chunks = len(buffer_bytes) // SAMPLE_LIMIT

            for i in range(num_full_chunks):
                chunk_bytes = buffer_bytes[i*SAMPLE_LIMIT:(i+1)*SAMPLE_LIMIT]

                # Convert PCM int16 -> float32 [-1,1]
                pcm_data = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                prev_pcm_data = pcm_data
                # Resample to model rate
                if VICIDIAL_SR != TARGET_SR:
                    pcm_data = librosa.resample(pcm_data, orig_sr=VICIDIAL_SR, target_sr=TARGET_SR)

                # Wrap in WAV bytes for feature extractor
                buf = io.BytesIO()
                sf.write(buf, pcm_data, samplerate=TARGET_SR, format="WAV")
                buf.seek(0)
                wav_bytes = buf.read()

                # Extract features & predict
                features = extract_features(wav_bytes, mode="bytes").reshape(1, -1)
                pred_encoded = model.predict(features)[0]
                pred_label = le.inverse_transform([pred_encoded])[0]
                pred_label_lower = pred_label.lower()

                await websocket.send_json({"prediction": pred_label_lower})
                print(f"[INFO] Prediction sent: {pred_label_lower}")

                existing_files = glob.glob(os.path.join(DEBUG_DIR, "*.wav"))
                # Save debug WAV with label and unique ID
                if len(existing_files) >= DEBUG_RECORDING_LIMIT:
                    # Do NOT save anymore
                    print(f"[INFO] Debug WAV limit reached ({DEBUG_RECORDING_LIMIT}). Skipping save.")
                else:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = uuid.uuid4().hex[:8]
                    debug_path = f"debug_wavs/debug_{timestamp}_{unique_id}_{pred_label_lower}.wav"
                    sf.write(debug_path, prev_pcm_data, samplerate=VICIDIAL_SR, format="WAV")
                    print(f"[INFO] Saved debug WAV: {debug_path}")

            # Remove only processed chunks
            buffer_bytes = buffer_bytes[num_full_chunks*SAMPLE_LIMIT:]

    except WebSocketDisconnect:
        print("[INFO] Client disconnected")






# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
