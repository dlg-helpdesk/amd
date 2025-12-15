import librosa
import numpy as np
import io

TARGET_DURATION = 3.0  # seconds
TARGET_SR = 16000  # sampling rate
N_MFCC = 40
N_FFT = 512

def pcm16_to_float32(audio_bytes):
    """
    Convert 16-bit PCM bytes to float32 in range [-1, 1]
    """
    # Interpret bytes as int16
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    # Normalize to [-1, 1] float32
    audio_float = audio_int16.astype(np.float32) / 32768.0
    return audio_float

def extract_features(input_data, mode="path", target_duration=TARGET_DURATION, target_sr=TARGET_SR, n_mfcc=N_MFCC, n_fft=N_FFT):
    """
    Extract MFCC features from an audio file with fixed duration.

    Steps:
    - Resample audio to target_sr
    - Pad or truncate to target_duration
    - Extract MFCCs with n_mfcc coefficients and n_fft window
    """
    # Load and resample
    if mode == 'bytes':
        y, sr = librosa.load(io.BytesIO(input_data), sr=target_sr)
    else:
        y, sr = librosa.load(input_data, sr=target_sr)

    # Calculate target number of samples
    target_samples = int(target_duration * target_sr)

    # Pad or truncate audio to target duration
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)

    # Average MFCCs over time
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    return mfccs_scaled
