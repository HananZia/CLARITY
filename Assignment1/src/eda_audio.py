
"""
Audio EDA:
- expects either:
    column 'audio_path' in data CSV pointing to files under dataset/audio/
    OR
    a folder dataset/audio/ with filenames referenced in CSV or matching index
- For each audio file: waveform plot, mel-spectrogram, duration calc
- Saves aggregated histograms (durations) and per-sample figures if desired (sampled)
"""

# Standard imports
import os
import matplotlib
matplotlib.use("Agg")  # for headless environments
import pandas as pd # data handling
from pathlib import Path
import numpy as np
import librosa # audio processing
import librosa.display
import matplotlib.pyplot as plt # for plotting
from tqdm import tqdm # progress bars
from utils import PLOTS_DIR, save_fig_pdf # custom utils

# Path setup
DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"
AUDIO_DIR = DATASET_DIR / "audio"
TRAIN_CSV = DATASET_DIR / "train.csv"

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# Find audio paths in dataframe
def find_audio_paths(df):
    # Heuristic: look for common audio column names
    candidates = ["audio_path", "audio", "file", "filename"]
    for c in candidates:
        if c in df.columns:
            return df[c].astype(str).fillna("").tolist()
    # If no column, look for file names matching index in audio dir
    if os.path.isdir(AUDIO_DIR):
        files = set(os.listdir(AUDIO_DIR))
        possible = []
        for idx in df.index:
            # try idx-based filenames
            for ext in [".wav", ".flac", ".mp3", ".ogg"]:
                name = f"{idx}{ext}"
                if name in files:
                    possible.append(name)
                    break
            else:
                possible.append("")  # missing
        return possible
    return [""] * len(df)

# Load dataframe
def load_df():
    if os.path.exists(TRAIN_CSV):
        return pd.read_csv(TRAIN_CSV)
    else:
        raise FileNotFoundError(f"{TRAIN_CSV} not found. Place CSVs into dataset/ or run save_dataset.py")

# Compute audio features
def compute_audio_features(path):
    # Loads audio and returns y, sr, duration, energy, low_energy_fraction
    try:
        y, sr = librosa.load(path, sr=None)  # preserve native sr
        duration = len(y) / sr
        energy = np.sum(y**2) / len(y) if len(y) else 0.0
        # fraction of frames below mean energy as a simple noise/quietness proxy
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        low_frac = float((rms < np.median(rms)).sum()) / max(1, len(rms))
        return {"y": y, "sr": sr, "duration": duration, "energy": energy, "low_energy_frac": low_frac}
    except Exception as e:
        # audio cannot be loaded
        return None

# Plotting functions
def plot_waveform(y, sr, out_name):
    fig, ax = plt.subplots(figsize=(10,3))
    librosa.display.waveshow(y, sr=sr)
    ax.set_title("Waveform")
    save_fig_pdf(fig, out_name)

# Plot mel-spectrogram
def plot_mel_spectrogram(y, sr, out_name):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10,4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel spectrogram")
    save_fig_pdf(fig, out_name)

# Main function
def main(sample_limit=10):
    df = load_df()
    audio_paths = find_audio_paths(df)
    # Convert relative audio paths to full paths if possible
    full_paths = []
    for p in audio_paths:
        if not p:
            full_paths.append(None)
        else:
            candidate = os.path.join(AUDIO_DIR, p)
            if os.path.exists(candidate):
                full_paths.append(candidate)
            elif os.path.exists(p):
                full_paths.append(p)
            else:
                full_paths.append(None)

# Check if any audio files found
    if not any(full_paths):
        print("No audio files found. Skipping audio EDA.")
        return

    durations = []
    low_fracs = []
    energies = []
    processed = 0

    # Process each audio file
    for idx, path in tqdm(enumerate(full_paths)):
        if path is None:
            continue
        feats = compute_audio_features(path)
        if feats is None:
            continue
        durations.append(feats["duration"])
        low_fracs.append(feats["low_energy_frac"])
        energies.append(feats["energy"])

        # save per-sample waveform + spectrogram for a few samples
        if processed < sample_limit:
            basename = f"sample_{idx}_waveform.pdf"
            plot_waveform(feats["y"], feats["sr"], basename)
            plot_mel_spectrogram(feats["y"], feats["sr"], f"sample_{idx}_mel.pdf")
            processed += 1

    # Aggregate plots
    if durations:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(durations, bins=50)
        ax.set_xlabel("Duration (s)")
        ax.set_title("Audio duration histogram")
        save_fig_pdf(fig, "audio_duration_histogram.pdf")

# Energy and low-energy fraction plots
    if energies:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(energies, bins=50)
        ax.set_xlabel("Energy (avg power)")
        ax.set_title("Audio energy histogram")
        save_fig_pdf(fig, "audio_energy_histogram.pdf")

# Low-energy fraction plots
    if low_fracs:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(low_fracs, bins=40)
        ax.set_xlabel("Low-energy frame fraction (proxy for noisy/quiet)")
        ax.set_title("Audio low-energy fraction")
        save_fig_pdf(fig, "audio_low_energy_fraction.pdf")

    print("Audio EDA complete. Plots saved in", PLOTS_DIR)

# Run main
if __name__ == "__main__":
    main()
