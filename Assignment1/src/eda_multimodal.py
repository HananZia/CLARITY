"""
Multimodal EDA:
- alignment verification: how many rows have text only, audio only, both, or neither
- simple cross-modal correlation: token length vs audio duration (if available)
- saves small plots as PDFs
"""

# Imports
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless environments
from pathlib import Path
import matplotlib.pyplot as plt # for plotting
from collections import Counter # for counting

from utils import PLOTS_DIR, save_fig_pdf

# Path setup
DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"
AUDIO_DIR = DATASET_DIR / "audio"
TRAIN_CSV = DATASET_DIR / "train.csv"

# Ensure plots directory exists
def load_df():
    if TRAIN_CSV.exists():
        return pd.read_csv(TRAIN_CSV)
    else:
        raise FileNotFoundError(f"{TRAIN_CSV} not found. Place CSVs into dataset/ or run save_dataset.py")

# Check if a row has audio data
def find_audio_in_row(row):
    # heuristics like in eda_audio
    for key in ["audio_path", "audio", "file", "filename"]:
        if key in row and pd.notna(row.get(key)) and str(row.get(key)).strip():
            return True
    return False

# Compute audio durations map
def compute_durations_map():
    durations = {}
    if not os.path.isdir(AUDIO_DIR):
        return durations
    import librosa
    for fname in os.listdir(AUDIO_DIR):
        path = os.path.join(AUDIO_DIR, fname)
        try:
            y, sr = librosa.load(path, sr=None)
            durations[fname] = len(y) / sr
        except Exception:
            continue
    return durations

# Main function
def main():
    df = load_df()
    text_col = "interview_answer" if "interview_answer" in df.columns else "text"
    df[text_col] = df[text_col].fillna("").astype(str)
    df["has_text"] = df[text_col].str.strip().astype(bool)

    # has audio?
    audio_col_candidates = [c for c in df.columns if c.lower() in {"audio_path","audio","file","filename"}]
    audio_col = audio_col_candidates[0] if audio_col_candidates else None
    if audio_col:
        df["has_audio"] = df[audio_col].fillna("").astype(bool)
    else:
        # try match to files in data/audio by filename matching (naive)
        if os.path.isdir(AUDIO_DIR):
            files = set(os.listdir(AUDIO_DIR))
            df["has_audio"] = df.index.astype(str).apply(lambda idx: any(fname.startswith(str(idx)) for fname in files))
        else:
            df["has_audio"] = False

# Modality counts
    counts = Counter()
    for _, r in df.iterrows():
        if r["has_text"] and r["has_audio"]:
            counts["text+audio"] += 1
        elif r["has_text"]:
            counts["text_only"] += 1
        elif r["has_audio"]:
            counts["audio_only"] += 1
        else:
            counts["neither"] += 1

    print("Modality counts:", counts)
    # bar plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title("Modality availability counts")
    save_fig_pdf(fig, "modality_availability_counts.pdf")

    # cross-modal correlation if durations available
    durations_map = compute_durations_map()
    if durations_map:
        # attempt to match each row's audio filename (if present) and compute duration
        durations = []
        token_lens = []
        for idx, r in df.iterrows():
            token_len = len(r[text_col].split())
            token_lens.append(token_len)
            audio_fname = None
            # prefer explicit audio column
            if audio_col:
                val = r.get(audio_col, "")
                if isinstance(val, str) and val.strip():
                    audio_fname = os.path.basename(val)
            if (not audio_fname) and os.path.isdir(AUDIO_DIR):
                # search for any file that contains the idx in name
                for fname in durations_map.keys():
                    if str(idx) in fname:
                        audio_fname = fname
                        break
            if audio_fname and audio_fname in durations_map:
                durations.append(durations_map[audio_fname])
            else:
                durations.append(np.nan)
        # scatter token_len vs duration
        import numpy as np
        token_lens = np.array(token_lens)
        durations = np.array(durations)
        mask = ~np.isnan(durations)
        if mask.sum() > 5:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(token_lens[mask], durations[mask], alpha=0.6)
            ax.set_xlabel("Token length")
            ax.set_ylabel("Audio duration (s)")
            ax.set_title("Token length vs audio duration")
            save_fig_pdf(fig, "tokenlen_vs_duration.pdf")
            # print correlation
            corr = np.corrcoef(token_lens[mask], durations[mask])[0,1]
            print("Correlation (token length vs duration):", corr)
        else:
            print("Not enough matched audio to compute cross-modal correlation.")
    else:
        print("No per-file durations found in data/audio/. Skipping cross-modal correlation.")

    print("Multimodal EDA complete. Plots saved in", PLOTS_DIR)

# Run main
if __name__ == "__main__":
    main()
