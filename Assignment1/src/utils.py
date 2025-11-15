
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def save_fig_pdf(fig, filename: str, tight=True):
    out = PLOTS_DIR / filename
    if tight:
        fig.tight_layout()
    fig.savefig(out, format="pdf")
    print(f"Saved plot: {out}")
    plt.close(fig)

def safe_get(df, col, idx, default=None):
    try:
        return df.loc[idx, col]
    except Exception:
        return default
