# CLARITY · Text-Based Political Interview Analysis

CLARITY is a compact exploratory data analysis (EDA) toolkit for political interview transcripts. It spotlights how clearly a respondent answers and whether they evade questions, pairing descriptive statistics with ready-to-publish visualizations.

## Highlights
- **Clarity vs. Evasion framing** for every answer in `interview_answer`.
- **Token, sentence, and n-gram diagnostics** to reveal lexical habits.
- **Optional sentiment and language profiling** when HuggingFace Transformers are available.
- **Multimodal hooks** (`eda_audio.py`, `eda_multimodal.py`) that gracefully skip when assets are missing.

## Repository Map
- `Assignment1/src/eda_text.py` – primary text EDA pipeline.
- `Assignment1/src/eda_audio.py` – basic waveform checks (auto-disabled without audio files).
- `Assignment1/src/eda_multimodal.py` – joins textual and acoustic cues when possible.
- `Assignment1/plots/` – PDF charts exported by every module.
- `Assignment1/notebooks/eda.ipynb` – rapid experimentation playground.

## Getting Started
1. **Clone and enter the project**
	```bash
	git clone https://github.com/<your-username>/CLARITY.git
	cd CLARITY
	```
2. **Create a virtual environment**
	```bash
	python -m venv venv
	# Windows
	venv\Scripts\activate
	# macOS / Linux
	source venv/bin/activate
	```
3. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```
4. **Fetch required NLTK corpora (one time)**
	```python
	import nltk
	nltk.download("punkt")
	nltk.download("punkt_tab")
	nltk.download("stopwords")
	```

## Running the Analyses
- **Text EDA (default workflow)**
  ```bash
  python Assignment1/src/eda_text.py
  ```
  - Input: `Assignment1/dataset/train.csv` (optionally `test.csv`).
  - Focus column: `interview_answer`.
  - Output: PDFs saved under `Assignment1/plots/` (token stats, n-grams, label balance, sentiment).
  - If Transformers are installed and a checkpoint is reachable, sentiment overlays automatically render.

- **Audio / Multimodal hooks**
  Run `eda_audio.py` or `eda_multimodal.py` to extend the analysis. Each script safely no-ops when required files are absent, so you can keep them in automated pipelines without extra guards.

- **Notebook exploration**
  Launch `Assignment1/notebooks/eda.ipynb` for ad-hoc slicing, plotting, or prompt engineering experiments.

## Outputs at a Glance
- Vocabulary coverage, token-length histograms, and sentence distribution plots.
- Top unigram and bigram charts with optional stopword filtering.
- Clarity/evasion label balance visualizations plus sentiment overlays when enabled.
- Language detection snapshots to flag non-English responses before modeling.

## Notes
- Plots are exported as PDFs to remain automation-friendly and render consistently on headless servers.
- Sentence-level fallbacks keep the pipeline running even if `punkt` is unavailable at runtime.
- Audio analysis is automatically skipped when waveform data is missing, preventing brittle jobs.

## License

This project is available under the MIT License. See `LICENSE.txt` for details.
