<!-- CLARITY — Text-Based Political Interview Analysis

This project performs text-based exploratory data analysis (EDA) on political interview responses.
It focuses on clarity and evasion labels and generates visualizations for deeper insights.

Project Structure
CLARITY/
│
├── Assignment1/
│   ├── src/
│   │   ├── eda_text.py        # Main script for text EDA
│   │   ├── eda_audio.py       # Optional audio EDA (skipped if no audio)
│   │   ├── utils.py           # Helper functions (plot saving, paths, etc.)
│   │   └── save_dataset.py    # Dataset download/preprocessing
│   │
│   ├── dataset/               # CSV dataset files
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── plots/                 # Generated plots saved as PDFs
│   └── README.md
│
├── venv/                      # Python virtual environment (ignored in git)
└── LICENSE                    # MIT License

Installation

Clone the repository

git clone https://github.com/<your-username>/CLARITY.git
cd CLARITY


Create and activate a virtual environment

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


Download NLTK resources

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

Usage

Run the main text EDA script:

python Assignment1/src/eda_text.py


Input dataset: Assignment1/dataset/train.csv (and optionally test.csv)

Text column: interview_answer

Output: Plots saved to Assignment1/plots/*.pdf

Optional sentiment analysis will run if Transformers are installed and a model is available.

Features

Token-level analysis: token length statistics, histograms, vocabulary size

N-grams: top unigrams and bigrams visualized

Sentence-level statistics: sentence counts and lengths (with fallback if NLTK missing)

Language detection: identifies languages in a sample of responses

Sentiment analysis: optional using HuggingFace Transformers

Label distributions: clarity and evasion labels

Notes

Audio EDA (eda_audio.py) is skipped if no audio files exist.

Plots are saved as PDFs to avoid GUI issues on headless systems.

Safe fallbacks included for missing sentence tokenizers.

License

This project is licensed under the MIT License — see LICENSE for details. -->