CLARITY — Text-Based Political Interview Analysis

This project performs text-based exploratory data analysis (EDA) on political interview responses. It focuses on clarity and evasion labels and generates plots and statistics for further analysis.

📁 Project Structure
CLARITY/
│
├── Assignment1/
│   ├── src/
│   │   ├── eda_text.py        # Main text EDA script
│   │   ├── eda_audio.py       # (Optional) Audio EDA; skipped if no audio
│   │   ├── utils.py           # Helper functions (plot saving, paths, etc.)
│   │   └── save_dataset.py    # Optional dataset download/preprocessing
│   │
│   ├── dataset/               # Place CSV files here
│   │   ├── train.csv          # Training data
│   │   └── test.csv           # Test data
│   │
│   ├── plots/                 # Generated plots saved as PDFs
│   └── README.md              # Project structure and usage
│
├── venv/                      # Python virtual environment
└── LICENSE                    # MIT License

📌 Key Notes

src/: Contains all Python scripts for EDA, dataset processing, and utility functions.

dataset/: Place your CSV dataset files here (train.csv and test.csv).

plots/: All visualizations (token distributions, n-grams, sentiment, labels) will be saved as PDFs here.

venv/: Python virtual environment (not required to be checked in).

▶️ Running the Project in VS Code

Open the project folder in VS Code.

Activate the virtual environment in the integrated terminal:

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install required packages:

pip install -r requirements.txt


Download NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


Run the main text EDA script:

python Assignment1/src/eda_text.py


View the generated plots in Assignment1/plots/.

📝 Notes for VS Code Users

You can run scripts directly in the terminal or use the Run Python File option.

Plots are saved as PDFs to avoid GUI backend issues.

Optional sentiment analysis will download models if Transformers are installed.

Audio analysis (eda_audio.py) is skipped if no audio files exist.
