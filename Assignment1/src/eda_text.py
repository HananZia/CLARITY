# src/eda_text.py
"""
Text EDA:
- load data CSV (train.csv expected in ../data)
- token length stats, histogram
- frequent n-grams (unigrams + bigrams)
- vocabulary size
- language detection (langdetect)
- sentiment distribution (transformers pipeline, optional)
- saves plots to ../plots as PDFs
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect
from transformers import pipeline
from tqdm import tqdm

from utils import PLOTS_DIR, save_fig_pdf

sns.set(style="whitegrid")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

def load_data():
    if os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV)
    else:
        raise FileNotFoundError(f"{TRAIN_CSV} not found. Run save_dataset.py or put train.csv in dataset/")
    return df

def token_stats(texts):
    token_lens = [len(str(t).split()) for t in texts]
    return np.array(token_lens)

def plot_token_length_hist(token_lens):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(token_lens, bins=50)
    ax.set_xlabel("Token length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Token length distribution (interview_answer)")
    save_fig_pdf(fig, "text_token_length_hist.pdf")

def top_ngrams(texts, ngram_range=(1,1), topk=30):
    vec = CountVectorizer(ngram_range=ngram_range, max_features=5000, stop_words='english')
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    top_idx = np.argsort(sums)[::-1][:topk]
    return [(terms[i], int(sums[i])) for i in top_idx]

def plot_top_ngrams(ngrams, filename="top_unigrams.pdf", title="Top tokens"):
    terms, counts = zip(*ngrams)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(range(len(terms))[::-1], counts)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1])
    ax.set_xlabel("Count")
    ax.set_title(title)
    save_fig_pdf(fig, filename)

def detect_languages(texts, sample_limit=2000):
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    langs = []
    for t in list(texts)[:sample_limit]:
        try:
            langs.append(detect(str(t)))
        except:
            langs.append("error")
    return Counter(langs)

def sentiment_analysis(texts, model_name="distilbert-base-uncased-finetuned-sst-2-english", sample_limit=1000):
    # Use transformers pipeline if model assets are available; this is optional and may download models.
    try:
        sentiment = pipeline("sentiment-analysis", model=model_name)
    except Exception as e:
        print("Sentiment pipeline unavailable:", e)
        return None
    results = []
    for t in texts[:sample_limit]:
        try:
            res = sentiment(str(t)[:512])[0]  # truncate to 512
            results.append(res)
        except Exception:
            results.append({"label":"ERROR","score":0.0})
    return results

def plot_label_distribution(df, label_col="evasion_label", filename="label_distribution.pdf", title="Evasion label distribution"):
    counts = df[label_col].value_counts(dropna=False).sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel(label_col)
    ax.set_ylabel("Count")
    ax.set_title(title)
    save_fig_pdf(fig, filename)

def main():
    print("Loading data...")
    df = load_data()
    # Determine which text column to use: prefer 'interview_answer'
    text_col = "interview_answer" if "interview_answer" in df.columns else "text"
    print("Using text column:", text_col)

    # Basic summary
    print("Num rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Sample:")
    print(df[[text_col]].head(3).to_string())

    texts = df[text_col].fillna("").astype(str).tolist()

    # Token length
    token_lens = token_stats(texts)
    print("Token length: mean", token_lens.mean(), "median", np.median(token_lens))
    plot_token_length_hist(token_lens)

    # Label distribution
    if "evasion_label" in df.columns:
        plot_label_distribution(df, label_col="evasion_label", filename="label_distribution_evasion.pdf", title="Evasion labels")
    if "clarity_label" in df.columns:
        plot_label_distribution(df, label_col="clarity_label", filename="label_distribution_clarity.pdf", title="Clarity labels")

    # Top unigrams and bigrams
    print("Computing top unigrams...")
    uni = top_ngrams(texts, ngram_range=(1,1), topk=30)
    plot_top_ngrams(uni, filename="top_unigrams.pdf", title="Top unigrams")
    print("Computing top bigrams...")
    bi = top_ngrams(texts, ngram_range=(2,2), topk=30)
    plot_top_ngrams(bi, filename="top_bigrams.pdf", title="Top bigrams")

    # Vocabulary size
    vec = CountVectorizer(stop_words='english')
    vec.fit(texts)
    vocab_size = len(vec.vocabulary_)
    print("Vocabulary size (after stopwords removal):", vocab_size)

    # Language detection (sample)
    print("Detecting languages (sample)...")
    lang_counts = detect_languages(texts, sample_limit=2000)
    print("Language counts (sample):", lang_counts)
    # Save simple bar plot for languages
    try:
        labels, vals = zip(*lang_counts.items())
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(labels, vals)
        ax.set_title("Language detection (sample)")
        save_fig_pdf(fig, "language_detection_sample.pdf")
    except Exception:
        pass

    # Sentiment (optional)
    print("Running optional sentiment analysis on sample (this will download a model if not present)...")
    try:
        sent_res = sentiment_analysis(texts, sample_limit=500)
        if sent_res:
            labels = [r["label"] for r in sent_res]
            from collections import Counter
            c = Counter(labels)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(list(c.keys()), list(c.values()))
            ax.set_title("Sentiment distribution (sample)")
            save_fig_pdf(fig, "sentiment_distribution_sample.pdf")
    except Exception as e:
        print("Skipping sentiment (error):", e)

    print("Text EDA complete. Plots in", PLOTS_DIR)

if __name__ == "__main__":
    main()
