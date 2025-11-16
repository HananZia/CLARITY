"""
Advanced Text EDA:
- load data CSV (train.csv)
- token stats, char stats, sentence stats
- histogram plots
- vocabulary size
- frequent n-grams (uni/bigrams)
- readability metrics
- TF-IDF similarity distribution
- keyword extraction (RAKE)
- wordcloud generation
- language detection (sampled)
- sentiment distribution (optional)
- saves plots as PDFs
"""

# Imports
import os
import matplotlib
matplotlib.use("Agg")  # headless mode
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # for plotting
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # for n-grams and TF-IDF
from sklearn.metrics.pairwise import cosine_similarity # for similarity
from langdetect import detect
from tqdm import tqdm
# NLTK setup
import nltk
try:
    from nltk import sent_tokenize
except:
    def sent_tokenize(x): # fallback simple sentence tokenizer
        return x.split(".")

# NLTK stopwords
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, flesch_kincaid_grade # readability
from wordcloud import WordCloud
from rake_nltk import Rake
from utils import PLOTS_DIR, save_fig_pdf # custom utils

sns.set(style="whitegrid")

DATASET_DIR = Path(__file__).parents[2] / "dataset"
TRAIN_CSV = DATASET_DIR / "train.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Ensure NLTK deps
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Data Loading
def load_data():
    if TRAIN_CSV.exists():
        return pd.read_csv(TRAIN_CSV)
    raise FileNotFoundError(f"{TRAIN_CSV} not found. Put train.csv in dataset/.")

# Compute basic text stats
def compute_text_stats(texts):
    stats = {
        "token_len": [],
        "char_len": [],
        "sentence_len": []
    }

    for t in texts:
        t = str(t)
        tokens = t.split()
        stats["token_len"].append(len(tokens))
        stats["char_len"].append(len(t))
        stats["sentence_len"].append(len(sent_tokenize(t)))

    return stats

# Plotting histograms
def plot_hist(data, xlabel, filename, title=None, bins=50):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title or xlabel)
    save_fig_pdf(fig, filename)

# N-grams
def top_ngrams(texts, ngram_range=(1, 1), topk=30):
    vec = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        max_features=8000,
        min_df=2
    )
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(counts)[::-1][:topk]
    return [(vocab[i], int(counts[i])) for i in idx]

# Plot horizontal bar chart
def plot_barh(pairs, filename, title):
    terms, counts = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(terms))[::-1], counts)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1])
    ax.set_title(title)
    save_fig_pdf(fig, filename)


# Language Detection
def detect_languages(texts, sample=1000):
    langs = []
    for t in texts[:sample]:
        try:
            langs.append(detect(str(t)))
        except:
            langs.append("error")
    return Counter(langs)


# Keyword Extraction (RAKE)
def extract_keywords(texts, topk=20):
    rake = Rake(stopwords.words("english"))
    all_keywords = Counter()

    for t in texts[:2000]:  # limit for speed
        rake.extract_keywords_from_text(t)
        extracted = rake.get_ranked_phrases()[0:5]
        all_keywords.update(extracted)

    return all_keywords.most_common(topk)


# Readability
def compute_readability(texts, sample=1000):
    scores = []
    grades = []
    for t in texts[:sample]:
        t = str(t)
        scores.append(flesch_reading_ease(t))
        grades.append(flesch_kincaid_grade(t))
    return np.array(scores), np.array(grades)


# TF-IDF Similarity Distribution
def similarity_distribution(texts, sample=1500):
    sample_texts = texts[:sample]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sample_texts)
    sim = cosine_similarity(X)
    tril = sim[np.tril_indices_from(sim, k=-1)]
    return tril


# Wordcloud
def generate_wordcloud(texts, filename):
    wc = WordCloud(
        width=1000,
        height=600,
        stopwords=set(stopwords.words("english")),
        background_color="white"
    ).generate(" ".join(texts[:5000]))

    fig = plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    save_fig_pdf(fig, filename)


# Main Function
def main():
    print("Loading dataset…")
    df = load_data()
    text_col = "interview_answer" if "interview_answer" in df.columns else "text"
    print("Using text column:", text_col)

    texts = df[text_col].fillna("").astype(str).tolist()

    # Basic stats
    stats = compute_text_stats(texts)
    print(f"Token length mean={np.mean(stats['token_len']):.2f}")
    print(f"Char length mean={np.mean(stats['char_len']):.2f}")
    print(f"Sentence count mean={np.mean(stats['sentence_len']):.2f}")

    plot_hist(stats["token_len"], "Token Count", "token_length.pdf", "Token Length Distribution")
    plot_hist(stats["char_len"], "Character Count", "char_length.pdf", "Character Length Distribution")
    plot_hist(stats["sentence_len"], "Sentence Count", "sentence_length.pdf", "Sentence Count Distribution")

    # N-grams
    print("Computing top unigrams…")
    uni = top_ngrams(texts, (1, 1))
    plot_barh(uni, "top_unigrams.pdf", "Top Unigrams")

    print("Computing top bigrams…")
    bi = top_ngrams(texts, (2, 2))
    plot_barh(bi, "top_bigrams.pdf", "Top Bigrams")

    # Vocabulary Size
    vec = CountVectorizer(stop_words="english")
    vec.fit(texts)
    print("Vocabulary size:", len(vec.vocabulary_))


    # Keyword Extraction (RAKE)
    print("Extracting keywords (RAKE)…")
    kw = extract_keywords(texts)
    print("Top keywords:", kw[:10])
    plot_barh(kw, "keywords.pdf", "Top Extracted Keywords")

    # Wordcloud
    print("Generating wordcloud…")
    generate_wordcloud(texts, "wordcloud.pdf")

    # Readability
    print("Computing readability metrics…")
    fre, fk = compute_readability(texts)
    print("Flesch Reading Ease mean:", fre.mean())
    print("Flesch-Kincaid grade mean:", fk.mean())
    plot_hist(fre, "Flesch Reading Ease", "flesch_reading_ease.pdf")
    plot_hist(fk, "Flesch-Kincaid Grade", "flesch_kincaid.pdf")

    # Similarity
    print("Computing TF-IDF similarity distribution…")
    sim = similarity_distribution(texts)
    plot_hist(sim, "TF-IDF pairwise similarity", "tfidf_similarity.pdf")

    # Language Detection
    print("Detecting languages…")
    lang_counts = detect_languages(texts)
    langs, vals = zip(*lang_counts.items())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(langs, vals)
    ax.set_title("Language Detection")
    save_fig_pdf(fig, "language_detection.pdf")

    print("All EDA completed. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
