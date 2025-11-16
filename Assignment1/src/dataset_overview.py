# import necessary libraries
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# Path to dataset
DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"
AUDIO_DIR = DATASET_DIR / "audio"
TRAIN_CSV = DATASET_DIR / "train.csv"
TEST_CSV = DATASET_DIR / "test.csv"

# load data from CSV files
def load_data(csv_path):
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"{csv_path} not found. Please place the file in dataset/")

# generate dataset summary
def dataset_summary(df, name="Dataset"):
    print(f"\n {name} Summary ")

    # Number of samples
    print(f"Number of samples: {len(df)}")

    # Column info
    col_info = []
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        col_info.append([col, dtype, missing, unique])

    print("\nColumns / Features:")
    print(tabulate(col_info, headers=["Column", "Type", "Missing", "Unique"], tablefmt="github"))

    # Token length stats for text columns
    text_cols = [c for c in df.columns if 'question' in c.lower() or 'answer' in c.lower()]
    token_stats = []
    for col in text_cols:
        tokens = df[col].dropna().astype(str).apply(lambda x: len(x.split()))
        token_stats.append([
            col,
            round(tokens.mean(), 2),
            int(tokens.median()),
            int(tokens.min()),
            int(tokens.max())
        ])

# display token stats
    if token_stats:
        print("\nText Columns Token Length Statistics:")
        print(tabulate(token_stats, headers=["Column", "Mean", "Median", "Min", "Max"], tablefmt="github"))

    # Label distributions
    label_cols = [c for c in df.columns if 'label' in c.lower()]
    for col in label_cols:
        dist = df[col].value_counts(dropna=False).reset_index()
        dist.columns = [col + " Value", "Count"]
        print(f"\nLabel Distribution for '{col}':")
        print(tabulate(dist.values, headers=dist.columns, tablefmt="github"))

# main function
def main():
    print("Loading train dataset...")
    train_df = load_data(TRAIN_CSV)
    dataset_summary(train_df, name="Train Set")

    print("\nLoading test dataset...")
    test_df = load_data(TEST_CSV)
    dataset_summary(test_df, name="Test Set")

# run the main function
if __name__ == "__main__":
    main()
