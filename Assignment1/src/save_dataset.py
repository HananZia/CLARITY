from datasets import load_dataset

dataset = load_dataset("ailsntua/QEvasion")

# Convert splits to CSV
dataset["train"].to_pandas().to_csv("dataset/train.csv", index=False)
dataset["test"].to_pandas().to_csv("dataset/test.csv", index=False)

print("Saved train.csv and test.csv into /dataset")