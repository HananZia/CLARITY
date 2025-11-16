from datasets import load_dataset # Import HF datasets

dataset = load_dataset("ailsntua/QEvasion") # load dataset from HF hub

# Convert splits to CSV
dataset["train"].to_pandas().to_csv("dataset/train.csv", index=False)
dataset["test"].to_pandas().to_csv("dataset/test.csv", index=False)

print("Saved train.csv and test.csv into /dataset")