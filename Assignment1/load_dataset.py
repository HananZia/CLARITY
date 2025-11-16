from datasets import load_dataset # Import HF datasets

dataset = load_dataset("ailsntua/QEvasion", cache_dir="dataset/") # save dataset from HF hub

print(dataset)