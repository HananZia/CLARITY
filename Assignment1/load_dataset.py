from datasets import load_dataset

dataset = load_dataset("ailsntua/QEvasion", cache_dir="dataset/")

print(dataset)