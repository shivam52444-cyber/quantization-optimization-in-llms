from dataset_loader import *
from loader import *

def tokenize_function(texts):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

inputs = tokenize_function(texts[:32])  # one batch
inputs = {k: v.to(device) for k, v in inputs.items()}