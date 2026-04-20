from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# take small subset (important for your laptop)
texts = dataset["train"]["text"][:1000]

# remove empty lines
texts = [t for t in texts if len(t.strip()) > 0]