import torch

from transformers import AutoModelForCausalLM
from peft import PeftModel

from loader import tokenizer, device
from eval import evaluate   # your evaluate function
from important_analysis import TextDataset, collate_fn

from datasets import load_dataset
from torch.utils.data import DataLoader


# ================================
# Load Base Model
# ================================
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# ================================
# Load LoRA Adapter
# ================================
lora_model = PeftModel.from_pretrained(
    base_model,
    "saved_models/lora_model"
)

lora_model.to(device)


# ================================
# Load SAME validation dataset
# ================================
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

val_texts = dataset["validation"]["text"][:500]
val_texts = [t for t in val_texts if len(t.strip()) > 0]

val_dataset = TextDataset(val_texts, tokenizer)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)


# ================================
# Evaluate
# ================================
lora_loss = evaluate(lora_model, val_loader, device)

print("✅ LoRA Loss:", lora_loss)