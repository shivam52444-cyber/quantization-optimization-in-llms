import torch
import copy
import os

from peft import LoraConfig, get_peft_model
from loader import model, tokenizer, device
from ptq import quantized_model   # ✅ make sure this exists
from important_analysis import TextDataset, collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader


# ================================
# Prepare Model
# ================================
lora_model = copy.deepcopy(quantized_model)
lora_model.to(device)


# ================================
# LoRA Config
# ================================
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

lora_model = get_peft_model(lora_model, config)
lora_model.print_trainable_parameters()


# ================================
# Dataset
# ================================
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

new_texts = dataset["train"]["text"][2000:2500]
new_texts = [t for t in new_texts if len(t.strip()) > 0]

new_dataset = TextDataset(new_texts, tokenizer)

new_loader = DataLoader(
    new_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


# ================================
# Train LoRA
# ================================
def train_lora(model, dataloader, device, max_batches=20):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i, batch in enumerate(dataloader):

        if i >= max_batches:
            break

        print(f"LoRA Batch {i}")

        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("LoRA training done")


train_lora(lora_model, new_loader, device)


# ================================
# Save Model
# ================================
os.makedirs("saved_models/lora_model", exist_ok=True)

lora_model.save_pretrained("saved_models/lora_model")
tokenizer.save_pretrained("saved_models/lora_model")

print("✅ LoRA model saved")