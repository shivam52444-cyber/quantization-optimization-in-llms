import torch
import copy
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from loader import model, tokenizer, device
from ptq import quantize_tensor


# ================================
# Dataset
# ================================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch])
    }


# ================================
# Load Data
# ================================
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

texts = dataset["train"]["text"][:1000]
texts = [t for t in texts if len(t.strip()) > 0]

dataset = TextDataset(texts, tokenizer)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)


# ================================
# Importance Computation
# ================================
def compute_importance(model, dataloader, device, num_batches=5):
    model.train()

    importance = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            importance[name] = torch.zeros_like(param.data)

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] += (param.grad.detach() ** 2)

    return importance


# ================================
# Top-K Mask
# ================================
def get_topk_mask(importance, k=0.05):
    masks = {}

    for name, imp in importance.items():
        flat = imp.view(-1)

        num_keep = int(k * flat.numel())
        threshold = torch.topk(flat, num_keep).values[-1]

        masks[name] = (imp >= threshold)

    return masks


# ================================
# Importance-Aware Quantization
# ================================
def importance_aware_quantization(model, masks, num_bits=8):

    for name, param in model.named_parameters():

        if name not in masks:
            continue

        mask = masks[name]

        with torch.no_grad():
            w = param.data
            w_q = quantize_tensor(w, num_bits)

            # keep important weights FP, rest quantized
            param.data = torch.where(mask, w, w_q)

    return model


# ================================
# RUN PIPELINE
# ================================
if __name__ == "__main__":

    import copy

    base_model = copy.deepcopy(model)

    print("Computing importance...")
    importance = compute_importance(base_model, dataloader, device)

    print("Creating mask...")
    masks = get_topk_mask(importance, k=0.05)

    print("Applying importance-aware PTQ...")
    model_ia = copy.deepcopy(base_model)
    model_ia = importance_aware_quantization(model_ia, masks)

    # evaluation (single batch quick check)
    batch = next(iter(dataloader))
    inputs = {k: v.to(device) for k, v in batch.items()}

    model_ia.eval()
    with torch.no_grad():
        loss = model_ia(**inputs, labels=inputs["input_ids"]).loss

    print("Importance-aware PTQ Loss:", loss.item())