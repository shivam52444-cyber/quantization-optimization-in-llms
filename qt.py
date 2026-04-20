import torch
import copy
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from loader import model, tokenizer, device

# ================================
# SYSTEM OPTIMIZATION
# ================================
torch.set_num_threads(2)

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
            max_length=64,   # 🔥 reduced
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
print("Loading dataset...")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

train_texts = dataset["train"]["text"][:1000]
val_texts   = dataset["validation"]["text"][:200]

train_texts = [t for t in train_texts if len(t.strip()) > 0]
val_texts   = [t for t in val_texts if len(t.strip()) > 0]

train_loader = DataLoader(TextDataset(train_texts, tokenizer), batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(TextDataset(val_texts, tokenizer), batch_size=4, shuffle=False, collate_fn=collate_fn)


# ================================
# STE Quantization
# ================================
class STEQuantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale):
        q = torch.round(input / scale)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quantize(param, num_bits=8):
    max_val = param.abs().max()
    qmax = 2**(num_bits-1) - 1
    scale = max_val / qmax if max_val != 0 else 1.0

    q = STEQuantize.apply(param, scale)

    return q.detach() + param - param.detach()


# ================================
# FAST QAT (ONLY IMPORTANT LAYERS)
# ================================
def apply_fake_quant(model):

    for name, param in model.named_parameters():

        # 🔥 only quantize heavy layers
        if "attn.c_attn.weight" in name or "mlp.c_fc.weight" in name:
            param.data = fake_quantize(param.data)

    return model


# ================================
# TRAIN QAT (FAST)
# ================================
def train_qat(model, dataloader, device, epochs=1):

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch} started")

        for i, batch in enumerate(dataloader):

            print(f"Batch {i}")  # 🔥 debug progress

            inputs = {k: v.to(device) for k, v in batch.items()}

            # apply quantization (lightweight)
            apply_fake_quant(model)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")


# ================================
# EVALUATION
# ================================
def evaluate(model, dataloader, device):

    model.eval()
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            losses.append(loss.item())

    return sum(losses) / len(losses)

qat_model = copy.deepcopy(model)


# ================================
# MAIN PIPELINE
# ================================
if __name__ == "__main__":

    print("\nStarting QAT pipeline...\n")

    qat_model = copy.deepcopy(model)

    # 🔥 TRAIN
    train_qat(qat_model, train_loader, device, epochs=1)

    # 🔥 EVALUATE
    qat_loss = evaluate(qat_model, val_loader, device)

    print("\n✅ QAT Loss:", qat_loss)