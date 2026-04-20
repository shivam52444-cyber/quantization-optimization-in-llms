import torch
from qt import *
from ptq import *
from loader import *

def evaluate(model, dataloader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            losses.append(loss.item())

    return sum(losses) / len(losses)

fp32_loss = evaluate(fp_32, val_loader, device)
ptq_loss = evaluate(quantized_model, val_loader, device)
importance_loss = evaluate(model, val_loader, device)
qat_loss = evaluate(qat_model, val_loader, device)

print("FP32:", fp32_loss)
print("PTQ:", ptq_loss)
print("Importance PTQ:", importance_loss)
print("QAT:", qat_loss)

import matplotlib.pyplot as plt

methods = [ "fp_32", "PTQ", "Imp PTQ", "QAT"]
losses = [ fp32_loss, ptq_loss, importance_loss, qat_loss]

plt.figure()
plt.plot(methods, losses, marker='o')
plt.title("Quantization Comparison")
plt.xlabel("Method")
plt.ylabel("Loss")
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import os

methods = ["FP32", "PTQ", "Imp PTQ", "QAT"]
losses = [fp32_loss, ptq_loss, importance_loss, qat_loss]

# safety check
assert len(methods) == len(losses)

# create folder if not exists
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(6,4))

plt.plot(methods, losses, marker='o')
plt.title("Quantization Comparison")
plt.xlabel("Method")
plt.ylabel("Loss")
plt.grid()

# 🔥 SAVE FIGURE
plt.savefig("results/quantization_comparison.png", dpi=300, bbox_inches='tight')

plt.show()

print("✅ Plot saved at: results/quantization_comparison.png")