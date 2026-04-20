import torch
from dataset_loader import *
from loader import *
from texttoken import *

def quantize_tensor(tensor, num_bits=8):
    """
    Uniform symmetric quantization
    """

    # Step 1: find max absolute value
    max_val = tensor.abs().max()

    # Step 2: compute quantization range
    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    # Step 3: compute scale
    scale = max_val / qmax if max_val != 0 else 1.0

    # Step 4: quantize
    q_tensor = torch.round(tensor / scale)

    # Step 5: clamp (important to avoid overflow)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)

    # Step 6: dequantize
    dq_tensor = q_tensor * scale

    return dq_tensor


def apply_ptq(model, num_bits=8):
    """
    Apply PTQ to all model weights
    """

    quantized_model = model

    for name, param in quantized_model.named_parameters():

        # skip bias (optional but common)
        if "bias" in name:
            continue

        # only quantize weights
        with torch.no_grad():
            param.data = quantize_tensor(param.data, num_bits)

    return quantized_model

def apply_ptq(model, num_bits=8):
    """
    Apply PTQ to all model weights
    """

    quantized_model = model

    for name, param in quantized_model.named_parameters():

        # skip bias (optional but common)
        if "bias" in name:
            continue

        # only quantize weights
        with torch.no_grad():
            param.data = quantize_tensor(param.data, num_bits)

    return quantized_model

quantized_model = apply_ptq(model, num_bits=8)

outputs = quantized_model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss

print("PTQ Loss:", loss.item())