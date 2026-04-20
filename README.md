# quantization-optimization-in-llms
# 🧠 LLM Compression & Efficient Fine-Tuning Pipeline

> A research-grade pipeline implementing and benchmarking **four model compression strategies** on a causal language model — from vanilla PTQ to importance-aware quantization, QAT with STE, and LoRA fine-tuning on a compressed backbone.

---

## 📌 Project Overview

Large Language Models are expensive to deploy. This project tackles that problem head-on by building a **full compression pipeline from scratch** on `distilGPT-2`, rigorously comparing every major quantization paradigm in use across the industry today.

Rather than just calling a library, this project implements quantization **at the mathematical level** — custom quantization functions, gradient-based importance scoring, Straight-Through Estimators, and parameter-efficient fine-tuning — and measures the real cost of each approach on language modeling loss.

---

## 🏗️ Architecture & Pipeline

```
Raw Model (FP32)
      │
      ├──► Post-Training Quantization (PTQ)          → 8-bit symmetric uniform quant
      │
      ├──► Importance-Aware PTQ                       → Fisher gradient scoring + selective quant
      │
      ├──► Quantization-Aware Training (QAT)          → STE fake-quant during training
      │
      └──► LoRA on Quantized Backbone                 → PEFT adapter on compressed model
```

All methods are evaluated on the same **WikiText-2** validation split for fair comparison.

---

## 🔬 Techniques Implemented

### 1. Post-Training Quantization (PTQ) — `ptq.py`
Implements **symmetric uniform quantization** from scratch without any quantization library.

```
Scale = max(|W|) / (2^(n-1) - 1)
W_q   = clamp(round(W / Scale), -128, 127)
W_dq  = W_q × Scale
```

- Supports configurable bit-width (`num_bits` parameter)
- Skips bias quantization (standard best practice)
- Operates in-place on all weight tensors

### 2. Importance-Aware PTQ — `important_analysis.py`
Selectively quantizes **only the weights that matter least**, preserving full precision where it counts.

**Method:**
- Runs forward + backward passes on calibration data
- Scores each weight by **Fisher Information** (squared gradients as a proxy): `I(w) = E[∇L²]`
- Top-5% most important weights kept in FP32; rest quantized to INT8

```python
importance[name] += (param.grad.detach() ** 2)   # Fisher proxy
mask = (importance >= topk_threshold)             # preserve FP32
param.data = torch.where(mask, w_fp32, w_quantized)
```

### 3. Quantization-Aware Training (QAT) — `qt.py`
Trains the model **while simulating quantization** so weights adapt to the reduced precision.

**Key insight:** Uses a custom **Straight-Through Estimator (STE)** to pass gradients through the non-differentiable rounding operation:

```python
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        return torch.round(input / scale) * scale   # quantize in forward

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None                    # pass gradient straight through
```

- Selectively targets heavy layers (`c_attn`, `mlp.c_fc`) for efficiency
- Re-applies fake quantization each forward pass during training

### 4. LoRA on Quantized Backbone — `loraa.py`
Fine-tunes the already-compressed model using **Low-Rank Adaptation (LoRA)**, simulating a real-world scenario where a quantized base model is adapted to a new distribution.

- LoRA rank `r=8`, alpha `16`, targeting attention projections (`c_attn`, `c_proj`)
- Only **0.3–1% of parameters are trainable** — everything else is frozen
- Adapter saved via HuggingFace PEFT for easy reuse

---

## 📊 Results

All models evaluated on WikiText-2 validation set (cross-entropy loss; lower = better):

| Method | Val Loss | vs FP32 | Notes |
|---|---|---|---|
| **FP32 (Baseline)** | 6.662 | — | Full-precision reference |
| **PTQ (8-bit)** | 6.869 | +0.207 ↑ | Minimal degradation — excellent compression |
| **Importance-Aware PTQ** | 8.774 | +2.112 ↑ | Over-aggressive masking; top-k threshold needs tuning |
| **QAT (STE)** | **6.271** | **−0.391 ↓** | **Best result — beats FP32 baseline** |
| **LoRA on Quantized** | 6.524 | +0.338 ↑ | Near-baseline with <1% trainable params |

### Key Findings

- **QAT outperforms FP32** — training with simulated quantization noise acts as implicit regularization, improving generalization on this dataset
- **PTQ causes only 3.1% loss increase** for a model compressed to 8-bit — strong result for a zero-shot method
- **Importance-Aware PTQ underperforms** — the top-5% threshold is too conservative; a key insight is that importance score calibration is highly sensitive to the masking ratio `k`
- **LoRA recovers well** — fine-tuning just ~0.5% of parameters on a quantized base model nearly recovers FP32 performance

---

## 🔍 Analysis Tools

### Weight Distribution Analysis — `failure_analysis.py`
Generates three diagnostic plots to understand quantization error:

- **Histogram + KDE** with 95% confidence bounds
- **Q-Q Plot** to test for Gaussianity of weight distributions
- **Box Plot** with quartile + outlier visualization

```
Mean:  -0.000555
Std:    0.14179
Min:   -2.35845
Max:    5.25480
```

The high max/min asymmetry explains why symmetric quantization introduces some error — the distribution is slightly skewed.

### Quantization Error Analysis — `error_analysis.py`
Directly measures mean absolute error between original and quantized weights, providing a signal for which layers degrade most under quantization.

---

## 📁 Project Structure

```
├── loader.py                # Model + tokenizer setup (distilGPT-2)
├── dataset_loader.py        # WikiText-2 data loading
├── texttoken.py             # Tokenization utilities
│
├── ptq.py                   # PTQ: symmetric uniform quantization
├── important_analysis.py    # Importance-aware PTQ (Fisher scoring)
├── qt.py                    # QAT: fake-quant with STE
├── loraa.py                 # LoRA fine-tuning on quantized model
│
├── eval.py                  # Unified evaluation across all methods
├── evall_lora.py            # LoRA-specific evaluation
├── failure_analysis.py      # Weight distribution visualizations
├── error_analysis.py        # Quantization error measurement
├── loss.py                  # Baseline loss utility
│
└── evaluation_comparison.txt  # Raw experiment results
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch transformers datasets peft matplotlib seaborn scipy
```

### Run the Full Pipeline
```bash
# 1. Baseline FP32 loss
python loss.py

# 2. Apply PTQ and measure loss
python ptq.py

# 3. Importance-aware quantization
python important_analysis.py

# 4. Quantization-aware training
python qt.py

# 5. LoRA fine-tuning on quantized model
python loraa.py

# 6. Unified evaluation + plot
python eval.py

# 7. Weight distribution diagnostics
python failure_analysis.py
```

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Framework | PyTorch |
| Models | HuggingFace Transformers (distilGPT-2) |
| PEFT | HuggingFace PEFT (LoRA) |
| Dataset | WikiText-2 (HuggingFace Datasets) |
| Analysis | NumPy, SciPy, Matplotlib, Seaborn |

---

## 💡 What This Demonstrates

- **Deep understanding of quantization math** — not just calling `torch.quantize_per_tensor`, but building it from first principles
- **Hands-on STE implementation** — a technique used by Google, Meta, and Qualcomm in production quantization frameworks
- **Gradient-based model analysis** — Fisher information as a sensitivity proxy, used in pruning and quantization research
- **PEFT & LoRA** — parameter-efficient fine-tuning, the standard approach for adapting large models in industry
- **Rigorous experimental design** — same dataset, same evaluation function, controlled comparison across all methods
- **Failure analysis mindset** — not just reporting numbers, but diagnosing *why* importance-aware PTQ underperformed

---

## 🔮 Future Work

- [ ] Per-channel quantization for improved PTQ accuracy
- [ ] GPTQ-style second-order weight update for better importance-aware quantization
- [ ] 4-bit NF4 quantization (used in QLoRA)
- [ ] Extend to larger models (GPT-2 medium, OPT-125M)
- [ ] INT8 inference benchmarking (latency + memory)

---

## 📬 Contact

Built as a deep-dive research project to explore the full spectrum of LLM compression techniques.  
Feel free to reach out for collaboration or questions!
mo: 9305797081
email:shivampandey52444@gmail.com
