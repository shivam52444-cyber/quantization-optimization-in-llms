from failure_analysis import *
w = weights["transformer.h.0.mlp.c_fc.weight"]

w_q = quantize_tensor(w)

error = (w - w_q).abs()

print("Mean Error:", error.mean().item())