weights = {}
from ptq import *
for name, param in model.named_parameters():
    weights[name] = param.data
    
weights["transformer.h.0.mlp.c_fc.weight"]

import numpy as np

w_np = weights["transformer.h.0.mlp.c_fc.weight"].cpu().numpy()

w = weights["transformer.h.0.mlp.c_fc.weight"]

print("Mean:", w.mean().item())
print("Std:", w.std().item())
print("Min:", w.min().item())
print("Max:", w.max().item())

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# convert tensor → numpy
w_np = w.cpu().numpy().flatten()

# -------------------------------
# 1. Histogram + KDE
# -------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# convert tensor → numpy
w_np = w.cpu().numpy().flatten()

# compute 95% quantile bounds
q_low, q_high = np.percentile(w_np, [2.5, 97.5])

# -------------------------------
# 1. Histogram + KDE + 95% bounds
# -------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(w_np, bins=100, kde=True)

# mark 95% region
plt.axvline(q_low, linestyle='--', label='2.5% quantile')
plt.axvline(q_high, linestyle='--', label='97.5% quantile')

plt.title("Weight Distribution with 95% Range")
plt.xlabel("Weight values")
plt.ylabel("Frequency")
plt.legend()

plt.savefig("weight_distribution_95.png", dpi=300)
plt.close()


# -------------------------------
# 2. Q-Q Plot
# -------------------------------
plt.figure(figsize=(6, 4))
stats.probplot(w_np, dist="norm", plot=plt)
plt.title("Q-Q Plot (Normal Distribution Check)")

plt.savefig("qq_plot.png", dpi=300)
plt.close()


# -------------------------------
# 3. Box Plot (Quartile + Outliers)
# -------------------------------
plt.figure(figsize=(6, 2))
sns.boxplot(x=w_np)

# highlight 95% region
plt.axvline(q_low, linestyle='--', label='2.5%')
plt.axvline(q_high, linestyle='--', label='97.5%')

plt.title("Box Plot with 95% Range")
plt.legend()

plt.savefig("boxplot_95.png", dpi=300)
plt.close()