# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (ML_python_312)
#     language: python
#     name: ml_python_312
# ---

# %% [markdown]
# # 2 The SVD on Weather Data

# %%
import numpy as np
from numpy.linalg import svd as svd
from numpy.linalg import matrix_rank as matrix_rank
import pandas as pd
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from a03_helper import *

# %%
# The date is being loaded via `a03_helper.py`

# Plot the coordinates
plot_xy(lon, lat)


# %% [markdown]
# ## 2a

# %%
# YOUR PART
# Normalize the data to z-scores. Store the result in X.
# For this, complete the code in `a03_helper.py`.

# %%
# Plot histograms of attributes
nextplot()
X.hist(ax=plt.gca())
plt.show()

# %% [markdown]
# ## 2b

# %%
# Compute the SVD of the normalized climate data and store it in variables U,s,Vt. What
# is the rank of the data?
# YOUR PART
# Thin SVD: U is N x D (2575 x 48), s holds the D singular values, Vt is D x D.
U, s, Vt = svd(X, full_matrices=False)
print("shapes: U", U.shape, " s", s.shape, " Vt", Vt.shape)
print("rank of X:", matrix_rank(X))
np.set_printoptions(precision=3, suppress=True)
print("singular values:\n", s)
print(f"smallest singular value: {s[-1]:.3g},  largest/smallest: {s[0] / s[-1]:.3g}")

# %% [markdown]
# ## 2c

# %%
# Here is an example plot.
plot_xy(lon, lat, U[:, 0])

# %%
# For interpretation, it may also help to look at the other component matrices and
# perhaps use other plot functions (e.g., plot_matrix).
# YOUR PART
# (1) First 5 left singular vectors on the map: color = entry of U[:, k] for each location
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for k in range(5):
    ax = axes.flat[k]
    plot_xy(lon, lat, U[:, k], axis=ax, s=4)
    ax.set_title(f"U[:, {k}]  (sigma = {s[k]:.1f})")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
axes.flat[5].axis("off")
plt.tight_layout()
plt.show()

# (2) Corresponding right singular vectors: which climate variables make up each pattern.
# Each row of Vt has 48 entries = 4 groups (min, max, avg temperature, rain) x 12 months.
months = np.arange(1, 13)
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)
for k in range(5):
    ax = axes[k]
    for g, name in enumerate(["min temp", "max temp", "avg temp", "rain"]):
        ax.plot(months, Vt[k, 12 * g : 12 * (g + 1)], marker=".", label=name)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Vt[{k}, :]")
    ax.set_xlabel("month")
    ax.set_xticks([1, 4, 7, 10])
axes[0].set_ylabel("weight")
axes[0].legend(fontsize="small")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2d

# %%
# Here is an example.
plot_xy(U[:, 0], U[:, 1], lat - np.mean(lat))

# %%
# YOUR PART
# Scatterplots between pairs of left singular vectors.
# Top row: color = North-South position (latitude, centered), bottom row: East-West (longitude).
pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for col, (i, j) in enumerate(pairs):
    for row, (coord_values, coord_name) in enumerate([(lat, "North-South"), (lon, "East-West")]):
        ax = axes[row, col]
        plot_xy(U[:, i], U[:, j], coord_values - np.mean(coord_values), axis=ax, aspect="auto", s=4)
        ax.set_xlabel(f"U[:, {i}]")
        ax.set_ylabel(f"U[:, {j}]")
        ax.set_title(f"U{i} vs U{j}, color: {coord_name}")
plt.tight_layout()
plt.show()

# How strongly is each left singular vector related to the geographic position?
geo = pd.DataFrame(
    {f"U{k}": [np.corrcoef(U[:, k], lat)[0, 1], np.corrcoef(U[:, k], lon)[0, 1]] for k in range(5)},
    index=["corr. with latitude", "corr. with longitude"],
)
geo.round(2)

# %% [markdown]
# ## 2e

# %%
# 2e(i) Guttman-Kaiser
# YOUR PART
# On z-scores, the variance of the k-th SVD score is sigma_k^2 / (N - 1) (lecture 07-4, sl. 12).
# Keep the largest Z such that this variance is > 1 for all k <= Z, i.e. every kept component
# has more variance than one original (normalized) feature.
N, D = X.shape
score_var = s**2 / (N - 1)
Z_gk = int(np.argmax(score_var <= 1)) if np.any(score_var <= 1) else D
print("variance of the first 8 SVD scores:", score_var[:8].round(2))
print("Guttman-Kaiser: Z =", Z_gk)

# %%
# 2e(ii) 90% squared Frobenius norm
# YOUR PART
# Smallest Z such that the first Z squared singular values make up >= 90% of the total
energy = np.cumsum(s**2) / np.sum(s**2)
Z_90 = int(np.argmax(energy >= 0.9)) + 1
print("cumulative share of squared Frobenius norm:", energy[:6].round(3))
print("90% of squared Frobenius norm: Z =", Z_90)

# %%
# 2e(iii) Scree test
# YOUR PART
# Scree plot: squared singular values in decreasing order (linear and log scale)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ks = np.arange(1, D + 1)
axes[0].plot(ks, s**2, marker="o")
axes[0].set_title("Scree plot")
axes[1].semilogy(ks, s**2, marker="o")
axes[1].set_title("Scree plot (log scale)")
for ax in axes:
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\sigma_k^2$")
    ax.axvline(5.5, color="red", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()
print("squared singular values 1-8:", (s[:8] ** 2).round(0))
print("ratio sigma_k^2 / sigma_{k+1}^2 for k = 1..7:", (s[:7] ** 2 / s[1:8] ** 2).round(2))

# %%
# 2e(iv) entropy
# YOUR PART
# Relative contributions f_k and their normalized entropy E (in [0, 1]);
# choose the smallest Z with f_1 + ... + f_Z >= E
f = s**2 / np.sum(s**2)
f_pos = f[f > 0]  # convention 0 * log(1/0) = 0
E = np.sum(f_pos * np.log(1 / f_pos)) / np.log(min(N, D))
Z_entropy = int(np.argmax(np.cumsum(f) >= E)) + 1
print(f"normalized entropy E = {E:.3f}")
print("Entropy-based method: Z =", Z_entropy)

# %%
# 2e(v) random flips
# Random sign matrix: np.random.choice([-1,1], X.shape)
# YOUR PART
# For each Z: residual X_{-Z} = X - X_Z, flip the sign of every entry at random, and compare
# spectral norms: (||X_{-Z}||_2 - ||flipped X_{-Z}||_2) / ||X_{-Z}||_F.
# Small values mean flipping barely changes the residual, i.e. it contains (almost) only noise.
# We average over several random sign matrices to reduce randomness.
np.random.seed(0)
Xa = X.to_numpy()
n_rep = 5
Zs = np.arange(0, 21)
flip_score = np.zeros(len(Zs))
for idx, Z in enumerate(Zs):
    R = Xa - svdcomp((U, s, Vt), range(Z)) if Z > 0 else Xa
    norm2, normF = np.linalg.norm(R, 2), np.linalg.norm(R, "fro")
    flipped = [np.linalg.norm(R * np.random.choice([-1, 1], R.shape), 2) for _ in range(n_rep)]
    flip_score[idx] = (norm2 - np.mean(flipped)) / normF

nextplot()
plt.plot(Zs, flip_score, marker="o")
plt.xlabel("Z")
plt.ylabel(r"$(\|X_{-Z}\|_2 - \|\tilde X_{-Z}\|_2) \,/\, \|X_{-Z}\|_F$")
plt.title("Random sign flips")
plt.xticks(Zs)
plt.show()
for Z, v in zip(Zs, flip_score):
    print(f"Z = {Z:2d}: {v:.4f}")

# %% [markdown]
# ## 2f

# %%
# Here is the empty plot that you need to fill (one line per choice of Z: RSME between
# original X and the reconstruction from size-Z SVD of noisy versions)
# YOUR PART
nextplot()
plt.plot()
plt.xlabel(r"Noise level ($\epsilon$)")
plt.ylabel("Reconstruction RMSE vs. original data")
