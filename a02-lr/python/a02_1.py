# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: py3
#     language: python
#     name: py3
# ---

# %% [markdown]
# # 1. Dataset Statistics

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy

# %load_ext autoreload
# %autoreload 2

from a02_helper import *
from a02_functions import normalize_data

# %%
# look some dataset statistics
scipy.stats.describe(X)

# %%
scipy.stats.describe(y)

# %%
# plot the distribution of all features
nextplot()
densities = [scipy.stats.gaussian_kde(X[:, j]) for j in range(D)]
xs = np.linspace(0, np.max(X), 200)
for j in range(D):
    plt.plot(xs, densities[j](xs), label=j)
plt.legend(ncol=5)
plt.show()

# %%
# this plots is not really helpful; go now explore further
# YOUR CODE HERE
import pandas as pd

# The three feature groups live on very different scales, so plot each group separately
# and cut the x-axis at the 99th percentile of the group to see where most of the mass is.
groups = {
    "word frequencies (features 0-47)": range(0, 48),
    "character frequencies (features 48-53)": range(48, 54),
    "capital run lengths (features 54-56)": range(54, 57),
}
for title, idx in groups.items():
    idx = list(idx)
    nextplot()
    xs_group = np.linspace(0, np.percentile(X[:, idx], 99), 200)
    for j in idx:
        plt.plot(xs_group, densities[j](xs_group), label=features[j])
    plt.title(f"Density of {title}")
    plt.legend(ncol=3, fontsize="x-small")
    plt.show()

# Scale and sparsity of each feature
stats = pd.DataFrame(
    {"mean": X.mean(axis=0), "max": X.max(axis=0), "share_zero": (X == 0).mean(axis=0)},
    index=features,
)
print(f"Median share of zeros per feature: {stats['share_zero'].median():.2f}")
stats.sort_values("max", ascending=False).head(10)

# %%
# Let's compute z-scores; create two new variables Xz and Xtestz by completing the
# `normalize_data` function in `a02_functions.py`. Once you implemented this function, Xz and
# Xtestz will be automatically provided to you in subsequent notebooks.
Xz, Xtestz = normalize_data(X, Xtest)


# %%
# Let's check.
np.mean(Xz, axis=0)  # should be all 0
np.var(Xz, axis=0)  # should be all 1
np.mean(Xtestz, axis=0)  # what do you get here?
np.var(Xtestz, axis=0)

np.sum(Xz**3)  # should be: 1925261.15

# %%
# Explore the normalized data
# YOUR CODE HERE
# Same density plot as above, now on the z-scores (all features on one common scale)
nextplot()
densities_z = [scipy.stats.gaussian_kde(Xz[:, j]) for j in range(D)]
xs_z = np.linspace(np.min(Xz), np.max(Xz), 400)
for j in range(D):
    plt.plot(xs_z, densities_z[j](xs_z), label=j)
plt.legend(ncol=5)
plt.title("Density of all features after z-score normalization")
plt.show()

# Zoom into the region where most of the data lies
nextplot()
xs_zoom = np.linspace(-1, 5, 400)
for j in range(D):
    plt.plot(xs_zoom, densities_z[j](xs_zoom), label=j)
plt.title("Zoom: z-scores between -1 and 5")
plt.show()

# Features with the most extreme z-scores (outliers after normalization)
stats_z = pd.DataFrame(
    {"min_z": Xz.min(axis=0), "max_z": Xz.max(axis=0), "share_zero": (X == 0).mean(axis=0)},
    index=features,
)
stats_z.sort_values("max_z", ascending=False).head(8)

# %%
