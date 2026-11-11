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

# %%
