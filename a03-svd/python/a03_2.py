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

# %% [markdown]
# ## 2c

# %%
# Here is an example plot.
plot_xy(lon, lat, U[:, 0])

# %%
# For interpretation, it may also help to look at the other component matrices and
# perhaps use other plot functions (e.g., plot_matrix).
# YOUR PART

# %% [markdown]
# ## 2d

# %%
# Here is an example.
plot_xy(U[:, 0], U[:, 1], lat - np.mean(lat))

# %% [markdown]
# ## 2e

# %%
# 2e(i) Guttman-Kaiser
# YOUR PART

# %%
# 2e(ii) 90% squared Frobenius norm
# YOUR PART

# %%
# 2e(iii) Scree test
# YOUR PART

# %%
# 2e(iv) entropy
# YOUR PART

# %%
# 2e(v) random flips
# Random sign matrix: np.random.choice([-1,1], X.shape)
# YOUR PART

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
