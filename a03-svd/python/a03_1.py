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
# # 1 Intuition on SVD

# %%
import numpy as np
from numpy.linalg import svd as svd
from numpy.linalg import matrix_rank as matrix_rank
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from a03_helper import *

# %%
M1 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

M2 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ]
)

M3 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])

M4 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
)

M5 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]
)

M6 = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
)


# %% [markdown]
# ## 1a

# %%
# Your part

# %% [markdown]
# ## 1b

# %%
# YOUR PART

# %% [markdown]
# ## 1c

# %%
# You can use the functions svdcomp and plot_matrix from a03_helper.py to visualize the results.
# YOUR PART

# %% [markdown]
# ## 1d

# %%
# Another method to compute the rank is matrix_rank.
# YOUR PART
