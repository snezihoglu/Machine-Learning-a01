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
# Answers obtained by hand, only by looking at the matrices (no computation).
#
# **M1:** Rows 1, 2 and 3 are identical and rows 4 and 5 are zero, so there is only one linearly
# independent row: rank 1.
# - sigma_1 = sqrt(sum of squared entries) = sqrt(9 ones) = 3
# - u_1 = (1, 1, 1, 0, 0)/sqrt(3)
# - v_1 = (1, 1, 1, 0, 0)/sqrt(3)
#
# **M2:** Rows 2, 3 and 4 are identical and rows 1 and 5 are zero, so rank 1.
# - sigma_1 = sqrt(3 * (2^2 + 1^2 + 2^2)) = sqrt(27) = 3 sqrt(3) ≈ 5.196
# - u_1 = (0, 1, 1, 1, 0)/sqrt(3)
# - v_1 = (0, 2, 1, 2, 0)/3
#
# **M3:** Rows 2 to 5 are identical and row 1 is zero, so rank 1.
# - sigma_1 = sqrt(4 * (1^2 + 1^2 + 1^2)) = sqrt(12) = 2 sqrt(3) ≈ 3.464
# - u_1 = (0, 1, 1, 1, 1)/2
# - v_1 = (0, 1, 1, 1)/sqrt(3)
#
# **M4:** Two independent blocks of ones (3x3 and 2x2), each contributing one singular value and one
# pair of singular vectors: rank 2.
# - sigma_1 = sqrt(9 ones) = 3, u_1 = v_1 = (1, 1, 1, 0, 0)/sqrt(3)
# - sigma_2 = sqrt(4 ones) = 2, u_2 = v_2 = (0, 0, 0, 1, 1)/sqrt(2)
#
# **M5:** Rows 1/2 and rows 4/5 repeat patterns; I considered row 3 to be the sum of row 1 and
# row 4, giving rank 2. The matrix is symmetric. I failed to determine the exact singular values
# and vectors by inspection.
#
# **M6:** Rows 1, 2, 4 and 5 are identical; row 3 differs (zero in the middle), giving two
# independent row patterns: rank 2. The matrix is symmetric. I failed to determine the exact
# singular values and vectors by inspection.

# %% [markdown]
# ## 1b

# %%
# YOUR PART
# Compute the SVD of each matrix and print it in the same form as the answers of 1a:
# rank, singular values, and the singular vectors belonging to the non-zero singular values.
matrices = {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5, "M6": M6}
np.set_printoptions(precision=4, suppress=True)

for name, M in matrices.items():
    U, s, Vt = svd(M)
    rank = matrix_rank(M)
    print(f"{name}: rank = {rank}, singular values = {s}")
    for k in range(rank):
        print(f"   sigma_{k + 1} = {s[k]:.4f}")
        print(f"   u_{k + 1} = {U[:, k]}")
        print(f"   v_{k + 1} = {Vt[k, :]}")
    print()

# %% [markdown]
# ## 1c

# %%
# You can use the functions svdcomp and plot_matrix from a03_helper.py to visualize the results.
# YOUR PART
# Best rank-1 approximation = first layer sigma_1 u_1 v_1^T, i.e. svdcomp(M, range(1)).
# Top row: original matrices, bottom row: their best rank-1 approximations (same color scale).
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for col, (name, M) in enumerate(matrices.items()):
    M_rank1 = svdcomp(M, range(1))
    for row, (A, title) in enumerate([(M, name), (M_rank1, f"{name}: rank-1 approx.")]):
        ax = axes[row, col]
        ax.matshow(A, cmap="PiYG", vmin=-2, vmax=2)
        for (i, j), value in np.ndenumerate(A):
            ax.text(j, i, f"{value:.2f}".rstrip("0").rstrip("."), ha="center", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()

# How much of each matrix does the first layer capture?
# Share of the squared Frobenius norm: sigma_1^2 / sum_k sigma_k^2
for name, M in matrices.items():
    s = svd(M, compute_uv=False)
    print(f"{name}: sigma_1^2 / sum sigma_k^2 = {s[0] ** 2 / np.sum(s ** 2):.3f}")

# %% [markdown]
# ## 1d

# %%
# Another method to compute the rank is matrix_rank.
# YOUR PART
# Raw singular values of M6 in full precision (no rounding / suppression of tiny values)
np.set_printoptions(precision=17, suppress=False)
s6 = svd(M6, compute_uv=False)
print("singular values of M6:", s6)
print("number of singular values that are exactly non-zero:", np.sum(s6 != 0))

# matrix_rank treats singular values below a tolerance as zero.
# Default tolerance: largest singular value * max(N, D) * machine epsilon
tol = s6.max() * max(M6.shape) * np.finfo(float).eps
print(f"machine epsilon = {np.finfo(float).eps:.3g}, tolerance = {tol:.3g}")
print("number of singular values above the tolerance:", np.sum(s6 > tol))
print("matrix_rank(M6) =", matrix_rank(M6))

# Exact values for comparison: M6 = J - e3 e3^T has singular values 2 + 2 sqrt(2) and 2 sqrt(2) - 2
print("2 + 2 sqrt(2) =", 2 + 2 * np.sqrt(2), "  2 sqrt(2) - 2 =", 2 * np.sqrt(2) - 2)
np.set_printoptions(precision=4, suppress=True)
