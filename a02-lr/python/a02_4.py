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
# # 4 Maximum Aposteriori Estimation

# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# %load_ext autoreload
# %autoreload 2

from a02_helper import *
from a02_functions import er, rr, d_rr, gd_er, gd_rr, logsigma, classify, optimize

# %% [markdown]
# ## 4a Gradient Descent

# %% [markdown]
# Implement the function returning the regularized risk of logistic
# regression, regularized with parameter lambda, in `a02_functions.py`. Then test it
# below.

# %%
# this should give:
# [15.356163676288993, 261.33830653343193]
[rr(y, Xz, np.linspace(-5, 5, D), 0), rr(y, Xz, np.linspace(-5, 5, D), 1)]

# %% [markdown]
# Now implement the function to obtain its gradient and test it, in the same manner as
# before.

# %%
# this should give:
# [array([-0.1798825 , -0.04693023, -0.27466027, -0.05118185, -0.26186368,
#         -0.25969397, -0.3003884 , -0.20292501, -0.21506925, -0.15360933,
#         -0.25165549, -0.11497659, -0.14866875, -0.07646526, -0.18350882,
#         -0.28216633, -0.25683434, -0.21190226, -0.29450173, -0.17748953,
#         -0.38329083, -0.03926913, -0.27393521, -0.20662464,  0.23056057,
#          0.20561304,  0.18575698,  0.17210766,  0.11730408,  0.1555117 ,
#          0.1342924 ,  0.12238809,  0.11268253,  0.12274729,  0.13289319,
#          0.14885237,  0.19473723,  0.03522699,  0.12881622,  0.07477366,
#          0.09411536,  0.11815139,  0.1471057 ,  0.09038803,  0.13539737,
#          0.14756532,  0.05466443,  0.08838642,  0.08228431,  0.11671288,
#          0.08454313, -0.13649572, -0.19724037, -0.01406327, -0.04967132,
#         -0.12338248, -0.1357652 ]),
#  array([-5.1798825 , -4.8683588 , -4.91751742, -4.51546757, -4.54757796,
#         -4.36683683, -4.22895983, -3.95292501, -3.78649782, -3.54646647,
#         -3.4659412 , -3.15069088, -3.00581161, -2.75503668, -2.68350882,
#         -2.60359491, -2.39969148, -2.17618797, -2.08021602, -1.78463238,
#         -1.81186225, -1.28926913, -1.34536378, -1.09948179, -0.48372514,
#         -0.33010124, -0.17138588, -0.00646377,  0.11730408,  0.33408312,
#          0.49143526,  0.65810238,  0.82696825,  1.01560443,  1.20432176,
#          1.39885237,  1.62330866,  1.64236985,  1.9145305 ,  2.03905937,
#          2.2369725 ,  2.43957997,  2.6471057 ,  2.76895946,  2.99254023,
#          3.18327961,  3.26895015,  3.48124356,  3.65371288,  3.86671288,
#          4.01311456,  3.97064714,  4.08847392,  4.45022245,  4.59318583,
#          4.69804609,  4.8642348 ])]
[d_rr(y, Xz, np.linspace(-5, 5, D), 0), d_rr(y, Xz, np.linspace(-5, 5, D), 1)]

# %% [markdown]
# Now define the (f,update) tuple handed to the `optimize` function for gradient descent
# on logistic regression with L2 regularization. Then run it below.

# %%
# %%
# let's run!
lambda_ = 0.1
np.random.seed(0)
w0 = np.random.normal(size=D)
wz_gd_rr, vz_gd_rr, ez_gd_rr = optimize(
    gd_rr(y, Xz, lambda_=lambda_), w0, nepochs=500, verbose=True
)

# %% [markdown]
# ## 4b Effect of Prior

# %%
# YOUR CODE HERE

# %% [markdown]
# ## 4c Composition of Weight Vector

# %%
# YOUR CODE HERE

# %%
