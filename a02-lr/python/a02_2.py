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

# %%
import numpy as np
import matplotlib as plt

# %load_ext autoreload
# %autoreload 2

from a02_helper import *
from a02_functions import (
    sigma,
    logsigma,
    er,
    d_er,
    gd_er,
    sgdepoch_er,
    sgd_er,
    optimize,
)

# %% [markdown]
# # 2. Empirical Risk Minimization

# %% [markdown]
# ## Helper functions

# %% [markdown]
# Define the logistic function in `a02_functions.py`. Make sure it operates on both
# scalars and vectors. Then test it below.

# %%
# this should give:
# [0.5, array([0.26894142, 0.5, 0.73105858])]
[sigma(0), sigma(np.array([-1, 0, 1]))]

# %% [markdown]
# Define the logarithm of the logistic function in `a02_functions.py`. Make sure it
# operates on both scalars and vectors.

# %%
# this should give:
# [-0.69314718055994529, array([-1.31326169, -0.69314718, -0.31326169])]
[logsigma(0), logsigma(np.array([-1, 0, 1]))]

# %% [markdown]
# ## 2b Empirical risk and gradient

# %% [markdown]
# Complete the function for the empirical risk of the logistic regression model in
# `a02_functions.py` and test it below.

# %%
# this should give:
# 15.356163676288993
er(y, Xz, np.linspace(-5, 5, D))

# %% [markdown]
# Now fill out the function that returns the gradient of the empirical risk of the
# logistic regression model in `a02_functions.py`, and test it below.

# %%
# this should give:
# array([-0.1798825 , -0.04693023, -0.27466027, -0.05118185, -0.26186368,
#        -0.25969397, -0.3003884 , -0.20292501, -0.21506925, -0.15360933,
#        -0.25165549, -0.11497659, -0.14866875, -0.07646526, -0.18350882,
#        -0.28216633, -0.25683434, -0.21190226, -0.29450173, -0.17748953,
#        -0.38329083, -0.03926913, -0.27393521, -0.20662464,  0.23056057,
#         0.20561304,  0.18575698,  0.17210766,  0.11730408,  0.1555117 ,
#         0.1342924 ,  0.12238809,  0.11268253,  0.12274729,  0.13289319,
#         0.14885237,  0.19473723,  0.03522699,  0.12881622,  0.07477366,
#         0.09411536,  0.11815139,  0.1471057 ,  0.09038803,  0.13539737,
#         0.14756532,  0.05466443,  0.08838642,  0.08228431,  0.11671288,
#         0.08454313, -0.13649572, -0.19724037, -0.01406327, -0.04967132,
#        -0.12338248, -0.1357652 ])
d_er(y, Xz, np.linspace(-5, 5, D))

# %% [markdown]
# ## 2c Gradient descent

# %% [markdown]
# In `a02_functions.py`, define the objective and update function for one
# gradient-descent epoch for fitting an MLE estimate of logistic regression with
# gradient descent (should return a tuple of two functions; see `optimize` in
# `a02_functions.py`). Then test it below.

# %%
# this should give
# [15.356163676288993,
#  array([-4.98486859, -5.00348273, -4.97957969, -4.99464192, -4.98028549,
#         -4.97881798, -4.97516411, -4.98255264, -4.97950841, -4.98353035,
#         -4.97505258, -4.99343004, -4.98739394, -4.991352  , -4.97951014,
#         -4.97641948, -4.97550312, -4.98109859, -4.97826941, -4.98042354,
#         -4.96918027, -4.98940344, -4.97057868, -4.980086  , -5.00671666,
#         -5.00338624, -5.01360572, -4.9991855 , -4.99962596, -4.9990924 ,
#         -4.99728516, -4.99710625, -5.00490407, -4.99711359, -4.99844041,
#         -4.99879209, -5.00255707, -4.99927668, -5.00020121, -4.99270287,
#         -5.00059118, -5.00229197, -4.9975015 , -5.00292195, -5.00493102,
#         -5.00724744, -4.99867731, -5.00170438, -4.99836863, -4.99853799,
#         -4.9994118 , -4.98326416, -4.97266254, -4.99221473, -4.98906378,
#         -4.98190217, -4.97744729])]
f, update = gd_er(y, Xz)
[f(np.linspace(-5, 5, D)), update(np.linspace(-5, -5, D), 0.1)]

# %%
# Now you can run gradient descent!
np.random.seed(0)
w0 = np.random.normal(size=D)
wz_gd, vz_gd, ez_gd = optimize(gd_er(y, Xz), w0, nepochs=500)

# %%
# look at how gradient descent made progess
# YOUR CODE HERE

# %% [markdown]
# ## 2d Stochastic gradient descent

# %% [markdown]
# Implement stochastic gradient descent in `a02_functions.py`. Then test it with the
# given results below.

# %%
# when you run this multiple times, with 50% probability you should get the
# following result (there is one other result which is very close):
# array([ -3.43689655e+02,  -1.71161311e+02,  -5.71093536e+02,
#         -5.16478220e+01,   4.66294348e+02,  -3.71589878e+02,
#          5.21493183e+02,   1.25699230e+03,   8.33804130e+02,
#          5.63185399e+02,   1.32761302e+03,  -2.64104011e+02,
#          7.10693307e+02,  -1.75497331e+02,  -1.94174427e+02,
#          1.11641507e+02,  -3.30817509e+02,  -3.46754913e+02,
#          8.48722111e+02,  -1.89136304e+02,  -4.25693844e+02,
#         -1.23084189e+02,  -2.95894797e+02,  -2.35789333e+02,
#         -3.38695243e+02,  -3.05642830e+02,  -2.28975383e+02,
#         -2.38075137e+02,  -1.66702530e+02,  -2.27341599e+02,
#         -1.77575620e+02,  -1.49093855e+02,  -1.70028859e+02,
#         -1.50243833e+02,  -1.82986008e+02,  -2.41143708e+02,
#         -3.31047159e+02,  -5.79991185e+01,  -1.98477863e+02,
#         -1.91264948e+02,  -1.17371919e+02,  -1.66953779e+02,
#         -2.01472565e+02,  -1.23330949e+02,  -3.00857740e+02,
#         -1.95853348e+02,  -7.44868073e+01,  -1.11172370e+02,
#         -1.57618226e+02,  -1.25729512e+00,  -1.45536466e+02,
#         -1.43362438e+02,  -3.00429708e+02,  -9.84391082e+01,
#         -4.54152047e+01,  -5.26492232e+01,  -1.45175427e+02])
sgdepoch_er(y[1:3], Xz[1:3, :], np.linspace(-5, 5, D), 1000)

# %%
# with 50% probability, you should get:
# [20.43248652284754,
#  array([ -3.43689655e+02,  -1.71161311e+02,  -5.71093536e+02,
#          -5.16478220e+01,   4.66294348e+02,  -3.71589878e+02,
#           5.21493183e+02,   1.25699230e+03,   8.33804130e+02,
#           5.63185399e+02,   1.32761302e+03,  -2.64104011e+02,
#           7.10693307e+02,  -1.75497331e+02,  -1.94174427e+02,
#           1.11641507e+02,  -3.30817509e+02,  -3.46754913e+02,
#           8.48722111e+02,  -1.89136304e+02,  -4.25693844e+02,
#          -1.23084189e+02,  -2.95894797e+02,  -2.35789333e+02,
#          -3.38695243e+02,  -3.05642830e+02,  -2.28975383e+02,
#          -2.38075137e+02,  -1.66702530e+02,  -2.27341599e+02,
#          -1.77575620e+02,  -1.49093855e+02,  -1.70028859e+02,
#          -1.50243833e+02,  -1.82986008e+02,  -2.41143708e+02,
#          -3.31047159e+02,  -5.79991185e+01,  -1.98477863e+02,
#          -1.91264948e+02,  -1.17371919e+02,  -1.66953779e+02,
#          -2.01472565e+02,  -1.23330949e+02,  -3.00857740e+02,
#          -1.95853348e+02,  -7.44868073e+01,  -1.11172370e+02,
#          -1.57618226e+02,  -1.25729512e+00,  -1.45536466e+02,
#          -1.43362438e+02,  -3.00429708e+02,  -9.84391082e+01,
#          -4.54152047e+01,  -5.26492232e+01,  -1.45175427e+02])]
f, update = sgd_er(y[1:3], Xz[1:3, :])
[f(np.linspace(-5, 5, D)), update(np.linspace(-5, 5, D), 1000)]

# %%
# you can run stochastic gradient descent!
wz_sgd, vz_sgd, ez_sgd = optimize(sgd_er(y, Xz), w0, nepochs=500)

# %% [markdown]
# ## 2e Compare GD and SGD

# %% [markdown]
# Explore the behavior of both methods, given the parameters provided to you!

# %%
# YOUR CODE HERE
