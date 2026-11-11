# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import math
import scipy
import psutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython import get_ipython
import psutil
inTerminal = not "IPKernelApp" in get_ipython().config
inJupyterNb = any(filter(lambda x: x.endswith("jupyter-notebook"), psutil.Process().parent().cmdline()))
get_ipython().run_line_magic("matplotlib", "" if inTerminal else "notebook" if inJupyterNb else "widget")

from a02_functions import normalize_data

# %%
# Plotting
inTerminal = not "IPKernelApp" in get_ipython().config
inJupyterNb = any(
    filter(
        lambda x: x.endswith("jupyter-notebook"), psutil.Process().parent().cmdline()
    )
)
inJupyterLab = any(
    filter(lambda x: x.endswith("jupyter-lab"), psutil.Process().parent().cmdline())
)

get_ipython().run_line_magic(
    "matplotlib", "" if inTerminal else "notebook" if inJupyterNb else "widget"
)


# %%
def nextplot():
    if inTerminal:
        plt.clf()  # this clears the current plot
    else:
        plt.figure()  # this creates a new plot
# %%
def plot_matrix(
    M, lim=None, cmap="PiYG", labels="{:.3g}", rownames=None, colnames=None, **kwargs
):
    """Plot the given matrix in a labeled heatmap.

    `M` is the matrix or Pandas data frame.

    `lim` controls the range for the color scale for the entries. If unset, range of
    (-1,+1)*maximum value in `M`. If its a single integer, use range (-1,+1)*lim. Else
    `lim` should be a tuple of (minimum value, maximum value). `cmap` is the colormap
    being used.

    `labels` is a format string used to print the values of each matrix entry. If set to
    `None`, values are not printed (e.g., that's useful for very large matrices).

    `rownames` and `colnames` can be explicitly specified. If they are unset and `M` is
    a Pandas dataframe, row and column names are used from `M`. Otherwise, use indexes.

    """
    if isinstance(M, pd.DataFrame):
        if not colnames:
            colnames = M.columns.values
        if not rownames:
            rownames = M.index.values
        M = M.to_numpy()
    if lim is None:
        lim = np.max(np.abs((M[:, :])))
    if not isinstance(lim, tuple):
        lim = (-lim, lim)
    lim_mean = (lim[0] + lim[1]) / 2.0
    lim_spread = lim[1] - lim_mean

    nextplot()
    plt.matshow(M, fignum=0, cmap=cmap, vmin=lim[0], vmax=lim[1], **kwargs)
    if labels:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(
                    j,
                    i,
                    labels.format(M[i, j]),
                    color="white"
                    if np.abs(M[i, j] - lim_mean) > lim_spread / 2.0
                    else "black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    plt.gca().set_xticks(range(M.shape[1]))
    if colnames is not None:
        plt.gca().set_xticklabels(colnames, rotation="vertical")
    plt.gca().set_yticks(range(M.shape[0]))
    if rownames is not None:
        plt.gca().set_yticklabels(rownames)
    plt.colorbar()

# %%
# Data (loading)
data = scipy.io.loadmat("data/spamData.mat")
X = data["Xtrain"]
N = X.shape[0]
D = X.shape[1]
Xtest = data["Xtest"]
Ntest = Xtest.shape[0]
y = data["ytrain"].squeeze().astype(int)
ytest = data["ytest"].squeeze().astype(int)

features = np.array(
    [
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_(",
        "char_freq_[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
)

# %%
# Normalization
if normalize_data(X, Xtest) is not None:
    Xz, Xtestz = normalize_data(X, Xtest)# use the normalized dataset throughout the assignment
