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
# # 3 Prediction

# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# %load_ext autoreload
# %autoreload 2

from a02_helper import *
from a02_functions import gd_er, predict, classify, optimize

# %%
# Fitted model
w0 = np.random.normal(size=D)
wz_gd, vz_gd, ez_gd = optimize(gd_er(y, Xz), w0, nepochs=500)

# %% [markdown]
# In `a02_functions.py`, complete the `predict` and `classify` methods for the predicted
# spam probability and predicted class label, respectively. Use them to explore your
# previously fitted model.

# %%
# Exploration example: confusion matrix
yhat = predict(Xtestz, wz_gd)
ypred = classify(Xtestz, wz_gd)
print(sklearn.metrics.confusion_matrix(ytest, ypred))  # true x predicted

# %%
# Exploration example: classification report
print(sklearn.metrics.classification_report(ytest, ypred))

# %%
# Exploration Example: precision-recall curve (with annotated thresholds)
nextplot()
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(ytest, yhat)
plt.plot(recall, precision)
for x in np.linspace(0, 1, 10, endpoint=False):
    index = int(x * (precision.size - 1))
    plt.text(recall[index], precision[index], "{:3.2f}".format(thresholds[index]))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# %%
# Explore which features are considered important
# YOUR CODE HERE
