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

# %% [markdown]
# # 2 Prediction

# %%
# %load_ext autoreload
# %autoreload 2
from a01_helper import *
from a01_functions import nb_train, nb_predict

# Test your code
model = nb_train(Xex, yex, alpha=2)
nb_predict(model, Xex)
# This should produce:
# {'logprob': array([-0.41925843, -0.55388511, -0.68309684, -0.29804486]),
#  'yhat': array([0, 1, 2, 0], dtype=int64)}

# %% [markdown]
# ## 2b) Prediction on the MNIST digits data
# Train on the full training set and predict the test sample (10 digits per class).

# %%
model = nb_train(X, y, alpha=2)
pred = nb_predict(model, Xtest_s)

print("true:     ", ytest_s[:20])
print("predicted:", pred["yhat"][:20])
print("confidence P(yhat|x):", np.round(np.exp(pred["logprob"][:20]), 3))
print(f"correct on sample: {np.sum(pred['yhat'] == ytest_s)} of {Ntest_s}")
