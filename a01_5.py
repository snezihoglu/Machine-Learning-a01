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
# # 5 Generating Data

# %%
from pathlib import Path

# %load_ext autoreload
# %autoreload 2

from a01_helper import *
from a01_functions import nb_train, nb_generate

# %%
# Generate 15 digits from each class for several smoothing values. Each model
# is trained once; the alpha=2 model is reused for the parameter plots below.
alpha_values = [1, 2, 10]
models_by_alpha = {
    alpha: nb_train(X, y, alpha=alpha) for alpha in alpha_values
}
model_nb2 = models_by_alpha[2]
ygen = np.repeat(np.arange(10), 15)
Xgen_by_alpha = {
    alpha: nb_generate(models_by_alpha[alpha], ygen) for alpha in alpha_values
}

plot_dir = Path("task5_plots")
plot_dir.mkdir(exist_ok=True)

for alpha in alpha_values:
    nextplot()
    showdigits(Xgen_by_alpha[alpha], ygen)
    title = (
        "Some generated digits for each class (alpha=2)"
        if alpha == 2
        else f"Generated digits (alpha={alpha})"
    )
    plt.suptitle(title)
    plt.savefig(plot_dir / f"generated_alpha_{alpha}.png", bbox_inches="tight")
    plt.show()

# Reuse the alpha=2 samples for the parameter plots below.
Xgen = Xgen_by_alpha[2]

# %%
# we can also plot the parameter vectors by choosing the most-likely
# value for each feature
ymax = np.arange(10)
Xmax = np.zeros((10, D))
for c in range(10):
    Xmax[c,] = np.apply_along_axis(np.argmax, 1, model_nb2["logcls"][c, :, :])

nextplot()
showdigits(Xmax, ymax)
plt.suptitle("Most likely value of each feature per class")
plt.savefig(plot_dir / "most_likely_alpha_2.png", bbox_inches="tight")
plt.show()

# %%
# Or the expected value of each feature. Here we leave the categorical domain
# and treat each feature as a number, i.e., this is NOT how categorical Naive
# Bayes sees it and we wouldn't be able to do this if the data were really
# categorical.
ymean = np.arange(10)
Xmean = np.zeros((10, D))
for c in range(10):
    Xmean[c,] = np.apply_along_axis(
        np.sum, 1, np.exp(model_nb2["logcls"][c, :, :]) * np.arange(256)
    )

nextplot()
showdigits(Xmean, ymean)
plt.suptitle("Expected value of each feature per class")
plt.savefig(plot_dir / "expected_value_alpha_2.png", bbox_inches="tight")
plt.show()


# %%
