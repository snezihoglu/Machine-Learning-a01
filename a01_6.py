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
# # 6 Missing Data
#
# The three derivations below are the graded part of Task 6. The sampling
# experiment at the end is optional.
#
# Let O be the set of observed feature indices. By Naive Bayes,
#
# $$p(y=c \mid x_O) =
# \frac{p(y=c)\prod_{j\in O}p(x_j\mid y=c)}
# {\sum_{c'=0}^{C-1}p(y=c')\prod_{j\in O}p(x_j\mid y=c')}.$$
#
# **6a: all features observed.** Set O={1,...,D} in the formula above. This
# gives the class posterior used to classify a complete example. It is useful
# when every feature of an example is available.
#
# **6b: only a prefix observed.** Set O={1,...,D'}. Thus
#
# $$p(y=c\mid x_{1:D'}) \propto
# p(y=c)\prod_{j=1}^{D'}p(x_j\mid y=c),$$
#
# with the normalizer obtained by summing the right-hand side over classes. It
# is useful for classifying incomplete examples, for example when only part of
# an image has arrived or is visible.
#
# **6c: distribution of the missing suffix.** Marginalize over the unknown
# class and use conditional independence of the features given that class:
#
# $$p(x_{D'+1:D}\mid x_{1:D'}) =
# \sum_{c=0}^{C-1}p(y=c\mid x_{1:D'})
# \prod_{j=D'+1}^{D}p(x_j\mid y=c).$$
#
# This describes plausible missing values given the observed prefix. It can be
# used to complete an incomplete example while preserving uncertainty; it is a
# distribution over completions, not just one guessed image.

# %%
from a01_helper import *
from a01_functions import logsumexp, nb_train

# %% [markdown]
# ## Optional sampling experiment
# Sample a class from the prefix posterior, then sample the missing features
# independently from that class's feature distributions. This produces draws
# from the Task 6c mixture distribution.

# %%
def posterior_given_prefix(model, xprefix):
    """Return p(y | xprefix) for one observed feature prefix."""
    logcls = model["logcls"]
    logpriors = model["logpriors"]
    C, D, K = logcls.shape
    xprefix = np.asarray(xprefix)
    if xprefix.ndim != 1:
        raise ValueError("xprefix must be one-dimensional")
    if len(xprefix) > D:
        raise ValueError("xprefix cannot contain more than D features")
    if not np.issubdtype(xprefix.dtype, np.integer):
        raise TypeError("xprefix must contain integer feature values")
    if np.any((xprefix < 0) | (xprefix >= K)):
        raise ValueError(f"feature values must be in the range [0, {K - 1}]")

    logjoint = logpriors.copy()
    if len(xprefix) > 0:
        logjoint += logcls[:, np.arange(len(xprefix)), xprefix].sum(axis=1)
    return np.exp(logjoint - logsumexp(logjoint))


def sample_suffix_completions(model, xprefix, n_samples, rng=None):
    """Sample full feature vectors conditional on a supplied feature prefix."""
    logcls = model["logcls"]
    C, D, K = logcls.shape
    xprefix = np.asarray(xprefix)
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer")
    class_posterior = posterior_given_prefix(model, xprefix)
    rng = np.random.default_rng() if rng is None else rng
    samples = np.empty((n_samples, D), dtype=np.int64)
    samples[:, : len(xprefix)] = xprefix
    for i in range(n_samples):
        c = rng.choice(C, p=class_posterior)
        for j in range(len(xprefix), D):
            samples[i, j] = rng.choice(K, p=np.exp(logcls[c, j, :]))
    return samples


# Train once, observe the top 14 image rows, and draw five plausible suffixes.
model_missing = nb_train(X, y, alpha=2)
Dprime = 14 * 28
x_full = Xtest[0].astype(np.int64)
xprefix = x_full[:Dprime]
prefix_posterior = posterior_given_prefix(model_missing, xprefix)
rng = np.random.default_rng(0)
completions = sample_suffix_completions(model_missing, xprefix, 5, rng=rng)
print("True label:", ytest[0])
print("Most probable class from observed prefix:", np.argmax(prefix_posterior))

partial_image = np.zeros_like(x_full)
partial_image[:Dprime] = xprefix
fig, axes = plt.subplots(7, 1, figsize=(3, 12))
axes[0].imshow(x_full.reshape(28, 28), cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original test image")
axes[1].imshow(partial_image.reshape(28, 28), cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Observed prefix (remaining pixels hidden)")
for i, sample in enumerate(completions, start=2):
    axes[i].imshow(sample.reshape(28, 28), cmap="gray", vmin=0, vmax=255)
    axes[i].set_title(f"Sampled completion {i - 1}")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig("task6_optional_completions.png", bbox_inches="tight")
plt.show()
