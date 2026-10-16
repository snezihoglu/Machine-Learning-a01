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
# # 3 Experiments on MNIST Digits Data

# %%
import sklearn, sklearn.metrics
# %load_ext autoreload
# %autoreload 2

from a01_helper import *
from a01_functions import nb_train, nb_predict

# %%
# Let's train the model on the digits data and predict
model_nb2 = nb_train(X, y, alpha=2)
pred_nb2 = nb_predict(model_nb2, Xtest)
yhat = pred_nb2["yhat"]
logprob = pred_nb2["logprob"]

# %%
# Accuracy
sklearn.metrics.accuracy_score(ytest, yhat)

# %%
# show some digits grouped by prediction; can you spot errors?
nextplot()
showdigits(Xtest, yhat)
plt.suptitle("Digits grouped by predicted label")
plt.show()

# %%
# do the same, but this time show wrong predicitions only
perror = ytest != yhat
nextplot()
showdigits(Xtest[perror, :], yhat[perror])
plt.suptitle("Errors grouped by predicted label")
plt.show()


# %%
# do the same, but this time on a sample of wrong preditions to see
# error proportions
ierror_s = np.random.choice(np.where(perror)[0], 100, replace=False)
nextplot()
showdigits(Xtest[ierror_s, :], yhat[ierror_s])
plt.suptitle("Errors grouped by predicted label")
plt.show()

# %%
# now let's look at this in more detail
print(sklearn.metrics.classification_report(ytest, yhat))
print(sklearn.metrics.confusion_matrix(ytest, yhat))  # true x predicted

# %%
# plot the confusion matrix
nextplot()
M = sklearn.metrics.confusion_matrix(ytest, yhat)
plt.imshow(M, origin="upper")
for ij, v in np.ndenumerate(M):
    i, j = ij
    plt.text(j, i, str(v), color="white", ha="center", va="center")
plt.xlabel("predicted")
plt.ylabel("true")
plt.colorbar()
plt.show()

# %%
# cumulative accuracy for predictions ordered by confidence (labels show predicted
# confidence)
order = np.argsort(logprob)[::-1]
accuracies = np.cumsum(ytest[order] == yhat[order]) / (np.arange(len(yhat)) + 1)
nextplot()
plt.plot(accuracies)
plt.xlabel("Predictions ordered by confidence")
plt.ylabel("Accuracy")
for x in np.linspace(0.7, 1, 10, endpoint=False):
    index = int(x * (accuracies.size - 1))
    print(np.exp(logprob[order][index]))
    plt.text(index, accuracies[index], "{:.10f}".format(np.exp(logprob[order][index])))
plt.show()

# %%
# Accuracy for predictions grouped by confidence (labels show
# predicted confidence). Make the plot large (or reduce number of bins) to see
# the labels.
bins = (np.linspace(0, 1, 50) * len(yhat)).astype(int)
mean_accuracy = [
    np.mean(ytest[order][bins[i] : bins[i + 1]] == yhat[order][bins[i] : bins[i + 1]])
    for i in range(len(bins) - 1)
]
nextplot()
plt.bar(np.arange(len(mean_accuracy)), mean_accuracy)
plt.xticks(
    np.arange(len(mean_accuracy)),
    [
        "{:.10f}".format(x)
        for x in np.exp(logprob[order][np.append(bins[1:-1], len(yhat) - 1)])
    ],
)
plt.gcf().autofmt_xdate()
plt.xlabel("Confidence bin")
plt.ylabel("Accuracy")
plt.show()

# %% [markdown]
# ## Additional analysis for the discussion
# Most frequent confusions and how confident the model is in its predictions.

# %%
# Most frequent confusions (true -> predicted), off-diagonal entries of M
M_off = M.copy()
np.fill_diagonal(M_off, 0)
print("Most frequent confusions (true -> predicted):")
for k in np.argsort(-M_off.ravel())[:6]:
    i, j = np.unravel_index(k, M_off.shape)
    print(f"  {i} -> {j}: {M_off[i, j]}")

# Overconfidence: how often is the model (almost) certain, and how often is it right then?
conf = np.exp(logprob)
correct = yhat == ytest
print(f"Predictions with confidence > 0.99: {np.mean(conf > 0.99):.1%}")
print(f"  accuracy among them:              {np.mean(correct[conf > 0.99]):.1%}")
print(f"Errors with confidence > 0.99:      {np.mean(conf[~correct] > 0.99):.1%}")
print(f"Median confidence of errors:        {np.median(conf[~correct]):.4f}")
