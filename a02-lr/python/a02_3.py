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
from a02_functions import gd_er, predict, classify, optimize, sgd_er, er

# %%
# Fitted model
np.random.seed(0)
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
# Fit the second model from Task 2 with the same initialization.
np.random.seed(0)
w0_sgd = np.random.normal(size=D)
wz_sgd, vz_sgd, ez_sgd = optimize(sgd_er(y, Xz), w0_sgd, nepochs=500, verbose=False)

task3_models = {"GD": wz_gd, "SGD": wz_sgd}
task3_probabilities = {name: predict(Xtestz, w) for name, w in task3_models.items()}
task3_labels = {name: classify(Xtestz, w) for name, w in task3_models.items()}

for name in task3_models:
    print(f"\n{name} evaluation")
    print("Accuracy:", sklearn.metrics.accuracy_score(ytest, task3_labels[name]))
    print("Mean test log loss:", er(ytest, Xtestz, task3_models[name]))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(sklearn.metrics.confusion_matrix(ytest, task3_labels[name], labels=[0, 1]))
    print(sklearn.metrics.classification_report(ytest, task3_labels[name], labels=[0, 1], digits=4, zero_division=0))

from pathlib import Path
task3_results = Path("../results/task3")
task3_results.mkdir(parents=True, exist_ok=True)

# Compare test-set confusion matrices for the two Task 2 models.
fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
for ax, (name, labels_hat) in zip(axes, task3_labels.items()):
    matrix = sklearn.metrics.confusion_matrix(ytest, labels_hat, labels=[0, 1])
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center")
    ax.set(title=name, xlabel="Predicted label", ylabel="True label", xticks=[0, 1], yticks=[0, 1])
fig.suptitle("Test-set confusion matrices")
fig.savefig(task3_results / "confusion_matrices.png", dpi=180, bbox_inches="tight")
plt.show()

# Compare precision-recall curves using the test-set spam probabilities.
fig, ax = plt.subplots(figsize=(6.5, 5))
for name, probabilities in task3_probabilities.items():
    precision, recall, _ = sklearn.metrics.precision_recall_curve(ytest, probabilities)
    ax.plot(recall, precision, label=name)
ax.axhline(np.mean(ytest), linestyle="--", color="gray", label="Spam prevalence")
ax.set(xlabel="Recall", ylabel="Precision", title="Test-set precision-recall curves", xlim=(0, 1), ylim=(0, 1.02))
ax.legend()
fig.tight_layout()
fig.savefig(task3_results / "precision_recall.png", dpi=180, bbox_inches="tight")
plt.show()

# Save the numerical evaluation alongside the plots for reproducibility.
with (task3_results / "metrics.txt").open("w", encoding="utf-8") as output:
    for name in task3_models:
        output.write(f"{name}\n")
        output.write(f"accuracy={sklearn.metrics.accuracy_score(ytest, task3_labels[name]):.8f}\n")
        output.write(f"mean_test_log_loss={er(ytest, Xtestz, task3_models[name]):.8f}\n")
        output.write("confusion_matrix_rows_true_columns_predicted:\n")
        output.write(str(sklearn.metrics.confusion_matrix(ytest, task3_labels[name], labels=[0, 1])) + "\n")
        output.write(sklearn.metrics.classification_report(ytest, task3_labels[name], labels=[0, 1], digits=4, zero_division=0))
        output.write("\n")

# Compare the signed coefficients by their average absolute magnitude.
importance = np.mean(np.abs(np.vstack([wz_gd, wz_sgd])), axis=0)
top = np.argsort(importance)[-12:]
top = top[np.argsort(importance[top])]
fig, ax = plt.subplots(figsize=(8, 6))
positions = np.arange(len(top))
height = 0.38
ax.barh(positions - height / 2, wz_gd[top], height, label="GD")
ax.barh(positions + height / 2, wz_sgd[top], height, label="SGD")
ax.set(yticks=positions, yticklabels=features[top], xlabel="Logistic-regression coefficient", title="Largest-magnitude feature weights")
ax.axvline(0, color="black", linewidth=0.7)
ax.legend()
fig.tight_layout()
fig.savefig(task3_results / "feature_weights.png", dpi=180, bbox_inches="tight")
plt.show()

weight_table = pd.DataFrame({"GD weight": wz_gd, "SGD weight": wz_sgd}, index=features)
weight_table["mean absolute weight"] = importance
weight_table.to_csv(task3_results / "feature_weights.csv")
print("Largest-magnitude features:")
print(weight_table.reindex(weight_table["mean absolute weight"].nlargest(12).index).round(4).to_string())
print("Smallest-magnitude features:")
print(weight_table.reindex(weight_table["mean absolute weight"].nsmallest(8).index).round(4).to_string())
