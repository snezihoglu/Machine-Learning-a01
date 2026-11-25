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
# # 3 SVD and k-means

# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %load_ext autoreload
# %autoreload 2
from a03_helper import *

# %%
# Cluster the normalized climate data into 5 clusters using k-means and store
# the vector giving the cluster labels for each location.
X_clusters = KMeans(5).fit(X).labels_

# %% [markdown]
# ## 3a

# %%
# Plot the results to the map: use the cluster labels to give the color to each
# point.
plot_xy(lon, lat, X_clusters)

# %% [markdown]
# ## 3b

# %%
# YOUR PART HERE

# %% [markdown]
# ## 3c

# %%
# Compute the PCA scores, store in S_PCA (of shape N x Z)
Z = 2
# YOUR PART HERE

# %%
# cluster and visualize
S_PCA_clusters = KMeans(5).fit(S_PCA).labels_
# match clusters as well as possible (try without)
S_PCA_clusters = match_categories(X_clusters, S_PCA_clusters)
nextplot()
axs = plt.gcf().subplots(1, 2)
plot_xy(lon, lat, X_clusters, axis=axs[0])
axs[0].set_title("Original data")
plot_xy(lon, lat, S_PCA_clusters, axis=axs[1])
axs[1].set_title(f"PCA $(Z={Z}$)")
plt.show()
