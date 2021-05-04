import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

def silhouette(vectors, cluster_labels, metric='euclidean',
               spacing_coeff=0.02, fig=None, ax=None, figsize=None, show=True,
               save_path=None):
    '''
    Silhouette plot of the clustering of `vectors` into `cluster_labels`.
    '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    n_clusters = np.sum(np.unique(cluster_labels) >= 0)
    # Compute the silhouette scores for each sample
    sample_s_values = silhouette_samples(vectors, cluster_labels,
                                         metric=metric)
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    y_spacing = int(len(vectors) * spacing_coeff)
    ax.set_ylim([0, len(vectors) + (n_clusters + 1) * y_spacing])
    y_lower = y_spacing
    for i in range(n_clusters):
        # Can have cluster_label == -1 for cluster-less counties, belonging to
        # noise, but their silhouette won't be plotted (although taken into
        # account in silhouette_score).
        color = cm.Dark2(float(i) / (n_clusters-1))
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        cluster_s_values = sample_s_values[cluster_labels == i]
        cluster_s_values.sort()
        size_cluster = cluster_s_values.shape[0]
        y_upper = y_lower + size_cluster
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_s_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        # Label the silhouette plots with their cluster numbers in the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i+1))
        # Compute the new y_lower for next plot
        y_lower = y_upper + y_spacing

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg_noise = silhouette_score(vectors, cluster_labels,
                                            metric=metric)
    noise_mask = cluster_labels > -1
    silhouette_avg_wo_noise = silhouette_score(vectors[noise_mask, :],
                                               cluster_labels[noise_mask],
                                               metric=metric)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is",
          f"{silhouette_avg_noise}, and considering noise as singletons: ",
          silhouette_avg_wo_noise)
    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg_noise, color="red", linestyle=":")
    ax.axvline(x=silhouette_avg_wo_noise, color="red", linestyle="--")
    # Clear the yaxis labels / ticks
    ax.set_yticks([])
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        fig.show()
    return fig, ax
