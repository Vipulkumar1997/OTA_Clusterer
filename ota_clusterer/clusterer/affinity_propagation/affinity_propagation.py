import logging
import sklearn.manifold
import sklearn.cluster
import matplotlib.pyplot as plt
import time
from ota_clusterer import settings
from itertools import cycle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_affinity_propagation_cluster_doc2vec_plot(doc2vec_model, tsne_model, filename):

    logger.info("Start creating affinity propagation cluster...")

    fnames = list(doc2vec_model.docvecs.doctags.keys())
    affinity_propagation = sklearn.cluster.AffinityPropagation().fit(tsne_model)

    cluster_centers_indices = affinity_propagation.cluster_centers_indices_
    labels = affinity_propagation.labels_
    n_clusters_ = len(cluster_centers_indices)

    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")
    colors = cycle("bgrcmyk")

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k  # class_members ist array von boolschen werten, beschreibt cluster membership
        cluster_center = tsne_model[cluster_centers_indices[k]]

        fnames_cluster = []
        fname_indices = [i for i, x in enumerate(class_members) if x]
        for i in fname_indices: fnames_cluster.append(fnames[i])

        plt.plot(tsne_model[class_members, 0], tsne_model[class_members, 1], col + ".")
        plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markersize=20)

        # plt.annotate(fnames[labels[k]], (cluster_center[0], cluster_center[1]), xytext=(0, -8),
        #        textcoords="offset points", va="center", ha="left")

        for x, fname in zip(tsne_model[class_members], fnames_cluster):
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linestyle='--', linewidth=1)
            plt.annotate(fname, (x[0], x[1]), xytext=(0, -8),
                         textcoords="offset points", va="center", ha="left")

    file_path = settings.DATA_DIR + "experiments/affinity_propagation/"
    file_name = 'affinity_propagation_cluster-' + filename + "-" + time.strftime("%d-%b-%Y-%X") + ".png"
    plt.savefig(file_path + file_name, facecolor="w", dpi=90)
    logger.info("saved " + filename + "at " + file_path)