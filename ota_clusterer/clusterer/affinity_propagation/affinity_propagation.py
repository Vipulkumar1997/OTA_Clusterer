#!/usr/bin/env python

__author__ = 'Sandro Cilurzo'

import time
import logging
from itertools import cycle
import sklearn.manifold
import sklearn.cluster
import matplotlib.pyplot as plt
from ota_clusterer import settings
from ota_clusterer.word_embeddings.doc2vec import doc2vec
from ota_clusterer.dimensionality_reduction.tsne import tsne

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def affinity_propagation_cluster(doc2vec_model, tsne_model, model_language, new_hostnames=None, save_to_directory=None):
    """creates an affinity propagation cluster and plots the result
    :param doc2vec_model: doc2vec model to infer document keys (document identifiers)
    :param tsne_model: tsne model to apply the affinity propagation algorithm
    :param model_language: language of the doc2vec model, will be added to the file name
    :param new_hostnames: list of new hostnames (new data) which were not included in doc2vec model
    :param save_to_directory: where to store the plot

    """

    logger.info("Start creating affinity propagation cluster...")
    data_point_labels = list(doc2vec_model.docvecs.doctags.keys())

    if new_hostnames is not None:
        for hostname in new_hostnames:
            data_point_labels.append(hostname)

    logger.info('Amount of Datapoints Labels = ' + str(len(data_point_labels)))

    assert (len(tsne_model) == len(data_point_labels))

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
        for i in fname_indices: fnames_cluster.append(data_point_labels[i])

        plt.plot(tsne_model[class_members, 0], tsne_model[class_members, 1], col + ".")
        plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markersize=20)

        # plt.annotate(data_point_labels[labels[k]], (cluster_center[0], cluster_center[1]), xytext=(0, -8),
        #        textcoords="offset points", va="center", ha="left")

        for x, fname in zip(tsne_model[class_members], fnames_cluster):
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linestyle='--', linewidth=1)
            plt.annotate(fname, (x[0], x[1]), xytext=(0, -8),
                         textcoords="offset points", va="center", ha="left")

    if save_to_directory is None:
        file_path = settings.DATA_DIR + "experiments/clusterer/affinity_propagation/"
    else:
        file_path = save_to_directory

    file_name = 'affinity_propagation_cluster-' + model_language + '-' + time.strftime("%d-%b-%Y-%X") + ".png"
    plt.savefig(file_path + file_name, facecolor="w", dpi=90)
    logger.info("saved " + file_name + " at " + file_path)


def create_affinity_propagation_cluster(doc2vec_model_file_path, tsne_model_file_path, model_language,
                                        save_to_directory, new_hostnames=None):
    """helper function to create affinity propagation clustering plot
    :param doc2vec_model_file_path: file path of doc2vec model
    :param tsne_model_file_path: file path of tsne model
    other parameters are explained in affinity_propagation_cluster function

    """

    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path=doc2vec_model_file_path)
    tsne_model = tsne.load_tsne_model(tsne_model_file_path=tsne_model_file_path)
    affinity_propagation_cluster(doc2vec_model, tsne_model, model_language, new_hostnames, save_to_directory)


def main():

    # example usage for creating an affinity propagation cluster
    doc2vec_model = doc2vec.load_existing_model(
        model_file_name='doc2vec-single_language_full-model-german-18-Feb-2018-22:31:27')

    tsne_model = tsne.load_tsne_model(
        model_file_name='t-sne-single_language_full-model-doc2vec-model-german-20-Feb-2018-08:56:12.npy')

    affinity_propagation_cluster(doc2vec_model=doc2vec_model,
                                 tsne_model=tsne_model,
                                 model_language='single_language_full_german')

    '''

    # example with affinity propagation cluster and new added data points
    doc2vec_model = doc2vec.load_existing_model(
        model_file_name='doc2vec-single_language_full-model-german-18-Feb-2018-22:31:27')
    tsne_model = tsne.load_tsne_model(
        model_file_name='t-sne-s-l-full-doc2vec-model-new-data-german-20-Feb-2018-09:58:50.npy')
    affinity_propagation_cluster(doc2vec_model,
                                 tsne_model,
                                 model_language='single-language_full-model-new-data-german',
                                 new_hostnames=['upkbs.ch',
                                                'curaneo.ch',
                                                'bscyb.ch',
                                                'scltigers.ch',
                                                'graubuenden.ch'])
    '''


if __name__ == "__main__":
    main()
