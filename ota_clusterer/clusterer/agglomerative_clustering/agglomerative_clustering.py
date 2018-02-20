#!/usr/bin/env python

__author__ = 'Sandro Cilurzo'

import time
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from ota_clusterer import settings
from ota_clusterer import logger
from ota_clusterer.dimensionality_reduction.tsne import tsne
from ota_clusterer.word_embeddings.doc2vec import doc2vec

logger = logger.get_logger()
logger.name = __name__


def agglomerative_clustering(doc2vec_model, tsne_model, numbers_of_clusters, model_language, new_hostnames=None,
                             save_to_directory=None):
    """applies agglomerative clustering algorithm to given tsne model
    :param doc2vec_model: infer documents labels (keys) from doc2vec model
    :param tsne_model: tsne model to apply algorithm to
    :param numbers_of_clusters: how many clusters should get build
    :param model_language: doc2vec model language, gets added to the plot file name
    :param new_hostnames: hostnames which where not included in doc2vec model while training (new data)
    :param save_to_directory: where to store the plot
    Reference: http://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py

    """

    logger.info("Start creating Agglomerative Cluster...")
    data_point_labels = list(doc2vec_model.docvecs.doctags.keys())

    if new_hostnames is not None:
        for hostname in new_hostnames:
            data_point_labels.append(hostname)

    logger.info('Amount of Datapoints Labels = ' + str(len(data_point_labels)))
    logger.info('Length of the t-sne model = ' + str(len(tsne_model)))

    assert (len(tsne_model) == len(data_point_labels))

    # calculate local connectivity
    knn_graph = kneighbors_graph(tsne_model, 30, include_self=False)

    # example: (5, 10, 15, 20, 25, 30)
    numbers_of_clusters = tuple(numbers_of_clusters)

    for connectivity in (None, knn_graph):
        for n_clusters in numbers_of_clusters:
            plt.figure(figsize=(40, 15))
            for index, linkage in enumerate(('average', 'complete', 'ward')):
                plt.subplot(1, 3, index + 1)
                model = AgglomerativeClustering(linkage=linkage,
                                                connectivity=connectivity,
                                                n_clusters=n_clusters)
                t0 = time.time()
                model.fit(tsne_model)
                elapsed_time = time.time() - t0
                plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=model.labels_,
                            cmap=plt.cm.spectral)

                # Annotate the data points
                for i, txt in zip(tsne_model, data_point_labels):
                    plt.annotate(txt, (i[0], i[1]), xytext=(0, -8), textcoords="offset points", va="center", ha="left")

                plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
                          fontdict=dict(verticalalignment='top'))
                plt.axis('equal')
                plt.axis('off')

                plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                    left=0, right=1)
                plt.suptitle('n_cluster=%i, connectivity=%r' %
                             (n_clusters, connectivity is not None), size=17)

            if save_to_directory is None:
                file_path = settings.DATA_DIR + "experiments/clusterer/agglomerative_clustering/"
            else:
                file_path = save_to_directory

            file_name = 'agglomerative_clustering-' + model_language + '-' + time.strftime("%d-%b-%Y-%X") + ".png"
            plt.savefig(file_path + file_name, facecolor="w", dpi=90)
            logger.info("saved " + file_name + " at " + file_path)

    plt.show()


def create_agglomerative_clustering(doc2vec_model_file_path, tsne_model_file_path, numbers_of_clusters, model_language,
                                    save_to_directory, new_hostnames=None):
    """helper function to create agglomerative clustering plot
    :param doc2vec_model_file_path: file path of doc2vec model
    :param tsne_model_file_path: file path of tsne model
    other parameters are explained in agglomerative_clustering function

    """

    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path=doc2vec_model_file_path)
    tsne_model = tsne.load_tsne_model(tsne_model_file_path=tsne_model_file_path)

    agglomerative_clustering(doc2vec_model=doc2vec_model,
                             tsne_model=tsne_model,
                             numbers_of_clusters=numbers_of_clusters,
                             model_language=model_language,
                             new_hostnames=new_hostnames,
                             save_to_directory=save_to_directory)


def main():
    # example usage for create Agglomerative Clustering
    doc2vec_model = doc2vec.load_existing_model(
        model_file_name='doc2vec-single_language_full-model-german-18-Feb-2018-22:31:27')

    tsne_model = tsne.load_tsne_model(
        model_file_name='t-sne-single_language_full-model-doc2vec-model-german-20-Feb-2018-08:56:12.npy')

    agglomerative_clustering(doc2vec_model,
                             tsne_model,
                             numbers_of_clusters=[5, 10, 15, 20, 25, 30],
                             model_language='single-language_full_model-german')

    '''

    # example for Agglomerative Clustering with new data
    doc2vec_model = doc2vec.load_existing_model(
        model_file_name='doc2vec-single_language_70_model-german-18-Feb-2018-18:53:39')

    tsne_model = tsne.load_tsne_model(
        model_file_name='t-sne-s-l-70-doc2vec-model-new-data-german-20-Feb-2018-09:47:02.npy')

    agglomerative_clustering(doc2vec_model,
                             tsne_model,
                             model_language='70-s-l-model-new-data-german',
                             numbers_of_clusters=[5, 10, 15, 20, 25, 30],
                             new_hostnames=['familotel.com',
                                            'regenbogenurlaub.de',
                                            'lakers.ch',
                                            'swisshotels.com',
                                            'ebookers.ch'])

    '''


if __name__ == "__main__":
    main()
