from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import time
from ota_clusterer import logger
from ota_clusterer.doc2vec import doc2vec
from ota_clusterer.clusterer.tsne import tsne

'''
Reference: http://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py
'''

logger = logger.get_logger()


def create_agglomerative_clustering(doc2vec_model, tsne_model):
    logger.info("Start creating Agglomerative Cluster...")
    data_point_labels = list(doc2vec_model.docvecs.doctags.keys())
    logger.info('Amount of Datapoints Labels = ' + str(len(data_point_labels)))
    logger.info('Length of the t-sne model = ' + str(len(tsne_model)))

    assert (len(tsne_model) == len(data_point_labels))

    # calculate local connectivity
    knn_graph = kneighbors_graph(tsne_model, 30, include_self=False)

    for connectivity in (None, knn_graph):
        for n_clusters in (5, 10, 15, 20, 25, 30):
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

    plt.show()


def main():
    # example usage for create Agglomerative Clustering
    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    tsne_model = tsne.load_tsne_model('t-sne-cluster-doc2vec-german-11-Dez-2017-17:40:57.npy')
    create_agglomerative_clustering(doc2vec_model, tsne_model)


if __name__ == "__main__":
    main()


