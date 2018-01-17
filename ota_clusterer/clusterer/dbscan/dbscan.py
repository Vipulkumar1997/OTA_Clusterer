#http://scikit-learn.org/stable/modules/clustering.html

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ota_clusterer import logger
from ota_clusterer.doc2vec import doc2vec
from ota_clusterer.clusterer.tsne import tsne


logger = logger.get_logger()


def create_dbscan_clustering(doc2vec_model, tsne_model):

    logger.info('Start creating DBSCAN Cluster...')
    data_point_labels = list(doc2vec_model.docvecs.doctags.keys())
    logger.info('Amount of Datapoints Labels = ' + str(len(data_point_labels)))
    logger.info('Length of the t-sne model = ' + str(len(tsne_model)))

    assert (len(tsne_model) == len(data_point_labels))

    # transform tsne-model
    tsne_model = StandardScaler().fit_transform(tsne_model)

    # DBSCAN Parameters
    eps = 0.35
    min_samples = 2
    logger.info('DBSCAN Parameters: eps = %s, min_samples= %d ' % (eps, min_samples))

    logger.info('Start fitting DBSCAN Algorithm')
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(tsne_model)
    logger.info('Sucessfully created DBSCAN Model...start creating visuallization')

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


    plt.figure(figsize=(16, 16))
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = tsne_model[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = tsne_model[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    # Annotate the data points
    for i, txt in zip(tsne_model, data_point_labels):
        plt.annotate(txt, (i[0], i[1]), xytext=(0, -8), textcoords="offset points", va="center", ha="left")

    plt.title('DBSCAN - Estimated number of clusters: %d' % n_clusters_)
    plt.suptitle('DBSCAN parameters = ' + 'eps=' + str(eps) + ', ' + 'min_sample=' + str(min_samples))
    plt.show()


def main():
    # example usage for create Agglomerative Clustering
    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    tsne_model = tsne.load_tsne_model('t-sne-cluster-doc2vec-german-11-Dez-2017-17:40:57.npy')
    create_dbscan_clustering(doc2vec_model, tsne_model)


if __name__ == "__main__":
    main()