import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from ota_clusterer import logger
from ota_clusterer.dimensionality_reduction.tsne import tsne
from ota_clusterer.word_embeddings.doc2vec import doc2vec

logger = logger.get_logger()
logger.name = __name__


def create_kmeans_clustering(doc2vec_model, tsne_model):
    logger.info("Start creating K-Means Clustering...")
    logger.info('Length of the t-sne model = ' + str(len(tsne_model)))

    fnames = list(doc2vec_model.docvecs.doctags.keys())
    fnames.append('fckickers.ch')
    fnames.append('pdgr.ch')
    logger.info('Amount of Datapoints Labels = ' + str(len(fnames)))

    assert (len(tsne_model) == len(fnames))

    k = 10
    random_state = 0
    logger.info('K-Means Parameters: k = %s, random_state= %d ' % (k, random_state))
    logger.info('Start training K-Means Model...')
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(tsne_model)
    logger.info('K-Means model sucessfully built...start with visualization')

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = tsne_model[:, 0].min() - 1, tsne_model[:, 0].max() + 1
    y_min, y_max = tsne_model[:, 1].min() - 1, tsne_model[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    logger.info('start creating color plot...')
    plt.figure(figsize=(16, 16))

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(tsne_model[:, 0], tsne_model[:, 1], 'k.', markersize=2)

    # Annotate the data points
    for i, txt in zip(tsne_model, fnames):
        plt.annotate(txt, (i[0], i[1]), xytext=(0, -8), textcoords="offset points", va="center", ha="left")

    logger.info('The centroids are getted plotted as white x...')
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    plt.title('K-Means clustering with K=%d over the t-sne reduced data' % k)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def main():
    # example usage for create K-Means Clustering
    # doc2vec_model = doc2vec.load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    # tsne_model = tsne.load_tsne_model('t-sne-cluster-doc2vec-german-11-Dez-2017-17:40:57.npy')
    # create_kmeans_clustering(doc2vec_model, tsne_model)

    # experiment with K-Means and new added data points
    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    tsne_model = tsne.load_tsne_model('t-sne-cluster-unseen-data-doc2vec-german-18-Jan-2018-15:14:31.npy')
    create_kmeans_clustering(doc2vec_model, tsne_model)


if __name__ == "__main__":
    main()