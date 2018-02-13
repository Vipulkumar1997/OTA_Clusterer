import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ota_clusterer import settings
from ota_clusterer import logger
from ota_clusterer.dimensionality_reduction.tsne import tsne
from ota_clusterer.word_embeddings.doc2vec import doc2vec

logger = logger.get_logger()
logger.name = __name__


def kmeans_clustering(doc2vec_model, tsne_model, model_language, k=3, new_hostnames=None, save_to_directory=None):
    """Creates K-Means clustering for given tsne model
    :param doc2vec_model: data point labels (keys) gets inferred from doc2vec model
    :param tsne_model: tsne model to apply clustering
    :param model_language: language of doc2vec model, gets added to the file name
    :param k: value to control how many clusters (k) should be generated
    :param new_hostnames: hostnames which where not included in doc2vec model while training (new data)
    :param save_to_directory: where to store the plot


    """
    logger.info("Start creating K-Means Clustering...")
    logger.info('Length of the t-sne model = ' + str(len(tsne_model)))

    data_point_labels = list(doc2vec_model.docvecs.doctags.keys())

    if new_hostnames is not None:
        for hostname in new_hostnames:
            data_point_labels.append(hostname)

    logger.info('Amount of Datapoints Labels = ' + str(len(data_point_labels)))

    assert (len(tsne_model) == len(data_point_labels))
    assert (k <= len(tsne_model))

    k = k
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
    for i, txt in zip(tsne_model, data_point_labels):
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

    if save_to_directory is None:
        file_path = settings.DATA_DIR + "experiments/kmeans/"
    else:
        file_path = save_to_directory

    file_name = 'kmeans_cluster-' + model_language + '-' + time.strftime("%d-%b-%Y-%X") + ".png"
    plt.savefig(file_path + file_name, facecolor="w", dpi=90)
    logger.info("saved " + file_name + "at " + file_path)

    plt.show()


def create_kmeans_clustering(doc2vec_model_file_path, tsne_model_file_path, model_language, k, save_to_directory):
    """helper function to create K-Means clustering plot
    :param doc2vec_model_file_path: file path of doc2vec model
    :param tsne_model_file_path: file path of tsne model
    other parameters are explained in dbscan_clustering function

    """

    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path=doc2vec_model_file_path)
    tsne_model = tsne.load_tsne_model(tsne_model_file_path=tsne_model_file_path)

    kmeans_clustering(doc2vec_model=doc2vec_model, tsne_model=tsne_model, model_language=model_language, k=k,
                      save_to_directory=save_to_directory)


def main():
    # example usage for create K-Means Clustering
    doc2vec_model = doc2vec.load_existing_model(model_file_name='doc2vec-model-german-11-Dec-2017-17:07:03')
    tsne_model = tsne.load_tsne_model(model_file_name='t-sne-cluster-doc2vec-german-11-Dez-2017-17:40:57.npy')
    kmeans_clustering(doc2vec_model, tsne_model, model_language='german', k=10)

    # experiment with K-Means and new added data points
    doc2vec_model = doc2vec.load_existing_model(model_file_name='doc2vec-model-german-11-Dec-2017-17:07:03')
    tsne_model = tsne.load_tsne_model(model_file_name='t-sne-cluster-unseen-data-doc2vec-german-18-Jan-2018-15:14:31.npy')
    kmeans_clustering(doc2vec_model, tsne_model, model_language='german', new_hostnames=['kickers.ch', 'pdgr.ch'], k=10)


if __name__ == "__main__":
    main()