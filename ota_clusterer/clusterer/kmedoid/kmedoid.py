from pyclustering.cluster.kmedoids import kmedoids;
from pyclustering.utils import timedcall;
from ota_clusterer import logger
from ota_clusterer.clusterer.kmedoid.ClusterVisualizer import ClusterVisualizer
from ota_clusterer.dimensionality_reduction.tsne import tsne
from ota_clusterer.word_embeddings.doc2vec import doc2vec

logger = logger.get_logger()


def kmedoid_clustering(doc2vec_model, tsne_model, start_medoids, new_hostnames=None):
    """creates K-Medoid clustering for given tsne model
    :param doc2vec_model: doc2vec model to infer data point labels (keys)
    :param tsne_model: tsne model to apply clustering
    :param start_medoids: medoids which be used as startin point
    :param new_hostnames: hostnames which where not included in doc2vec model while training (new data)

    """

    logger.info("Start creating K-Medoid Cluster...")
    data_point_labels = list(doc2vec_model.docvecs.doctags.keys())

    if new_hostnames is not None:
        for hostname in new_hostnames:
            data_point_labels.append(hostname)

    logger.info('Amount of Datapoints Labels = ' + str(len(data_point_labels)))
    logger.info('Length of the t-sne model = ' + str(len(tsne_model)))

    assert (len(tsne_model) == len(data_point_labels))

    # Example: start_medoids = [0, 5, 10, 15, 20]
    start_medoids = start_medoids
    logger.info('Number of Medoids = %s' % len(start_medoids))
    logger.info('Given Medoids = %s' % str(start_medoids))

    logger.info('Start creating K-Medoid Model...')
    tolerance = 0.2
    kmedoids_instance = kmedoids(tsne_model, start_medoids, tolerance);
    (ticks, result) = timedcall(kmedoids_instance.process)
    
    clusters = kmedoids_instance.get_clusters();
    medoids = kmedoids_instance.get_medoids();
    print("Sample: ",  "\t\tExecution time: ", ticks, "\n");

    cluster_visualizer = ClusterVisualizer(1, data=tsne_model, labels=data_point_labels);
    cluster_visualizer.append_clusters(clusters, tsne_model, 0);

    cluster_visualizer.append_cluster(medoids, marker = '*', markersize = 12, color='red')
    cluster_visualizer.show(k=len(start_medoids), tolerance=tolerance);


def create_kmedoid_clustering(doc2vec_model_file_path, tsne_model_file_path, start_medoids, new_hostnames=None):
    """helper function to create K-Medoid clustering plot
    :param doc2vec_model_file_path: file path of doc2vec model
    :param tsne_model_file_path: file path of tsne model
    other parameters are explained in dbscan_clustering function

    """

    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path=doc2vec_model_file_path)
    tsne_model = tsne.load_tsne_model(tsne_model_file_path=tsne_model_file_path)
    kmedoid_clustering(doc2vec_model, tsne_model, start_medoids, new_hostnames=new_hostnames)


def main():
    # example usage for create K-Medoid Clustering
    doc2vec_model = doc2vec.load_existing_model(model_file_name='doc2vec-model-german-11-Dec-2017-17:07:03')
    tsne_model = tsne.load_tsne_model(model_file_name='t-sne-cluster-doc2vec-german-11-Dez-2017-17:40:57.npy')
    kmedoid_clustering(doc2vec_model, tsne_model, start_medoids=[0, 5, 10, 15, 20])


if __name__ == "__main__":
    main()
