"""!
@brief Examples of usage and demonstration of abilities of K-Medoids algorithm in cluster analysis.
@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2018
@copyright GNU Public License
@cond GNU_PUBLIC_LICENSE
    PyClustering is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    PyClustering is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond
"""

from pyclustering.cluster.kmedoids import kmedoids;
from pyclustering.utils import timedcall;

from ota_clusterer import logger
from ota_clusterer.clusterer.kmedoid.ClusterVisualizer import ClusterVisualizer
from ota_clusterer.dimensionality_reduction.tsne import tsne
from ota_clusterer.word_embeddings.doc2vec import doc2vec

logger = logger.get_logger()


def kmedoid_clustering(doc2vec_model, tsne_model):
    logger.info("Start creating K-Medoid Cluster...")
    data_point_labels = list(doc2vec_model.docvecs.doctags.keys())
    logger.info('Amount of Datapoints Labels = ' + str(len(data_point_labels)))
    logger.info('Length of the t-sne model = ' + str(len(tsne_model)))

    assert (len(tsne_model) == len(data_point_labels))

    start_medoids = [0, 5, 10, 15, 20]
    logger.info('Number of Medoids = %s' % len(start_medoids))

    logger.info('Start creating K-Medoid Model...')
    tolerance = 0.2
    kmedoids_instance = kmedoids(tsne_model, start_medoids, tolerance);
    (ticks, result) = timedcall(kmedoids_instance.process)
    
    clusters = kmedoids_instance.get_clusters();
    medoids = kmedoids_instance.get_medoids();
    print("Sample: ",  "\t\tExecution time: ", ticks, "\n");

    cluster_visualizer = ClusterVisualizer(1, data=tsne_model, labels=data_point_labels);
    cluster_visualizer.append_clusters(clusters, tsne_model, 0);

    # TODO delete comment
    # visualizer.append_cluster([ tsne_model[index] for index in start_medoids ], marker = 'x', markersize = 10, color='red');

    cluster_visualizer.append_cluster(medoids, marker = '*', markersize = 12, color='red')
    cluster_visualizer.show(k = len(start_medoids), tolerance=tolerance);


def create_kmedoid_clustering(doc2vec_model_file_path, tsne_model_file_path):
    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path=doc2vec_model_file_path)
    tsne_model = tsne.load_tsne_model(tsne_model_file_path=tsne_model_file_path)
    kmedoid_clustering(doc2vec_model, tsne_model)


def main():
    # example usage for create Agglomerative Clustering
    doc2vec_model = doc2vec.load_existing_model(model_file_name='doc2vec-model-german-11-Dec-2017-17:07:03')
    tsne_model = tsne.load_tsne_model(model_file_name='t-sne-cluster-doc2vec-german-11-Dez-2017-17:40:57.npy')
    kmedoid_clustering(doc2vec_model, tsne_model)


if __name__ == "__main__":
    main()
