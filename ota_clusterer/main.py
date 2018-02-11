import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import ota_clusterer.webcrawler.Crawler as crawler
from ota_clusterer.word_embeddings.doc2vec import doc2vec
from ota_clusterer.dimensionality_reduction.tsne import tsne
from ota_clusterer.clusterer.affinity_propagation import affinity_propagation
from ota_clusterer.clusterer.kmeans import kmeans
from ota_clusterer.clusterer.kmedoid import kmedoid
from ota_clusterer.clusterer.dbscan import dbscan
from ota_clusterer.clusterer.agglomerative_clustering import agglomerative_clustering
from ota_clusterer import logger

logger = logger.get_logger()
logger.name = __name__

if __name__ == "__main__":

    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description="""
    
CLI for OTA Clusterer
     
    
                             \\
                              \\
                               \\\\
                                \\\\
                                 >\\/7
                             _.-(6'  \\
                            (=___._/` \\
                                 )  \ |
                                /   / |
                               /    > /
                              j    < _\\
                          _.-' :      ``.
                          \ r=._\        `.
                         <`\\_  \         .`-.
                          \ r-7  `-. ._  ' .  `\\
                           \`,      `-.`7  7)   )
                            \/         \|  \\'  / `-._
                                       ||    .'
Magic happens...                       \\\\  (
with ML <3                              >\  >
Sandro Cilurzo                        ,.-' >.'
                                    <.'_.''
                                      <'') """, formatter_class=formatter_class)

    parser.add_argument('-hostnames',
                        metavar='hostnames (urls) to crawl',
                        nargs='+',
                        help='hostnames to crawl')

    parser.add_argument('-crawled_data_directory',
                        metavar='where to store the crawled data',
                        help='directory path to store files, current directory is the default path')

    parser.add_argument('-urls_list',
                        metavar='file path tp url list',
                        help='path to url list')

    parser.add_argument('--crawl',
                        help='crawling given hostnames',
                        action='store_true',
                        dest='crawl')

    parser.add_argument('--crawl-list',
                        help='crawling a list of hostnames',
                        action='store_true',
                        dest='crawl_list')

    parser.add_argument('--new_doc2vec_model',
                        help='creates a new doc2vec model',
                        action='store_true',
                        dest='create_doc2vec_model')

    parser.add_argument('-data_input_directory',
                        metavar='previous stored crawling data',
                        help='directory where previous crawled data is stored')

    parser.add_argument('-models_output_directory',
                        metavar='where to store generated models (doc2vec and tsne)',
                        help='directory to store the models')

    parser.add_argument('-doc2vec_model_file_path',
                        metavar='doc2vec model to load',
                        help='loads doc2vec model from given file path')

    parser.add_argument('--new_tsne_model',
                        help='creates a new tsne model',
                        action='store_true',
                        dest='create_tsne_model')

    parser.add_argument('-tsne_file_name',
                        help='tsne model file name')

    parser.add_argument('--new_tsne_model_for_unseen_data',
                        help='creates a new tsne model for unseen data',
                        action='store_true',
                        dest='create_tsne_model_for_unseen_data')

    parser.add_argument('-unseen_hostnames',
                        metavar='hostnames of unseen data',
                        nargs='+',
                        help='hostnames of unseen data which should get processed')

    parser.add_argument('-model_language',
                        metavar='german or english',
                        help='language used in doc2vec model (german or english)')

    parser.add_argument('-clustering_output_directory',
                        metavar='path of directory to store clustering plots',
                        help='directory to store the clustering plots')

    parser.add_argument('-load_tsne_model',
                        metavar='load tsne model for clustering',
                        help='load tsne model for clustering')

    parser.add_argument('--affinity_propagation',
                        help='apply affinity_propagation clustering algorithm to given tsne model',
                        action='store_true',
                        dest='affinity_propagation')

    parser.add_argument('--kmeans',
                        help='apply k-means clustering algorithm to given tsne model',
                        action='store_true',
                        dest='kmeans')

    parser.add_argument('--kmedoid',
                        help='apply k-medoid clustering algorithm to given tsne model',
                        action='store_true',
                        dest='kmedoid')

    parser.add_argument('--dbscan',
                        help='apply DBSCAN clustering algorithm to given tsne model',
                        action='store_true',
                        dest='dbscan')

    parser.add_argument('--agglomerative_clustering',
                        help='apply agglomerative clustering algorithm to given tsne model',
                        action='store_true',
                        dest='agglomerative_clustering')

    args = parser.parse_args()

    if args.crawl:
        logger.info('Hostnames to crawl via CLI = ' + str(args.hostnames))
        logger.info('Store crawled data at:  ' + args.crawled_data_directory)
        crawler.crawl_given_urls(args.hostnames, args.crawled_data_directory)

    elif args.crawl_list:
        logger.info('Crawl list of hostnames via CLI from: ' + args.urls_list)
        logger.info('Store crawled data at: ' + args.output_directory)
        crawler.crawl_given_urls(args.urls_list, args.output_directory)

    elif args.create_doc2vec_model:
        logger.info('Create doc2vec model via CLI from following data: ' + args.data_input_directory +
                    ' and store data at ' + args.models_output_directory)

        doc2vec.create_new_doc2vec_model(documents_file_path=args.data_input_directory,
                                         save_to_directory=args.models_output_directory)

    elif args.create_tsne_model:
        logger.info('create tsne model from CLI from following doc2vec model: ' + args.doc2vec_model_file_path)
        tsne.create_new_doc2vec_tsne_model(doc2vec_model_file_path=args.doc2vec_model_file_path,
                                   output_directory_tsne_model=args.models_output_directory,
                                   tsne_file_name=args.tsne_file_name)

    elif args.create_tsne_model_for_unseen_data:
        logger.info(
            'create tsne model with unseen data based on following doc2vec model ' + args.doc2vec_model_file_path)
        tsne.create_new_doc2vec_tsne_model_for_unseen_data(doc2vec_model_file_path=args.doc2vec_model_file_path,
                                                           output_directory_tsne_model=args.doc2vec_model_file_path,
                                                           unseen_data=args.unseen_hostnames,
                                                           model_language=args.model_language,
                                                           tsne_file_name=args.tsne_file_name)

    elif args.affinity_propagation:
        logger.info('create affinity propagation clustering for the given tsne model' + args.load_tsne_model)
        affinity_propagation.create_affinity_propagation_cluster(doc2vec_model_file_path=args.doc2vec_model_file_path,
                                                                 tsne_model_file_path=args.load_tsne_model,
                                                                 model_language=args.model_language,
                                                                 save_to_directory=args.clustering_output_directory)

    elif args.kmeans:
        logger.info('create K-Means clustering for the given tsne model' + args.load_tsne_model)
        kmeans.create_kmeans_clustering(doc2vec_model_file_path=args.doc2vec_model_file_path,
                                        tsne_model_file_path=args.load_tsne_model,
                                        model_language=args.model_language,
                                        save_to_directory=args.clustering_output_directory)

    elif args.kmedoid:
        logger.info('create K-Medoid clustering for the given tsne model' + args.load_tsne_model)
        kmedoid.create_kmedoid_clustering(doc2vec_model_file_path=args.doc2vec_model_file_path,
                                          tsne_model_file_path=args.load_tsne_model)

    elif args.dbscan:
        logger.info('create DBSCAN clustering for the given tsne model' + args.load_tsne_model)
        dbscan.create_dbscan_clustering(doc2vec_model_file_path=args.doc2vec_model_file_path,
                                        tsne_model_file_path=args.load_tsne_model,
                                        model_language=args.model_language,
                                        save_to_directory=args.clustering_output_directory)

    elif args.agglomerative_clustering:
        logger.info('create agglomerative clustering for the given tsne model' + args.load_tsne_model)
        agglomerative_clustering.create_agglomerative_clustering(doc2vec_model_file_path=args.doc2vec_model_file_path,
                                                                 tsne_model_file_path=args.load_tsne_model,
                                                                 model_language=args.model_language,
                                                                 save_to_directory=args.clustering_output_directory)
