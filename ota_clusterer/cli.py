#!/usr/bin/env python

__author__ = 'Sandro Cilurzo'

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

    input_output_parameters_cli = parser.add_argument_group('# data input and output parameters for multiple functions')
    generic_function_parameters_cli = parser.add_argument_group('generic (shared) functions parameters')

    crawling_cli = parser.add_argument_group('# crawler function optioners and parameters')

    doc2vec_cli = parser.add_argument_group('# doc2vec function options and parameters')

    tsne_cli = parser.add_argument_group('# tsne function options and parameters')

    affinity_propagation_cli = parser.add_argument_group('# Affinity Propagation function and parameters')
    kmeans_cli = parser.add_argument_group('# K-Means function and parameters')
    kmedoid_cli = parser.add_argument_group('# K-Medoid function and parameters')
    dbscan_cli = parser.add_argument_group('# DBSCAN function and parameters')
    agglomerative_clustering_cli = parser.add_argument_group('# Agglomerative Clustering function and parameters')

    input_output_parameters_cli.add_argument('-crawled_dir',
                                             help='directory path to store(d) crawled data')

    input_output_parameters_cli.add_argument('-models_dir',
                                             help='directory to store(d) the models')

    input_output_parameters_cli.add_argument('-load_doc2vec_model',
                                             help='loads doc2vec model from given file path')

    input_output_parameters_cli.add_argument('-model_language',
                                             help='language used in doc2vec model (german or english)')

    input_output_parameters_cli.add_argument('-load_tsne_model',
                                             help='load tsne model for clustering')

    input_output_parameters_cli.add_argument('-clustering_dir',
                                             help='directory to store the clustering plots')

    crawling_cli.add_argument('-hostnames',
                              nargs='+',
                              help='hostnames to crawl')

    crawling_cli.add_argument('-urls_list',
                              help='path to url list')

    crawling_cli.add_argument('--crawl',
                              help='crawling given hostnames (required params: -hostnames, -crawled_dir)',
                              action='store_true',
                              dest='crawl')

    crawling_cli.add_argument('--crawl-list',
                              help='crawling a list of hostnames (required params: -urls_list, -crawled_dir)',
                              action='store_true',
                              dest='crawl_list')

    doc2vec_cli.add_argument('--doc2vec_model',
                             help='creates a new doc2vec model (required params: -crawled_dir, -models_dir)',
                             action='store_true',
                             dest='create_doc2vec_model')

    tsne_cli.add_argument('-tsne_file_name',
                          help='tsne model file name')

    tsne_cli.add_argument('-new_hostnames',
                          nargs='+',
                          help='hostnames of new data which should get processed')



    tsne_cli.add_argument('--tsne_model',
                          help='creates a new tsne model (required params: -load_doc2vec_model, -models_dir, '
                               '-tsne_file_name)',
                          action='store_true',
                          dest='create_tsne_model')

    tsne_cli.add_argument('--tsne_model_extended',
                          help='creates a new tsne model for unseen data (required params: -load_doc2vec_model, '
                               '-crawled_dir, -models_dir, -new_hostnames, -model_language, -tsne_file_name)',
                          action='store_true',
                          dest='create_tsne_model_for_new_data')

    affinity_propagation_cli.add_argument('--affinity_propagation',
                                          help='affinity propagation clustering over given tsne model (required '
                                               'params: '
                                               '-load_doc2vec_model, -load_tsne_model, -model_language, '
                                               '-clustering_dir, -new_hostnames (optional))',
                                          action='store_true',
                                          dest='affinity_propagation')

    kmeans_cli.add_argument('-k',
                            type=int,
                            help='Amount of clusters (k) in K-Means, default value is 3')

    kmeans_cli.add_argument('--kmeans',
                            help='k-means clustering algorithm to given tsne model (required params: '
                                 '-k, -load_doc2vec_model, -load_tsne_model, -model_language, -clustering_dir, '
                                 '-new_hostnames (optional))',
                            action='store_true',
                            dest='kmeans')

    kmedoid_cli.add_argument('-medoids',
                             type=int,
                             nargs='+',
                             help='Start Medoids, space seperated medoid points example: 0 5 10 20')

    kmedoid_cli.add_argument('--kmedoid',
                             help='k-medoid clustering algorithm to given tsne model (required params: '
                                  '-medoids, -load_doc2vec_model, -load_tsne_model, -new_hostnames (optional))',
                             action='store_true',
                             dest='kmedoid')

    dbscan_cli.add_argument('-eps',
                            type=float,
                            help='Epsilon value to define neighbourhood size')

    dbscan_cli.add_argument('-min_samples',
                            type=int,
                            help='Minimum samples in a cluster')

    dbscan_cli.add_argument('--dbscan',
                            help='DBSCAN clustering algorithm to given tsne model(required params: '
                                 '-eps, -min_samples, -load_doc2vec_model, -load_tsne_model, -model_language, '
                                 '-new_hostnames (optional) -clustering_dir)',
                            action='store_true',
                            dest='dbscan')

    agglomerative_clustering_cli.add_argument('-cluster_nr',
                                              type=int,
                                              nargs='+',
                                              help='Space seperated numbers of clusters: 5 10 15 20')

    agglomerative_clustering_cli.add_argument('--agglomerative_clustering',
                                              help='agglomerative clustering algorithm to given tsne model (required '
                                                   'params: '
                                                   '-cluster_nr, -load_doc2vec_model, -load_tsne_model, -model_language, '
                                                   '-clustering_dir, -new_hostnames (optional))',
                                              action='store_true',
                                              dest='agglomerative_clustering')

    args = parser.parse_args()

    if args.crawl:
        logger.info('Hostnames to crawl via CLI = ' + str(args.hostnames))
        logger.info('Store crawled data at:  ' + args.crawled_dir)
        crawler.crawl_given_hostnames(args.hostnames, args.crawled_dir)

    elif args.crawl_list:
        logger.info('Crawl list of hostnames via CLI from: ' + args.urls_list)
        logger.info('Store crawled data at: ' + args.crawled_dir)
        crawler.crawl_list_of_hostnames(args.urls_list, args.crawled_dir)

    elif args.create_doc2vec_model:
        logger.info('Create doc2vec model via CLI from following data: ' + args.crawled_dir +
                    ' and store data at ' + args.models_dir)

        doc2vec.create_new_doc2vec_model(documents_file_path=args.crawled_dir,
                                         save_to_directory=args.models_dir)

    elif args.create_tsne_model:
        logger.info('create tsne model from CLI from following doc2vec model: ' + args.load_doc2vec_model)
        tsne.create_tsne_for_doc2vec_model(doc2vec_model_file_path=args.load_doc2vec_model,
                                           output_directory_tsne_model=args.models_dir,
                                           tsne_file_name=args.tsne_file_name)

    elif args.create_tsne_model_for_new_data:
        logger.info(
            'create tsne model with unseen data based on following doc2vec model ' + args.load_doc2vec_model)
        tsne.create_tsne_for_doc2vec_model_with_new_documents(doc2vec_model_file_path=args.load_doc2vec_model,
                                                              new_documents=args.new_hostnames,
                                                              model_language=args.model_language,
                                                              output_directory=args.models_dir,
                                                              tsne_file_name=args.tsne_file_name,
                                                              documents_file_path=args.crawled_dir)

    elif args.affinity_propagation:
        logger.info('create affinity propagation clustering for the given tsne model' + args.load_tsne_model)
        affinity_propagation.create_affinity_propagation_cluster(doc2vec_model_file_path=args.load_doc2vec_model,
                                                                 tsne_model_file_path=args.load_tsne_model,
                                                                 model_language=args.model_language,
                                                                 save_to_directory=args.clustering_dir,
                                                                 new_hostnames=args.new_hostnames)

    elif args.kmeans:
        logger.info('create K-Means clustering for the given tsne model' + args.load_tsne_model)
        kmeans.create_kmeans_clustering(doc2vec_model_file_path=args.load_doc2vec_model,
                                        tsne_model_file_path=args.load_tsne_model,
                                        model_language=args.model_language,
                                        k=args.k,
                                        save_to_directory=args.clustering_dir,
                                        new_hostnames=args.new_hostnames)

    elif args.kmedoid:
        logger.info('create K-Medoid clustering for the given tsne model' + args.load_tsne_model)
        kmedoid.create_kmedoid_clustering(doc2vec_model_file_path=args.load_doc2vec_model,
                                          tsne_model_file_path=args.load_tsne_model,
                                          start_medoids=args.medoids,
                                          new_hostnames=args.new_hostnames)

    elif args.dbscan:
        logger.info('create DBSCAN clustering for the given tsne model' + args.load_tsne_model)
        dbscan.create_dbscan_clustering(doc2vec_model_file_path=args.load_doc2vec_model,
                                        tsne_model_file_path=args.load_tsne_model,
                                        eps=args.eps,
                                        min_samples=args.min_samples,
                                        model_language=args.model_language,
                                        save_to_directory=args.clustering_dir,
                                        new_hostnames=args.new_hostnames)

    elif args.agglomerative_clustering:
        logger.info('create agglomerative clustering for the given tsne model' + args.load_tsne_model)
        agglomerative_clustering.create_agglomerative_clustering(doc2vec_model_file_path=args.load_doc2vec_model,
                                                                 tsne_model_file_path=args.load_tsne_model,
                                                                 numbers_of_clusters=args.cluster_nr,
                                                                 model_language=args.model_language,
                                                                 save_to_directory=args.clustering_dir,
                                                                 new_hostnames=args.new_hostnames)
