import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
from ota_clusterer.webcrawler.Crawler import Crawler
from ota_clusterer.word_embeddings.doc2vec import doc2vec
from ota_clusterer.dimensionality_reduction.tsne.tsne import create_new_doc2vec_tsne_model
from ota_clusterer.dimensionality_reduction.tsne.tsne import create_new_doc2vec_tsne_model_for_unseen_data
from ota_clusterer import logger

logger = logger.get_logger()
logger.name = __name__


def crawl_hostnames(hostnames, directory_to_save_results):
    crawler = Crawler()
    logger.info('crawl following hostnames: ' + str(hostnames))
    crawler.crawl_hostnames(hostnames, directory_to_save_results)


def crawl_list_of_hostnames(urls_list_file_path, directory_to_save_results):
    crawler = Crawler()
    hostnames = crawler.get_hostnames(urls_list_file_path)
    crawler.crawl_hostnames(hostnames, directory_to_save_results)


def create_new_doc2vec_model(data_path, directory_to_save_model):
    doc2vec.create_new_doc2vec_model(documents_file_path=data_path, save_to_directory=directory_to_save_model)


def create_clustering_model(tsne_model, clustering_algorithm, all=False):
    if all is True:
        print('affinity_propagation()')

        # TODO affinity_propagation
        # TODO agglomerative_clustering
        # TODO dbscan
        # TODO kmeans
        # TODO kmedoid

    elif clustering_algorithm == 'affinity':
        # TODO
        print('affinity')

    elif clustering_algorithm == 'agglomerative':
        print('affinity')

    elif clustering_algorithm == 'dbscan':
        print('affinity')

    elif clustering_algorithm == 'kmeans':
        print('affinity')

    elif clustering_algorithm == 'kmedoid':
        print('affinity')


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
                        nargs='+',
                        help='hostnames to crawl')

    parser.add_argument('-crawled_data_directory',
                        help='directory path to store files, current directory is the default path')

    parser.add_argument('-urls_list',
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
                        help='directory where previous crawled data is stored')

    parser.add_argument('-models_output_directory',
                        help='directory to store the models')

    parser.add_argument('-doc2vec_model_file_path',
                        help='loads doc2vec model from given file path')

    parser.add_argument('--new_tsne_model',
                        help='creates a new tsne model with given name',
                        action='store_true',
                        dest='create_tsne_model')

    parser.add_argument('-tsne_file_name',
                        help='tsne model file name')

    parser.add_argument('--new_tsne_model_for_unseen_data',
                        help='creates a new tsne model for unseen data',
                        action='store_true',
                        dest='create_tsne_model_unseen_data')

    args = parser.parse_args()

    if args.crawl:
        logger.info('Hostnames to crawl via CLI = ' + str(args.hostnames))
        logger.info('Store crawled data at:  ' + args.crawled_data_directory)
        crawl_hostnames(args.hostnames, args.crawled_data_directory)

    elif args.crawl_list:
        logger.info('Crawl list of hostnames via CLI from: ' + args.urls_list)
        logger.info('Store crawled data at: ' + args.output_directory)
        crawl_list_of_hostnames(args.urls_list, args.output_directory)

    elif args.create_doc2vec_model:
        logger.info('Create doc2vec model via CLI from following data: ' + args.data_input_directory +
                    ' and store data at ' + args.models_output_directory)
        create_new_doc2vec_model(data_path=args.data_input_directory,
                                 directory_to_save_model=args.models_output_directory)

    elif args.create_tsne_model:
        logger.info('create tsne model from CLI from following doc2vec model: ' + args.doc2vec_model_file_path)
        create_new_doc2vec_tsne_model(doc2vec_model_file_path=args.doc2vec_model_file_path,
                                      output_directory_tsne_model=args.models_output_directory,
                                      tsne_file_name=args.tsne_file_name)
