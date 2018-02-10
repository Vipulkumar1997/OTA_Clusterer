import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
from ota_clusterer.webcrawler.Crawler import Crawler
from ota_clusterer.word_embeddings.doc2vec import doc2vec
from ota_clusterer.clusterer.tsne import tsne
from ota_clusterer.clusterer.affinity_propagation import affinity_propagation
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
    doc2vec_model_english, doc2vec_model_german = doc2vec.create_new_doc2vec_model(documents_file_path=data_path,
                                                                                   save_to_directory=directory_to_save_model)


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

    parser = argparse.ArgumentParser(description='CLI for OTA Clusterer')

    parser.add_argument('--hostnames', nargs='+', help='hostnames to crawl')
    parser.add_argument('--crawled_data_directory',
                        help='directory path to store files, current directory is the default path')
    parser.add_argument('--urls_list', help='path to url list')
    parser.add_argument('--crawl', help='crawling given hostnames', action='store_true', dest='crawl')
    parser.add_argument('--crawl-list', help='crawling a list of hostnames', action='store_true', dest='crawl_list')

    parser.add_argument('--new_doc2vec_model', help='creates a new doc2vec model', action='store_true',
                        dest='create_doc2vec_model')
    parser.add_argument('--data_input_directory', help='directory where crawled data is stored')
    parser.add_argument('--model_output_directory', help='directory to store the doc2vec model')
    parser.add_argument('--load_doc2vec_model_from_path', help='loads doc2vec model from file path')

    parser.add_argument('--new_tsne_model', help='creates a new tsne model', action='store_true', dest='tsne')

    args = parser.parse_args()
    if args.crawl:
        logger.info('Hostnames from CLI = ' + str(args.hostnames))
        logger.info('File Path from CLI = ' + args.crawled_data_directory)
        crawl_hostnames(args.hostnames, args.crawled_data_directory)

    elif args.crawl_list:
        crawl_list_of_hostnames(args.urls_list, args.output_directory)

    elif args.create_doc2vec_model:
        logger.info('input data directory = '+ str(args.data_input_directory))
        logger.info('Output data directory = '+ str(args.model_output_directory))
        create_new_doc2vec_model(data_path=args.data_input_directory,
                                 directory_to_save_model=args.model_output_directory)

