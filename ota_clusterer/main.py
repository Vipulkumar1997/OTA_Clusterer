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

def crawl_hostnames(hostnames, directory_to_save_results):
    crawler = Crawler()
    logger.info('crawl following hostnames: ' + str(hostnames))
    crawler.crawl_hostnames(hostnames, directory_to_save_results)

def crawl_list_of_hostnames(hostnames_file_path, directory_to_save_results):
    crawler = Crawler()
    hostnames = crawler.get_hostnames(hostnames_file_path)
    crawler.crawl_hostnames(hostnames, directory_to_save_results)

def create_new_word_embeddings_model(directory_path):
    doc2vec_model_english, doc2vec_model_german = doc2vec.create_new_doc2vec_model()

    # TODO safe doc2vec_model_in_directory
    # TODO safe tsne_model
    return doc2vec_model_english, doc2vec_model_german

def create_clustering_model(tsne_model,  clustering_algorithm, all=False):
    if all == True:
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
    parser.add_argument('hostnames', metavar='N', nargs='+', help='hostnames to crawl')
    parser.add_argument('output_directory', metavar='output directory', help='directory path to store files, current directory is the default path')
    parser.add_argument('--crawl', help='crawling given hostnames', action='store_true', dest='crawl')
    parser.add_argument('--crawl-list', help='crawling a list of hostnames', action='store_true', dest='crawl_list')
    parser.add_argument('--model', help='creates a doc2vec model', action='store_true', dest='model')
    parser.add_argument('--tsne', help='creates a tsne model', action='store_true', dest='tsne')
    args = parser.parse_args()
    if args.crawl:
        logger.info('Hostnames from CLI = ' + str(args.hostnames))
        logger.info('File Path from CLI = ' + args.output_directory)
        crawl_hostnames(args.hostnames, args.output_directory)

    elif args.model:
        create_doc2vec_model()

    # doc2vec_model_english, doc2vec_model_german =  create_doc2vec_model()
    # create_tsne_model(doc2vec_model_english)
    # create_tsne_model(doc2vec_model_german)
