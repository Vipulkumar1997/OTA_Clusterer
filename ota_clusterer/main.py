import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
from ota_clusterer.webcrawler.Crawler import Crawler
from ota_clusterer.word_embeddings.doc2vec import doc2vec
from ota_clusterer.clusterer.tsne import tsne
from ota_clusterer import logger

logger = logger.get_logger()


def get_crawled_data():
    crawler = Crawler()
    hostnames = crawler.get_hostnames()
    crawler.crawl_hostnames(hostnames)


def create_doc2vec_model():
    doc2vec_model_english, doc2vec_model_german = doc2vec.create_new_doc2vec_model()
    return doc2vec_model_english, doc2vec_model_german


def create_tsne_model(doc2vec_model):
    tsne.create_new_doc2vec_tsne_model_and_clustering(doc2vec_model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl', help='crawling given website', action='store_true', dest='crawl')
    parser.add_argument('--model', help='creates a doc2vec model', action='store_true', dest='model')
    parser.add_argument('--tsne', help='creates a tsne model', action='store_true', dest='tsne')
    args = parser.parse_args()
    if args.crawl:
        get_crawled_data()

    elif args.model:
        create_doc2vec_model()

    # doc2vec_model_english, doc2vec_model_german =  create_doc2vec_model()
    # create_tsne_model(doc2vec_model_english)
    # create_tsne_model(doc2vec_model_german)
