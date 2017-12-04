from ota_clusterer.webcrawler.webcrawler.Crawler import Crawler
from ota_clusterer.doc2vec import doc2vec
from ota_clusterer.clusterer.tsne import tsne
from ota_clusterer import logger

logger = logger.get_logger()


def get_crawled_data():
    crawler = Crawler()
    crawler.main()


def create_doc2vec_model():
    doc2vec_model_english, doc2vec_model_german = doc2vec.create_new_doc2vec_model()
    return doc2vec_model_english, doc2vec_model_german


def create_tsne_model(doc2vec_model):
    tsne.create_new_doc2vec_tsne_model_and_clustering(doc2vec_model)


if __name__ == "__main__":
     get_crawled_data()
    # doc2vec_model_english, doc2vec_model_german =  create_doc2vec_model()
    # create_tsne_model(doc2vec_model_english)
    # create_tsne_model(doc2vec_model_german)
