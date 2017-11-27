import logging
import seaborn as seaborn
import matplotlib.pyplot as plt
from ota_clusterer import settings
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_tsne_word2vec_model_scatter_plot(dataframe, filename):
    seaborn.set("poster")
    dataframe.plot.scatter("x", "y", s=10, figsize=(10, 6))

    file_name = 't-sne-' + filename + "-" + time.strftime("%d-%b-%Y-%X")
    file_path = settings.DATA_DIR + "experiments/t-sne/"
    plt.savefig(file_path + file_name)

    plt.show(block=True)


def create_tsne_doc2vec_model_plot(doc2vec_model, tsne_model, filename):
    labels = list(doc2vec_model.docvecs.doctags.keys())
    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")

    for label, doc in zip(labels, tsne_model):
        plt.plot(doc[0], doc[1], ".")
        plt.annotate(label, (doc[0], doc[1]))

    file_name = 't-sne-' + filename + "-" + time.strftime("%d-%b-%Y-%X")
    file_path = settings.DATA_DIR + "experiments/t-sne/"
    plt.savefig(file_path + file_name)
