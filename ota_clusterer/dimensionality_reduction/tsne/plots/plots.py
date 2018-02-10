import logging

import matplotlib.pyplot as plt
import seaborn as seaborn

from ota_clusterer.dimensionality_reduction.tsne import tsne

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_simple_tsne_model_plot(tsne_model, labels, filename):
    fig, ax = plt.subplots()

    for i, label in enumerate(labels):
        ax.annotate(label, (tsne_model[i, 0], tsne_model[i, 1]))

    plt.scatter(tsne_model[:, 0], tsne_model[:, 1])

    file_path_name = tsne.get_file_path_and_name_to_save(filename, "experiments/t-sne/")
    plt.savefig(file_path_name)
    plt.show(block=True)


def create_tsne_word2vec_model_scatter_plot(dataframe, filename):
    seaborn.set("poster")
    dataframe.plot.scatter("x", "y", s=10, figsize=(10, 6))

    file_path_name = tsne.get_file_path_and_name_to_save(filename, "experiments/t-sne/")
    plt.savefig(file_path_name)

    plt.show(block=True)


def create_tsne_doc2vec_model_plot(doc2vec_model, tsne_model, filename):
    labels = list(doc2vec_model.docvecs.doctags.keys())
    plt.figure(num=1, figsize=(10, 6), facecolor="w", edgecolor="k")

    for label, doc in zip(labels, tsne_model):
        plt.plot(doc[0], doc[1], ".")
        plt.annotate(label, (doc[0], doc[1]))

    file_path_name = tsne.get_file_path_and_name_to_save(filename, "experiments/t-sne/")
    plt.savefig(file_path_name)
