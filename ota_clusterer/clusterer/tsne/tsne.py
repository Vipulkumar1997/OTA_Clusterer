import logging
import sklearn.manifold
import sklearn.cluster
import pandas as pd
import seaborn as seaborn
import matplotlib.pyplot as plt
from ota_clusterer.doc2vec import doc2vec
import numpy as np
import time
from ota_clusterer import settings
from itertools import cycle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_2d_tsne_model(vector_matrix, filename):
    tsne = sklearn.manifold.TSNE(n_components=2,
                                 early_exaggeration=6,
                                 learning_rate=200,
                                 n_iter=2000,
                                 random_state=2,
                                 verbose=5)

    tsne_2d_model = tsne.fit_transform(vector_matrix)

    file_path = settings.DATA_DIR + "t-sne/models"
    filename = filename + "-" + time.strftime("%d-%b-%Y-%X") + '-array'
    np.save(file_path + filename, tsne_2d_model)

    return tsne_2d_model


def generate_word2vec_word_coordinate_dataframe(tsne_word_vector_matrix, doc2vec_model):
    dataframe = pd.DataFrame(
        [(word, coords[0], coords[1])
         for word, coords in [
             (word, tsne_word_vector_matrix[doc2vec_model.wv.vocab[word].index])
             for word in doc2vec_model.wv.vocab
         ]],
        columns=["word", "x", "y"])

    return dataframe


# TODO: refactoring in other script
def create_word2vec_scatter_plot(dataframe, filename):
    seaborn.set("poster")
    dataframe.plot.scatter("x", "y", s=10, figsize=(10, 6))

    file_name = 't-sne-' + filename + "-" + time.strftime("%d-%b-%Y-%X")
    file_path = settings.DATA_DIR + "experiments/t-sne/"
    plt.savefig(file_path + file_name)

    plt.show(block=True)


# TODO: refactoring in other script
def create_doc_model_plot(tsne_doc_model, labels, filename):
    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")

    for label, doc in zip(labels, tsne_doc_model):
        plt.plot(doc[0], doc[1], ".")
        plt.annotate(label, (doc[0], doc[1]))

    file_name = 't-sne-' + filename + "-" + time.strftime("%d-%b-%Y-%X")
    file_path = settings.DATA_DIR + "experiments/t-sne/"
    plt.savefig(file_path + file_name)


# TODO: refactoring in other script
def create_affinity_propagation_cluster_doc2vec_plot(doc2vec_tsne_model, fnames, filename):
    affinity_propagation = sklearn.cluster.AffinityPropagation().fit(doc2vec_tsne_model)

    cluster_centers_indices = affinity_propagation.cluster_centers_indices_
    labels = affinity_propagation.labels_
    n_clusters_ = len(cluster_centers_indices)

    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")
    colors = cycle("bgrcmyk")

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k  # class_members ist array von boolschen werten, beschreibt cluster membership
        cluster_center = doc2vec_tsne_model[cluster_centers_indices[k]]

        fnames_cluster = []
        fname_indices = [i for i, x in enumerate(class_members) if x]
        for i in fname_indices: fnames_cluster.append(fnames[i])

        plt.plot(doc2vec_tsne_model[class_members, 0], doc2vec_tsne_model[class_members, 1], col + ".")
        plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markersize=20)

        # plt.annotate(fnames[labels[k]], (cluster_center[0], cluster_center[1]), xytext=(0, -8),
        #        textcoords="offset points", va="center", ha="left")

        for x, fname in zip(doc2vec_tsne_model[class_members], fnames_cluster):
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linestyle='--', linewidth=1)
            plt.annotate(fname, (x[0], x[1]), xytext=(0, -8),
                         textcoords="offset points", va="center", ha="left")

    file_path = settings.DATA_DIR + "experiments/t-sne/"
    file_name = 'affinity_propagation_cluster-' + filename + "-" + time.strftime("%d-%b-%Y-%X") + ".png"
    plt.savefig(file_path + file_name, facecolor="w", dpi=90)


def load_tsne_model(modelname):
    file_path = settings.DATA_DIR + "tsne/models/"
    tsne_model = np.load(file_path + modelname)
    return tsne_model


# TODO: refactoring in other script
def generate_word2vec_plot(doc2vec_model, tsne_model_name=None):
    if tsne_model_name is None:
        word_vector_matrix = doc2vec.create_word_vector_matrix(doc2vec_model)
        tsne_word_vector_model = create_2d_tsne_model(word_vector_matrix, 'wordvectors')

    else:
        tsne_word_vector_model = load_tsne_model(tsne_model_name)

    word2vec_dataframe = generate_word2vec_word_coordinate_dataframe(tsne_word_vector_model, doc2vec_model)
    create_word2vec_scatter_plot(word2vec_dataframe, 'word2vec-')


# TODO: refactoring in other script
def generate_doc2vec_plot(doc2vec_model, tsne_model_name=None):
    if tsne_model_name is None:
        doc_vector_matrix = doc2vec.create_doc_vector_matrix(doc2vec_model)
        tsne_doc_vector_model = create_2d_tsne_model(doc_vector_matrix, 'docvectors-')

    else:
        tsne_doc_vector_model = load_tsne_model(tsne_model_name)

    labels = list(doc2vec_model.docvecs.doctags.keys())
    create_doc_model_plot(tsne_doc_vector_model, labels, 'docvectors-')


def main():
    doc2vec_model = doc2vec.load_existing_model('doc2vecmodel-25-Nov-2017-11:58:06')

    # generate_word2vec_plot(doc2vec_model, 'wordvectors-25-Nov-2017-16:05:48-array.npy')
    # generate_doc2vec_plot(doc2vec_model, 'docvectors-iter-10000-25-Nov-2017-12:03:40-array.npy')

    labels = list(doc2vec_model.docvecs.doctags.keys())
    # doc2vec_vector_model = create_2d_docvec_tsne_model(doc2vec_model, 'cluster-doc2vec-')
    doc2vec_vector_matrix = doc2vec.create_doc_vector_matrix(doc2vec_model)
    tsne_model = create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-doc2vec-')

    # refactoring in other script
    create_affinity_propagation_cluster_doc2vec_plot(tsne_model, labels, 'cluster-doc2vec-')


if __name__ == "__main__":
    main()
