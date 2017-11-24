import logging
import sklearn.manifold
import pandas as pd
import seaborn as seaborn
import matplotlib.pyplot as plt
from ota_clusterer.doc2vec import doc2vec
import numpy as np
import time
from ota_clusterer import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_2d_tsne_model(vector_matrix, mode='load', filename=None):
    tsne = sklearn.manifold.TSNE(n_components=2,
                                 early_exaggeration=6,
                                 learning_rate=500,
                                 n_iter=2000,
                                 random_state=2,
                                 verbose=5)

    tsne_2d_model = tsne.fit_transform(vector_matrix)

    if mode == 'new':
        filename = filename + "-" + time.strftime("%d-%b-%Y-%X") + '-array'
        np.save(filename, tsne_2d_model)

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


def create_scatter_plot(dataframe, filename):
    seaborn.set("poster")
    dataframe.plot.scatter("x", "y", s=10, figsize=(10, 6))

    file_name = 't-sne-' + filename + "-" + time.strftime("%d-%b-%Y-%X")
    file_path = settings.DATA_DIR + "experiments/t-sne/"
    plt.savefig(file_path + file_name)

    plt.show(block=True)


def create_doc_model_plot(tsne_doc_model, labels, filename):
    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")

    for label, doc in zip(labels, tsne_doc_model):
        plt.plot(doc[0], doc[1], ".")
        plt.annotate(label, (doc[0], doc[1]))

    file_name = 't-sne-' + filename + "-" + time.strftime("%d-%b-%Y-%X")
    file_path = settings.DATA_DIR + "experiments/t-sne/"
    plt.savefig(file_path + file_name)


def main():
    # generate new t-sne model
    # doc2vec_model = doc2vec.load_existing_model('doc2vec_model')
    # word_vector_matrix = doc2vec.create_word_vector_matrix(doc2vec_model)
    # tsne_word_vector_model= create_2d_tsne_model(word_vector_matrix)

    doc2vec_model = doc2vec.load_existing_model('doc2vec_model')
    # tsne_word_vector_model = np.load('tsne.wordvector.array.npy')
    # word2vec_dataframe = generate_word2vec_word_coordinate_dataframe(tsne_word_vector_model, doc2vec_model)
    # create_scatter_plot(word2vec_dataframe, 'word2vec')

    # generate new doc vector t-sne model
    doc_vector_matrix = doc2vec.create_doc_vector_matrix(doc2vec_model)
    tsne_doc_vector_model = create_2d_tsne_model(doc_vector_matrix,
                                                 mode="new",
                                                 filename='docvectors')
    labels = list(doc2vec_model.docvecs.doctags.keys())
    create_doc_model_plot(tsne_doc_vector_model, labels, 'docvectors')


if __name__ == "__main__":
    main()
