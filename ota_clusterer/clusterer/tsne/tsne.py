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


def create_2d_tsne_model(vector_matrix, filename):
    tsne = sklearn.manifold.TSNE(n_components=2,
                                 early_exaggeration=6,
                                 learning_rate=200,
                                 n_iter=2000,
                                 random_state=2,
                                 verbose=5)

    tsne_2d_model = tsne.fit_transform(vector_matrix)

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


def create_word2vec_scatter_plot(dataframe, filename):
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


def load_tsne_vector_model(modelname):
    tsne_vector_model = np.load(modelname)
    return tsne_vector_model


def generate_word2vec_plot(doc2vec_model, tsne_model_name=None):
    if tsne_model_name is None:
        word_vector_matrix = doc2vec.create_word_vector_matrix(doc2vec_model)
        tsne_word_vector_model = create_2d_tsne_model(word_vector_matrix, 'wordvectors')

    else:
        tsne_word_vector_model = load_tsne_vector_model(tsne_model_name)

    word2vec_dataframe = generate_word2vec_word_coordinate_dataframe(tsne_word_vector_model, doc2vec_model)
    create_word2vec_scatter_plot(word2vec_dataframe, 'word2vec-')


def generate_doc2vec_plot(doc2vec_model, tsne_model_name=None):
    if tsne_model_name is None:
        doc_vector_matrix = doc2vec.create_doc_vector_matrix(doc2vec_model)
        tsne_doc_vector_model = create_2d_tsne_model(doc_vector_matrix, 'docvectors-')

    else:
        tsne_doc_vector_model = load_tsne_vector_model(tsne_model_name)

    labels = list(doc2vec_model.docvecs.doctags.keys())
    create_doc_model_plot(tsne_doc_vector_model, labels, 'docvectors-')


def main():
    doc2vec_model = doc2vec.load_existing_model('doc2vecmodel-25-Nov-2017-11:58:06')

    generate_word2vec_plot(doc2vec_model, 'wordvectors-25-Nov-2017-16:05:48-array.npy')
    generate_doc2vec_plot(doc2vec_model, 'docvectors-iter-10000-25-Nov-2017-12:03:40-array.npy')



if __name__ == "__main__":
    main()
