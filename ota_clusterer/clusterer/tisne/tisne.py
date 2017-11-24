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


def create_2d_tsne(word_vector_matrix):
    tsne = sklearn.manifold.TSNE(n_components = 2,
                                 early_exaggeration = 6,
                                 learning_rate = 500,
                                 n_iter = 2000,
                                 random_state = 2,
                                 verbose=5)

    tsne_2d_word_vector_matrix = tsne.fit_transform(word_vector_matrix)
    np.save('tsne.array', tsne_2d_word_vector_matrix)

    return tsne_2d_word_vector_matrix


def generate_word_coordinate_dataframe(tsne_word_vector_matrix, doc2vec_model):
    dataframe = pd.DataFrame(
        [(word, coords[0], coords[1])
         for word, coords in [
             (word, tsne_word_vector_matrix[doc2vec_model.wv.vocab[word].index])
            for word in doc2vec_model.wv.vocab
         ]],
        columns=["word", "x", "y"])

    return dataframe


def create_plot(dataframe):
    seaborn.set("poster")
    dataframe.plot.scatter("x", "y", s=10, figsize=(10,6))

    file_name = 't-sne-' + time.strftime("%d-%b-%Y-%X")
    file_path = settings.DATA_DIR + "experiments/t-sne/"
    plt.savefig(file_path + file_name)
    
    plt.show(block=True)


def main():
    doc2vec_model = doc2vec.load_existing_model('doc2vec_model')
    # word_vector_matrix = doc2vec.create_word_vector_matrix(doc2vec_model)
    # tsne_word_vector_matrix = create_2d_tsne(word_vector_matrix)
    tsne_word_vector_matrix = np.load('tsne.array.npy')
    word_coordinate_dataframe = generate_word_coordinate_dataframe(tsne_word_vector_matrix, doc2vec_model)
    create_plot(word_coordinate_dataframe)


if __name__ == "__main__":
    main()