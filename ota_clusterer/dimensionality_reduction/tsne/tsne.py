import time
import os
import errno
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.manifold

from ota_clusterer import logger
from ota_clusterer import settings
from ota_clusterer.clusterer.affinity_propagation import affinity_propagation
from ota_clusterer.word_embeddings.doc2vec import doc2vec

logger = logger.get_logger()
logger.name = __name__


# values inspired by http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
def create_2d_tsne_model(vector_matrix, file_name, output_directory_tsne_model):
    tsne = sklearn.manifold.TSNE(n_components=2,
                                 perplexity=30,
                                 early_exaggeration=12,
                                 learning_rate=200,
                                 init='pca',
                                 n_iter=20000,
                                 random_state=2,
                                 verbose=5)

    tsne_2d_model = tsne.fit_transform(vector_matrix)

    file_name = 't-sne-' + file_name + "-" + time.strftime("%d-%b-%Y-%X")

    if output_directory_tsne_model is None:
        output_directory_tsne_model = settings.DATA_DIR + "tsne/models/"

    if not os.path.exists(output_directory_tsne_model + '/tsne'):
        try:
            os.makedirs(output_directory_tsne_model + 'tsne')
        except OSError as e:
            if e.errno != errno.EEXIST:
                    raise

    file_path_and_name = output_directory_tsne_model + '/tsne/' + file_name
    np.save(file_path_and_name, tsne_2d_model)

    return tsne_2d_model


def generate_tsne_word2vec_dataframe(tsne_word2vec_model, doc2vec_model):
    dataframe = pd.DataFrame(
        [(word, coords[0], coords[1])
         for word, coords in [
             (word, tsne_word2vec_model[doc2vec_model.wv.vocab[word].index])
             for word in doc2vec_model.wv.vocab
         ]],
        columns=["word", "x", "y"])

    return dataframe


def load_tsne_model(file_model_name, tsne_model_file_path):
    if tsne_model_file_path is None:
        tsne_model_file_path = settings.DATA_DIR + "tsne/models/"

    tsne_model = np.load(tsne_model_file_path + file_model_name)
    return tsne_model


def create_new_doc2vec_tsne_model(doc2vec_model_file_path, output_directory_tsne_model, tsne_file_name):
    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path=doc2vec_model_file_path)
    doc2vec_vector_matrix = doc2vec.get_doc_vector_matrix(doc2vec_model)
    create_2d_tsne_model(doc2vec_vector_matrix, tsne_file_name, output_directory_tsne_model)


def create_new_doc2vec_tsne_model_for_unseen_data(doc2vec_model_file_path, unseen_data, model_language, output_directory_tsne_model, tsne_file_name):
    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path)
    doc2vec_vector_matrix = doc2vec.create_doc_vector_matrix_for_unseen_documents(doc2vec_model,
                                                                                  unseen_documents=unseen_data,
                                                                                  language=model_language)
    create_2d_tsne_model(doc2vec_vector_matrix, tsne_file_name, output_directory_tsne_model)


def main():

    # Example to to create tsne model with unseen data
    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    doc2vec_vector_matrix = doc2vec.create_doc_vector_matrix_for_unseen_documents(doc2vec_model,
                                                                                  unseen_documents=['fckickers.ch',
                                                                                                    'pdgr.ch'],
                                                                                  language='german')
    create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-unseen-data-doc2vec-german')

    '''

    # example usage for affinity_propagation clustering of new t-sne model in english and german
    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-english-11-Dec-2017-17:07:03')
    doc2vec_vector_matrix = doc2vec.get_doc_vector_matrix(doc2vec_model)
    tsne_model = create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-doc2vec-english')
    affinity_propagation.create_affinity_propagation_cluster_doc2vec_plot(doc2vec_model, tsne_model, 'cluster-doc2vec-english-')

    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    doc2vec_vector_matrix = doc2vec.get_doc_vector_matrix(doc2vec_model)
    tsne_model = create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-doc2vec-german')
    affinity_propagation.create_affinity_propagation_cluster_doc2vec_plot(doc2vec_model, tsne_model, 'cluster-doc2vec-german-')

    '''


if __name__ == "__main__":
    main()
