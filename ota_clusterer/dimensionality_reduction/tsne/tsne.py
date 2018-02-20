#!/usr/bin/env python

__author__ = 'Sandro Cilurzo'

import time
import os
import errno
import numpy as np
import sklearn.cluster
import sklearn.manifold
from ota_clusterer import logger
from ota_clusterer import settings
from ota_clusterer.word_embeddings.doc2vec import doc2vec

logger = logger.get_logger()
logger.name = __name__


def create_and_save_2d_tsne_model(vector_matrix, tsne_file_name, output_directory_tsne_model=None):
    """generates a 2D tsne model for a given vector matrix
    :param vector_matrix: matrix with numerical vector values
    :param tsne_file_name: file name to store the tsne model
    :param output_directory_tsne_model: where to store the tsne model
    :return: 2D tsne model

    """

    # values inspired by http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    tsne = sklearn.manifold.TSNE(n_components=2,
                                 perplexity=30,
                                 early_exaggeration=12,
                                 learning_rate=200,
                                 init='pca',
                                 n_iter=20000,
                                 random_state=2,
                                 verbose=5)

    tsne_2d_model = tsne.fit_transform(vector_matrix)

    tsne_file_name = 't-sne-' + tsne_file_name + "-" + time.strftime("%d-%b-%Y-%X")

    if output_directory_tsne_model is None:
        output_directory_tsne_model = settings.DATA_DIR + "tsne/models/"

    else:
        if not os.path.exists(output_directory_tsne_model + '/tsne'):
            try:
                os.makedirs(output_directory_tsne_model + 'tsne')
            except OSError as e:
                if e.errno != errno.EEXIST:
                        raise
        output_directory_tsne_model += '/tsne/'

    file_path_and_name = output_directory_tsne_model + tsne_file_name
    np.save(file_path_and_name, tsne_2d_model)

    return tsne_2d_model


def load_tsne_model(model_file_name=None, tsne_model_file_path=None):
    """load previous trained tsne model from file system
    :param model_file_name: file name of the tsne model to load
    :param tsne_model_file_path: file path of the tsne model
    :return: loaded tsne model 
    
    """
    if model_file_name is not None and tsne_model_file_path is None:
        tsne_model_file_path = settings.DATA_DIR + "tsne/models/" + model_file_name

    tsne_model = np.load(tsne_model_file_path)
    return tsne_model


def create_tsne_for_doc2vec_model(doc2vec_model_file_path, output_directory_tsne_model, tsne_file_name):
    """creates a tsne model based on a given doc2vec model
    :param doc2vec_model_file_path: file path to doc2vec model
    :param output_directory_tsne_model: where to store the doc2vec model
    :param tsne_file_name: 
    
    """
    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path=doc2vec_model_file_path)
    doc2vec_vector_matrix = doc2vec.get_doc_vectors_matrix(doc2vec_model)
    create_and_save_2d_tsne_model(doc2vec_vector_matrix, tsne_file_name, output_directory_tsne_model)


def create_tsne_for_doc2vec_model_with_new_documents(doc2vec_model_file_path, new_documents, model_language,
                                                     output_directory,
                                                     tsne_file_name,
                                                     documents_file_path):
    """creates a tsne model based on a given doc2vec model with new documents which hasn't been processed in given 
    doc2vec model
    :param doc2vec_model_file_path: file path to doc2vec model
    :param new_documents: new documents to process
    :param model_language: used language in doc2vec model
    :param output_directory: where to store the tsne model
    :param tsne_file_name: file name of the tsne model
    :param documents_file_path: file path where the crawled documents are stored for processing
    
    """

    doc2vec_model = doc2vec.load_existing_model(doc2vec_model_file_path=doc2vec_model_file_path)
    doc2vec_vectors_matrix = doc2vec.create_doc_vector_matrix_for_new_documents(doc2vec_model,
                                                                               new_documents=new_documents,
                                                                               model_language=model_language,
                                                                               documents_file_path=documents_file_path)

    create_and_save_2d_tsne_model(doc2vec_vectors_matrix, tsne_file_name, output_directory)


def main():
    # Some examples...

    # Create 2D t-SNE Model
    doc2vec_model = doc2vec.load_existing_model(model_file_name='doc2vec-single_language_full-model-german-18-Feb-2018-22:31:27')
    doc2vec_vector_matrix = doc2vec.get_doc_vectors_matrix(doc2vec_model)
    create_and_save_2d_tsne_model(doc2vec_vector_matrix, 'single_language_full-model-doc2vec-model-german')


    ''' 
    
    Example to to create tsne model with new data
    
    logger.info('Start building tsne model with new data at: ' + time.strftime("%d-%b-%Y-%X"))
    doc2vec_model = doc2vec.load_existing_model(model_file_name='doc2vec-model-german-17-Feb-2018-02:14:04')
    doc2vec_vector_matrix = doc2vec.create_doc_vector_matrix_for_new_documents(doc2vec_model,
                                                                               new_documents=['upkbs.ch',
                                                                                              'curaneo.ch',
                                                                                              'bscyb.ch',
                                                                                              'scltigers.ch',
                                                                                              'graubuenden.ch'],
                                                                               model_language='german',
                                                                               documents_file_path='/home/sandro/vm1/OTA_Clusterer/data/crawling_data_experiments/')
    create_and_save_2d_tsne_model(doc2vec_vector_matrix, 'full-doc2vec-model-new-data-german')
    logger.info('Finished building tsne model with new data at: ' + time.strftime("%d-%b-%Y-%X"))
    
    '''


if __name__ == "__main__":
    main()
