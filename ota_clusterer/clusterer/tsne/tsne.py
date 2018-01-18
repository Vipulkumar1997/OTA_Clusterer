import logging
import sklearn.manifold
import sklearn.cluster
import pandas as pd
from ota_clusterer.word_embeddings.doc2vec import doc2vec
import numpy as np
import time
from ota_clusterer import settings
from ota_clusterer.clusterer.tsne.plots import plots
from ota_clusterer.clusterer.affinity_propagation import affinity_propagation
from ota_clusterer import logger

logger = logger.get_logger()


# values inspired by http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
def create_2d_tsne_model(vector_matrix, filename):
    tsne = sklearn.manifold.TSNE(n_components=2,
                                 perplexity=30,
                                 early_exaggeration=12,
                                 learning_rate=200,
                                 init='pca',
                                 n_iter=20000,
                                 random_state=2,
                                 verbose=5)

    tsne_2d_model = tsne.fit_transform(vector_matrix)

    file_path_name = get_file_path_and_name_to_save(filename, "tsne/models/")
    np.save(file_path_name, tsne_2d_model)

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


def load_tsne_model(modelname):
    file_path = settings.DATA_DIR + "tsne/models/"
    tsne_model = np.load(file_path + modelname)
    return tsne_model


# TODO Probably refactoring in utils file
def get_file_path_and_name_to_save(file_name, file_path):
    file_name = 't-sne-' + file_name + "-" + time.strftime("%d-%b-%Y-%X")
    file_path = settings.DATA_DIR + file_path
    file_path_and_name = file_path + file_name
    return file_path_and_name


def create_new_doc2vec_tsne_model_and_clustering(doc2vec_model):
    doc2vec_vector_matrix = doc2vec.get_doc_vector_matrix(doc2vec_model)
    tsne_model = create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-doc2vec-')
    affinity_propagation.create_affinity_propagation_cluster_doc2vec_plot(doc2vec_model, tsne_model, 'cluster-doc2vec-')


def main():
    # Experiment to extend TSNE with unseen data --> refactoring asap!
    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    doc2vec_vector_matrix = doc2vec.create_doc_vector_matrix_for_unseen_documents(doc2vec_model,
                                                                                  unseen_documents=['fckickers.ch',
                                                                                                    'pdgr.ch'],
                                                                                  language='german')
    create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-unseen-data-doc2vec-german')
    

    '''
    
    # example usage for affinity_propagation clustering of new t-sne model
    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-english-11-Dec-2017-17:07:03')
    doc2vec_vector_matrix = doc2vec.get_doc_vector_matrix(doc2vec_model)
    tsne_model = create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-doc2vec-english')
    affinity_propagation.create_affinity_propagation_cluster_doc2vec_plot(doc2vec_model, tsne_model, 'cluster-doc2vec-english-')

    # example usage for clustering of new t-sne model
    doc2vec_model = doc2vec.load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    doc2vec_vector_matrix = doc2vec.get_doc_vector_matrix(doc2vec_model)
    tsne_model = create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-doc2vec-german')
    affinity_propagation.create_affinity_propagation_cluster_doc2vec_plot(doc2vec_model, tsne_model, 'cluster-doc2vec-german-')

    # TODO - DELETE ASAP
    # doc2vec_model = doc2vec.load_existing_model('doc2vec-model-english-11-Dec-2017-17:07:03')
    # labels = doc2vec_model.docvecs.doctags.keys()
    # tsne_model = load_tsne_model('cluster-doc2vec--27-Nov-2017-14:49:59-array.npy')
    # plots.create_simple_tsne_model_plot(tsne_model, labels, 'simple_tsne_plot')
    
    '''


if __name__ == "__main__":
    main()
