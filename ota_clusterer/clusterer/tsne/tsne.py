import logging
import sklearn.manifold
import sklearn.cluster
import pandas as pd
from ota_clusterer.doc2vec import doc2vec
import numpy as np
import time
from ota_clusterer import settings
from ota_clusterer.clusterer.tsne.plots import plots
from ota_clusterer.clusterer.affinity_propagation import affinity_propagation

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



def main():
    # example usage for clustering of new t-sne model
    # doc2vec_model = doc2vec.load_existing_model('doc2vecmodel-27-Nov-2017-14:43:10')
    # doc2vec_vector_matrix = doc2vec.get_doc_vector_matrix(doc2vec_model)
    # tsne_model = create_2d_tsne_model(doc2vec_vector_matrix, 'cluster-doc2vec-')
    # affinity_propagation.create_affinity_propagation_cluster_doc2vec_plot(doc2vec_model, tsne_model, 'cluster-doc2vec-')

    doc2vec_model = doc2vec.load_existing_model('doc2vecmodel-27-Nov-2017-14:43:10')
    labels = doc2vec_model.docvecs.doctags.keys()
    tsne_model = load_tsne_model('cluster-doc2vec--27-Nov-2017-14:49:59-array.npy')
    plots.create_simple_tsne_model_plot(tsne_model, labels, 'simple_tsne_plot')


if __name__ == "__main__":
    main()
