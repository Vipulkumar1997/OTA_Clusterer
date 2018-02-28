#!/usr/bin/env python

__author__ = 'Sandro Cilurzo'

import gensim
import os
import errno
import glob
import time
import numpy as np
import re
from ota_clusterer import settings
from ota_clusterer import logger
from ota_clusterer.word_embeddings.preprocessing import preprocessing

logger = logger.get_logger()
logger.name = __name__


def create_document_corpus_by_language(documents_path, single_language_support=False):
    """ reads in crawled documents and create a doc2vec document corpus
    previous crawled documents gets preprocessed and a doc2vec document corpus gets build separated by language (german
    and english)
    :param documents_path: file path of the crawled documents
    :param single_language_support: just support one language per hostname
    :return: german and english document corpus
    """

    logger.info('start creating document corpus by language')
    folders_in_directory = glob.glob(documents_path)

    if not folders_in_directory:
        raise IOError

    else:

        preprocessed_documents_corpus_english = []
        preprocessed_documents_corpus_german = []

        for folder_name in folders_in_directory:
            logger.info('start getting files of folder ' + folder_name)
            pattern = os.path.join(folder_name, '*.txt')
            file_names = glob.glob(pattern)

            if not file_names:
                raise IOError

            else:

                if single_language_support is True:
                    logger.info('single language support is enabled')
                    preprocessed_documents_by_directory_english = []
                    preprocessed_documents_by_directory_german = []

                logger.info('start read in files')
                for file_name in file_names:
                    with open(file_name, 'r') as file:
                        document = file.read()

                    document = gensim.utils.simple_preprocess(document)
                    document_language = preprocessing.detect_language(document)
                    if document_language == 'english' or document_language == 'german':
                        preprocessed_document = preprocessing.preprocess_document(document, document_language)

                        tagged_document_name = remove_file_path_from_folder_name(folder_name)
                        tagged_document = gensim.models.doc2vec.TaggedDocument(preprocessed_document,
                                                                               ["{}".format(tagged_document_name)])

                        if document_language == 'english':
                            if single_language_support is True:
                                preprocessed_documents_by_directory_english.append(tagged_document)
                            else:
                                preprocessed_documents_corpus_english.append(tagged_document)

                        elif document_language == 'german':
                            if single_language_support is True:
                                preprocessed_documents_by_directory_german.append(tagged_document)
                            else:
                                preprocessed_documents_corpus_german.append(tagged_document)

                if single_language_support is True:
                    number_of_english_documents = len(preprocessed_documents_by_directory_english)
                    number_of_german_documents = len(preprocessed_documents_by_directory_german)

                    if number_of_english_documents > number_of_german_documents:
                        for document in preprocessed_documents_by_directory_english:
                            preprocessed_documents_corpus_english.append(document)

                        logger.info(
                            'added ' + str(number_of_english_documents) + ' documents from ' + folder_name + ' to english corpus')

                    elif number_of_german_documents > number_of_english_documents:
                        for document in preprocessed_documents_by_directory_german:
                            preprocessed_documents_corpus_german.append(document)

                        logger.info(
                            'added ' + str(number_of_german_documents) + ' documents from ' + folder_name + ' to german corpus')

                    elif number_of_english_documents == number_of_german_documents:
                        logger.info('added documents of ' + folder_name + ' to both corpus')
                        for document in preprocessed_documents_by_directory_english:
                            preprocessed_documents_corpus_english.append(document)

                        for document in preprocessed_documents_by_directory_german:
                            preprocessed_documents_corpus_german.append(document)

        logger.info(
            'Added ' + str(len(preprocessed_documents_corpus_english)) + ' documents to the english document corpus')
        logger.info(
            'Added ' + str(len(preprocessed_documents_corpus_german)) + ' documents to the german document corpus')

        return preprocessed_documents_corpus_english, preprocessed_documents_corpus_german


def preprocess_new_documents(documents_path):
    """documents which were not included in a previous trained doc2vec model, gets preprocessed for further processing
    :param documents_path: file path to the newly crawled documents
    :return: preprocessed and concatenated english and german documents

    """

    logger.info('start processing unseen document')
    folders_in_directory = glob.glob(documents_path)

    if not folders_in_directory:
        raise IOError

    else:

        preprocessed_and_concatenated_documents_english = []
        preprocessed_and_concatenated_documents_german = []
        english_documents_counter = 0
        german_documents_counter = 0

        for folder_name in folders_in_directory:
            logger.info('start getting files of folder ' + folder_name)
            pattern = os.path.join(folder_name, '*.txt')
            file_names = glob.glob(pattern)

            if not file_names:
                raise IOError

            else:
                logger.info('start read in files')
                for file_name in file_names:
                    with open(file_name, 'r') as file:
                        document = file.read()

                    document = gensim.utils.simple_preprocess(document)
                    document_language = preprocessing.detect_language(document)
                    if document_language == 'english':
                        english_documents_counter += 1
                        preprocessed_document = preprocessing.preprocess_document(document,
                                                                                                     document_language)
                        preprocessed_and_concatenated_documents_english += preprocessed_document

                    elif document_language == 'german':
                        german_documents_counter += 1
                        preprocessed_document = preprocessing.preprocess_document(document,
                                                                                                     document_language)
                        preprocessed_and_concatenated_documents_german += preprocessed_document

        logger.info('Concatenated and preprocessed ' + str(english_documents_counter) + ' to one english document')
        logger.info('Concatenated and preprocessed ' + str(german_documents_counter) + ' to one german document')

        return preprocessed_and_concatenated_documents_english, preprocessed_and_concatenated_documents_german


def remove_file_path_from_folder_name(folder_name):
    """file path information gets removed with regex from folder name
    :param folder_name: folder name with file path information
    :return: cleaned folder name without file path

    """

    pattern = re.compile(r"(?:\\.|[^/\\])*/$")
    cleaned_folder_name = re.findall(pattern, folder_name)
    cleaned_folder_name = cleaned_folder_name[0].strip('/')
    return cleaned_folder_name


def create_doc2vec_model(document_corpus):
    """doc2vec model gets build for given document corpus
    :param document_corpus: previous build document corpus consists of multiple labeled documents
    :return: doc2vec model

    """

    # doc2vec hyperparameters -- inspired by: https://github.com/jhlau/doc2vec
    vector_size = 300
    window_size = 15
    min_count = 1
    sampling_threshold = 1e-5
    negative_size = 5
    train_epoch = 100
    dm = 0  # 0 = dbow; 1 = dmpv
    worker_count = 3  # number of parallel processes

    logger.info('start building Doc2Vec model')
    model = gensim.models.Doc2Vec(size=vector_size,
                                  window=window_size,
                                  min_count=min_count,
                                  sample=sampling_threshold,
                                  workers=worker_count,
                                  hs=0,
                                  dm=dm,
                                  negative=negative_size,
                                  dbow_words=1,
                                  dm_concat=1,
                                  iter=train_epoch)

    model.build_vocab(document_corpus)
    logger.info("model's vocubulary length: " + str(len(model.wv.vocab)))

    logger.info("start to train the model")
    model.train(document_corpus,
                total_examples=model.corpus_count,
                epochs=model.iter)

    return model


def save_doc2vec_model(doc2vec_model, file_name, directory_path=None):
    """persist trained doc2vec model
    :param doc2vec_model: previous trained doc2vec model
    :param file_name: file name for persisting
    :param directory_path: where to persist doc2vec model

    """

    file_name = file_name + "-" + time.strftime("%d-%b-%Y-%X")

    if directory_path is not None:
        doc2vec_model_path = directory_path

        if not os.path.exists(doc2vec_model_path + '/doc2vec'):
            try:
                os.makedirs(doc2vec_model_path + 'doc2vec')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        logger.info("save new doc2vec model at: " + doc2vec_model_path + 'doc2vec/')
        doc2vec_model.save(doc2vec_model_path + 'doc2vec/' + file_name)

    else:
        doc2vec_model_path = settings.DATA_DIR + 'doc2vec/models/'
        logger.info("save new doc2vec model at: " + doc2vec_model_path)
        doc2vec_model.save(doc2vec_model_path + file_name)


def load_existing_model(doc2vec_model_file_path=None, model_file_name=None):
    if doc2vec_model_file_path is None and model_file_name is not None:
        logger.info('Loading doc2vec models directly from project structure...')
        doc2vec_model_file_path = settings.DATA_DIR + 'doc2vec/models/' + model_file_name

    logger.info('load model from following path: ' + str(doc2vec_model_file_path))
    loaded_model = gensim.models.Doc2Vec.load(doc2vec_model_file_path)
    return loaded_model


def get_word_vectors_matrix(doc2vec_model):
    """retrieve word vectors value of given doc2vec model
    :param doc2vec_model:
    :return: word vectors matrix

    """
    logger.info('get word vectors matrix of doc2vec model')
    word_vectors = doc2vec_model.wv.syn0
    return word_vectors


def get_doc_vectors_matrix(doc2vec_model):
    """retrieve document vectors value of given doc2vec model
    :param doc2vec_model:
    :return: document vectors matrix

    """
    logger.info('get document vectors matrix of doc2vec model')
    docvec_vectors = doc2vec_model.docvecs
    return docvec_vectors


def create_doc_vector_matrix_for_new_documents(doc2vec_model, new_documents, model_language, documents_file_path=None):
    """creates a new document vectors matrix with inferred vectors of new documents
    :param doc2vec_model: previous trained doc2vec model
    :param new_documents: new documents which should be used to infer vectors
    :param model_language: language of doc2vec model
    :param documents_file_path: file path of crawled documents
    :return: extended document vector matrix

    """

    doc2vec_vector_matrix_to_extend = get_doc_vectors_matrix(doc2vec_model)
    doc2vec_vector_matrix_array_values = doc2vec_vector_matrix_to_extend.doctag_syn0

    for document_folder_name in new_documents:
        doc_vectors_english, doc_vectors_german = get_doc_vectors_for_new_documents(doc2vec_model, document_folder_name,
                                                                                    documents_file_path)

        if model_language == 'english':
            # dimension == vector size parameter in the doc2vec model (create_doc2vec_model())
            doc_vectors = np.reshape(doc_vectors_english, (1, 300))

        elif model_language == 'german':
            # dimension == vector size parameter in the doc2vec model (create_doc2vec_model())
            doc_vectors = np.reshape(doc_vectors_german, (1, 300))

        doc2vec_vector_matrix_array_values = np.concatenate((doc2vec_vector_matrix_array_values, doc_vectors), axis=0)

    doc2vec_vector_matrix_to_extend.doctag_syn0 = doc2vec_vector_matrix_array_values
    doc2vec_vector_matrix_to_extend.count += len(new_documents)
    doc2vec_vector_matrix_extended = doc2vec_vector_matrix_to_extend

    return doc2vec_vector_matrix_extended


def get_doc_similarities_by_document_name(doc2vec_model, document_name):
    """retrieve cosine similarities of document by document name
    :param doc2vec_model: trained doc2vec model
    :param document_name: name of document in doc2vec model
    :return: cosine similarities

    """
    logger.info('get document similarities')
    similarities = doc2vec_model.docvecs.most_similar(document_name)
    return similarities


def get_doc_similarities_by_new_vector(doc2vec_model, new_vector):
    """retrieve cosine similarities by new vector (word)
    :param doc2vec_model: trained doc2vec model
    :param new_vector: word to infer vector values
    :return: cosine similarities for given new vector (word)

    """
    logger.info('get document similarities by new vector')
    similarities = doc2vec_model.docvecs.most_similar([new_vector])
    return similarities


def get_most_similar_doc_matrix(doc2vec_model):
    """retrieve cosine similarities matrix of most similar documents of given doc2vec model
    :param doc2vec_model:
    :return: cosine similarities matrix

    """

    similarities_matrix = []
    for document_name in doc2vec_model.docvecs.doctags:
        similarities_matrix.append(get_doc_similarities_by_document_name(doc2vec_model, document_name))

    similarities_matrix = np.asarray(similarities_matrix)

    return similarities_matrix


def get_doc_vectors_for_new_documents(doc2vec_model, documents_folder_name, documents_file_path=None):
    """retrieve document vectors for new documents based on previous trained doc2vec model
    :param doc2vec_model: trained doc2vec model
    :param documents_folder_name: name of documents folder
    :param documents_file_path: file path to the folder
    :return: english and german document vectors

    """

    logger.info('get doc vector values of unseen documents')
    if documents_file_path is None:
        documents_file_path = settings.DATA_DIR + 'crawling_data/' + documents_folder_name + '/'

    else:
        documents_file_path = documents_file_path + documents_folder_name + '/'

    logger.info('get documents from following path: ' + documents_file_path)

    preprocessed_documents_english, preprocessed_documents_german = preprocess_new_documents(documents_file_path)

    doc_vectors_english = doc2vec_model.infer_vector(preprocessed_documents_english, alpha=0.025, min_alpha=0.01,
                                                    steps=1)
    doc_vectors_german = doc2vec_model.infer_vector(preprocessed_documents_german, alpha=0.025, min_alpha=0.01, steps=1)

    return doc_vectors_english, doc_vectors_german


def create_new_doc2vec_model(documents_file_path=None, save_to_directory=None, single_language_support=False):
    """helper function to create a new doc2vec model
    :param documents_file_path: file path to the crawled and stored documents
    :param save_to_directory: where to save the doc2vec model
    :return: english and german doc2vec model

    """

    if documents_file_path is not None:
        documents_file_path = documents_file_path + '*/'
        logger.info('document file has been set to: ' + str(documents_file_path))

    else:
        documents_file_path = settings.DATA_DIR + 'crawling_data/*/'
        logger.info('No documets file path has been given, default file path used: ' + str(documents_file_path))

    logger.info('Start creating new doc2vec model...')
    document_corpus_english, document_corpus_german = create_document_corpus_by_language(documents_file_path,
                                                                                         single_language_support)

    doc2vec_model_english = create_doc2vec_model(document_corpus_english)
    doc2vec_model_german = create_doc2vec_model(document_corpus_german)

    save_doc2vec_model(doc2vec_model_english, 'doc2vec-model-english', directory_path=save_to_directory)
    save_doc2vec_model(doc2vec_model_german, 'doc2vec-model-german', directory_path=save_to_directory)

    return doc2vec_model_english, doc2vec_model_german


def main():

    # Get doc2vec similarities from given model and document
    doc2vec_model = load_existing_model(model_file_name='doc2vec_single_language_full_model_german_18_Feb_2018_22_31_27')
    print(get_doc_similarities_by_document_name(doc2vec_model, 'booking-valais.ch'))

    '''
    
    # Some examples...
    # Create new doc2vec model based on data from data/crawling_data:
    # create_new_doc2vec_model()

    # Create a new 'single_language_support' doc2vec model based on data from: data/crawling_data:
    #create_new_doc2vec_model(single_language_support=True)

    # Get doc similarities of new data (not included in training set)
    doc2vec_model = load_existing_model(model_file_name='standard-models/doc2vec_model_german_17_Feb_2018_02_14_04')
    doc_vectors_english, doc_vectors_german = get_doc_vectors_for_new_documents(doc2vec_model=doc2vec_model,
                                                                                documents_folder_name='statravel.ch')
    print(get_doc_similarities_by_new_vector(doc2vec_model, doc_vectors_english))
    print(get_doc_similarities_by_new_vector(doc2vec_model, doc_vectors_german))
    
    '''


if __name__ == "__main__":
    main()
