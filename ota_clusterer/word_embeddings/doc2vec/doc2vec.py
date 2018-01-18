import gensim
import os
import glob
from ota_clusterer import settings
import time
from ota_clusterer.word_embeddings.preprocessing import preprocessing
import numpy as np
from ota_clusterer import logger
import re

logger = logger.get_logger()
logger.name = __name__


def create_document_corpus_by_language(document_path):
    logger.info('start creating document corpus by language')
    folders_in_directory = glob.glob(document_path)

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
                logger.info('start read in files')
                for file_name in file_names:
                    logger.debug('File Names: ', file_names)
                    with open(file_name, 'r') as file:
                        document = file.read()

                    document = gensim.utils.simple_preprocess(document)
                    document_language = preprocessing.detect_language(document)
                    if document_language == 'english' or document_language == 'german':
                        preprocessed_document, document_language = preprocessing.preprocess_document(document, document_language)

                        tagged_document_name = cleaning_path_out_of_folder_name(folder_name)
                        tagged_document = gensim.models.doc2vec.TaggedDocument(preprocessed_document,
                                                                               ["{}".format(tagged_document_name)])

                        if document_language == 'english':
                            preprocessed_documents_corpus_english.append(tagged_document)

                        elif document_language == 'german':
                            preprocessed_documents_corpus_german.append(tagged_document)

        logger.info('Added ' + str(len(preprocessed_documents_corpus_english)) + ' documents to the english document corpus')
        logger.info('Added ' + str(len(preprocessed_documents_corpus_german)) + ' documents to the german document corpus')

        return preprocessed_documents_corpus_english, preprocessed_documents_corpus_german


def preprocess_unseen_documents(document_path):
    logger.info('start processing unseen document')
    folders_in_directory = glob.glob(document_path)

    if not folders_in_directory:
        raise IOError

    else:

        preprocessed_and_concatenated_document_english = []
        preprocessed_and_concatenated_document_german = []
        english_document_counter = 0
        german_document_counter = 0

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
                        english_document_counter += 1
                        preprocessed_document, document_language = preprocessing.preprocess_document(document, document_language)
                        preprocessed_and_concatenated_document_english += preprocessed_document

                    elif document_language == 'german':
                        german_document_counter += 1
                        preprocessed_document, document_language = preprocessing.preprocess_document(document, document_language)
                        preprocessed_and_concatenated_document_german += preprocessed_document


        logger.info('Concatenated and preprocessed ' + str(english_document_counter) + ' to one english document ')
        logger.info('Concatenated and preprocessed ' + str(german_document_counter) + ' to one german document')

        return preprocessed_and_concatenated_document_english, preprocessed_and_concatenated_document_german


def create_doc2vec_model(document_corpus):

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


def save_doc2vec_model(doc2vec_model, file_name):
    doc2vec_model_path = settings.DATA_DIR + 'doc2vec/models/'
    file_name = file_name + "-" + time.strftime("%d-%b-%Y-%X")
    logger.info("save new doc2vec model at: " + doc2vec_model_path + file_name)
    doc2vec_model.save(doc2vec_model_path + file_name)


def get_word_vector_matrix(model):
    logger.info('get word vectors of doc2vec model')
    word_vectors = model.wv.syn0
    return word_vectors


def get_doc_vector_matrix(model):
    logger.info('get document vectors of doc2vec model')
    docvec_vectors = model.docvecs
    return docvec_vectors


def create_doc_vector_matrix_for_unseen_documents(doc2vec_model, unseen_documents, language):
    doc2vec_vector_matrix_to_extend = get_doc_vector_matrix(doc2vec_model)
    doc2vec_vector_matrix_array_values = doc2vec_vector_matrix_to_extend.doctag_syn0

    for document in unseen_documents:
        doc_vectors_english, doc_vectors_german = get_doc_vectors_of_unseen_documents(doc2vec_model, document)

        if language == 'english':
            # dimension == vector size parameter in the doc2vec model (create_doc2vec_model())
            doc_vectors = np.reshape(doc_vectors_english, (1, 300))

        elif language == 'german':
            # dimension == vector size parameter in the doc2vec model (create_doc2vec_model())
            doc_vectors = np.reshape(doc_vectors_german, (1, 300))

        doc2vec_vector_matrix_array_values = np.concatenate((doc2vec_vector_matrix_array_values, doc_vectors), axis=0)

    doc2vec_vector_matrix_to_extend.doctag_syn0 = doc2vec_vector_matrix_array_values
    doc2vec_vector_matrix_to_extend.count += len(unseen_documents)

    return doc2vec_vector_matrix_to_extend


def get_doc_similarities_by_document_name(doc2vec_model, document_name):
    logger.info('get document similarities')
    similarities = doc2vec_model.docvecs.most_similar(document_name)
    return similarities


def get_doc_similarities_by_new_vector(doc2vec_model, new_vector):
    logger.info('get document similarities by new vector')
    similarities = doc2vec_model.docvecs.most_similar([new_vector])
    return similarities


def get_doc_vectors_of_unseen_documents(doc2vec_model, document_folder_name):
    logger.info('get doc vector values of unseen documents')
    document_file_path = settings.DATA_DIR + 'crawling_data/' + document_folder_name + '/'
    preprocessed_documents_english, preprocessed_documents_german = preprocess_unseen_documents(document_file_path)

    doc_vector_english = doc2vec_model.infer_vector(preprocessed_documents_english, alpha=0.025, min_alpha=0.01, steps=1)
    doc_vector_german = doc2vec_model.infer_vector(preprocessed_documents_german, alpha=0.025, min_alpha=0.01, steps=1)

    return doc_vector_english, doc_vector_german


def load_existing_model(model_name):
    models_file_path = settings.DATA_DIR + 'doc2vec/models/'
    logger.info('load model from following path: ' + models_file_path)

    loaded_model = gensim.models.Doc2Vec.load(models_file_path + model_name)
    return loaded_model


def create_new_doc2vec_model():
    logger.info('Start creating new doc2vec model...')
    # crawling_data_file_path = settings.PROJECT_ROOT + '/data/crawling_data/*/'
    # utils.save_all_documents(crawling_data_file_path)

    documents_file_path = settings.DATA_DIR + 'crawling_data/*/'
    document_corpus_english, document_corpus_german = create_document_corpus_by_language(documents_file_path)

    doc2vec_model_english = create_doc2vec_model(document_corpus_english)
    doc2vec_model_german = create_doc2vec_model(document_corpus_german)

    save_doc2vec_model(doc2vec_model_english, 'doc2vec-model-english')
    save_doc2vec_model(doc2vec_model_german, 'doc2vec-model-german')

    return doc2vec_model_english, doc2vec_model_german


def get_most_similar_doc_matrix(doc2vec_model):
    similarities_matrix = []
    for document_name in doc2vec_model.docvecs.doctags:
        similarities_matrix.append(get_doc_similarities_by_document_name(doc2vec_model, document_name))

    similarities_matrix = np.asarray(similarities_matrix)

    return similarities_matrix


def cleaning_path_out_of_folder_name(folder_name):
    pattern = re.compile(r"(?:\\.|[^/\\])*/$")
    file_name = re.findall(pattern, folder_name)
    file_name = file_name[0].strip('/')
    return file_name


def main():
    # create_new_doc2vec_model()
    # get doc2vec similarities
    # doc2vec_model = load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    # print(get_doc_similarities(doc2vec_model, 'booking-valais.ch'))

    # Experiments with unseen data
    doc2vec_model = load_existing_model('doc2vec-model-german-11-Dec-2017-17:07:03')
    doc_vectors_english, doc_vectors_german = get_doc_vectors_of_unseen_documents(doc2vec_model=doc2vec_model, document_folder_name='statravel.ch')
    print(get_doc_similarities_by_new_vector(doc2vec_model, doc_vectors_english))
    print(get_doc_similarities_by_new_vector(doc2vec_model, doc_vectors_german))


if __name__ == "__main__":
    main()