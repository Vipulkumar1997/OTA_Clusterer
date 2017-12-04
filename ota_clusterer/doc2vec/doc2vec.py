import gensim
import os
import glob
from ota_clusterer import settings
from ota_clusterer.doc2vec.utils import utils
import time
from ota_clusterer.doc2vec.plots import plots
from ota_clusterer.doc2vec.preprocessing import  preprocessing
import numpy as np
from ota_clusterer import logger

logger = logger.get_logger()
logger.name = __name__


def create_document_corpus_by_language(document_path):
    logger.info('start creating document corpus by language')
    pattern = os.path.join(document_path, '*.txt')

    preprocessed_documents_corpus_english = []
    preprocessed_documents_corpus_german = []

    file_names = glob.glob(pattern)

    if not file_names:
        raise IOError

    else:
        logger.info('Start read in files')
        for file_name in file_names:
            logger.debug('File Names: ', file_names)
            with open(file_name, 'r') as file:
                document = file.read()

            document = gensim.utils.simple_preprocess(document)
            preprocessed_document, document_language = preprocessing.preprocess_document(document)

            tagged_document_name = utils.cleaning_path_out_of_file_name(file_name)
            tagged_document = gensim.models.doc2vec.TaggedDocument(preprocessed_document,
                                                                   ["{}".format(tagged_document_name)])


            if document_language == 'english':
                preprocessed_documents_corpus_english.append(tagged_document)

            elif document_language == 'german':
                preprocessed_documents_corpus_german.append(tagged_document)

    logger.info('Added ' + str(len(preprocessed_documents_corpus_english)) + ' documents to the english document corpus')
    logger.info('Added ' + str(len(preprocessed_documents_corpus_german)) + ' documents to the germand document corpus')
    return preprocessed_documents_corpus_english, preprocessed_documents_corpus_german


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
    doc2vec_model_path = settings.PROJECT_ROOT + '/data/doc2vec/models/'
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


def get_doc_similarities(doc2vec_model, document_name):
    logger.info('get document similarities')
    similarities = doc2vec_model.docvecs.most_similar(document_name)
    return similarities


def load_existing_model(model_name):
    models_file_path = settings.DATA_DIR + 'doc2vec/models/'
    logger.info('load model from following path: ' + models_file_path)

    loaded_model = gensim.models.Doc2Vec.load(models_file_path + model_name)
    return loaded_model

# TODO Renaming this function


def create_new_doc2vec_model():
    logger.info('Start creating new doc2vec model...')
    crawling_data_file_path = settings.PROJECT_ROOT + '/data/crawling_data/data_raw/*/'
    utils.save_all_documents(crawling_data_file_path)

    documents_file_path = settings.PROJECT_ROOT + '/data/doc2vec/'
    document_corpus_english, document_corpus_german = create_document_corpus_by_language(documents_file_path)

    doc2vec_model_english = create_doc2vec_model(document_corpus_english)
    doc2vec_model_german = create_doc2vec_model(document_corpus_german)

    save_doc2vec_model(doc2vec_model_english, 'doc2vec-model-english')
    save_doc2vec_model(doc2vec_model_german, 'doc2vec-model-german')

    return doc2vec_model_english, doc2vec_model_german


def get_most_similar_doc_matrix(doc2vec_model):
    similarities_matrix = []
    for document_name in doc2vec_model.docvecs.doctags:
        similarities_matrix.append(get_doc_similarities(doc2vec_model, document_name))

    similarities_matrix = np.asarray(similarities_matrix)

    return similarities_matrix


def main():
    #create_new_doc2vec_model()

    # get doc2vec similarities
    doc2vec_model = load_existing_model('doc2vec-model-english-28-Nov-2017-13:41:32')
    print(get_doc_similarities(doc2vec_model, 'www.booking.com.txt'))


if __name__ == "__main__":
    main()