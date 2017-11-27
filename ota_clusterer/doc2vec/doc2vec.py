import gensim
import os
import glob
import logging
from ota_clusterer import settings
from ota_clusterer.doc2vec.preprocessing import preprocessing
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_document_corpus(document_path):
    logger.info('start getting document corpus')
    pattern = os.path.join(document_path, '*.txt')

    document_corpus = []
    file_names = glob.glob(pattern)

    if not file_names:
        raise IOError

    else:
        logger.info('Start read in files')
        for file_name in file_names:
            logger.debug('File Names: ', file_names)
            with open(file_name, 'r') as file:
                document = file.read()

            preprocessed_document = gensim.utils.simple_preprocess(document)
            tagged_document_name = preprocessing.cleaning_path_out_of_file_name(file_name)
            tagged_document = gensim.models.doc2vec.TaggedDocument(preprocessed_document, ["{}".format(tagged_document_name)])
            document_corpus.append(tagged_document)

    return document_corpus


def create_doc2vec_model(document_corpus):
    logger.info('start building Doc2Vec model')
    model = gensim.models.Doc2Vec(size=300,
                                  min_count=3,
                                  iter=100,
                                  )

    model.build_vocab(document_corpus)
    logger.info("model's vocubulary length: " + str(len(model.wv.vocab)))

    logger.info("start to train the model")
    model.train(document_corpus,
                total_examples=model.corpus_count,
                epochs=model.iter)

    return model


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


def create_new_doc2vec_model():
    logger.info('Start creating new doc2vec model...')
    crawling_data_file_path = settings.PROJECT_ROOT + '/data/crawling_data/*/'
    preprocessing.save_all_documents(crawling_data_file_path)

    documents_file_path = settings.PROJECT_ROOT + '/data/doc2vec/'
    document_corpus = create_document_corpus(document_path=documents_file_path)
    doc2vec_model = create_doc2vec_model(document_corpus)

    doc2vec_model_path = settings.PROJECT_ROOT + '/data/doc2vec/models/'
    filename = 'doc2vecmodel' + "-" + time.strftime("%d-%b-%Y-%X")
    logger.info("save new doc2vec model at: " + doc2vec_model_path + filename)
    doc2vec_model.save(doc2vec_model_path + filename)


def main():
    # create_new_doc2vec_model()

    # get doc2vec similarities
    doc2vec_model = load_existing_model('doc2vecmodel-27-Nov-2017-14:43:10')
    print(get_doc_similarities(doc2vec_model, 'www.cardinalhealth.com.txt'))

if __name__ == "__main__":
    main()