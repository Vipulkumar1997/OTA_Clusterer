import gensim
import os
import glob
import logging
from ota_clusterer import settings
from ota_clusterer.doc2vec.preprocessing import preprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_document_corpus(document_path):
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


def get_doc2vec_model(document_corpus):
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


def create_word_vector_matrix(model):
    all_word_vectors_matrix = model.wv.syn0
    return all_word_vectors_matrix


def load_existing_model(model_name):
    models_file_path = settings.DATA_DIR + 'doc2vec/models/'
    logger.info('load model from following path: ' + models_file_path)

    loaded_model = gensim.models.Doc2Vec.load(models_file_path + model_name)
    return loaded_model


def main():
    crawling_data_file_path = settings.PROJECT_ROOT + '/data/crawling_data/*/'
    preprocessing.save_all_documents(crawling_data_file_path)

    documents_file_path = settings.PROJECT_ROOT + '/data/doc2vec/'
    document_corpus = get_document_corpus(document_path=documents_file_path)
    doc2vec_model = get_doc2vec_model(document_corpus)

    doc2vec_model_path = settings.PROJECT_ROOT + '/data/doc2vec/models/'
    doc2vec_model.save(doc2vec_model_path + 'doc2vec_model')

    return doc2vec_model


if __name__ == "__main__":
    main()