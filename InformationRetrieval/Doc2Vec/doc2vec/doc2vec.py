import gensim
import os
import glob
import logging

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
            tagged_document = gensim.models.doc2vec.TaggedDocument(preprocessed_document, ["{}".format(file_name)])
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


def main():
    document_corpus = get_document_corpus('data/')
    model = get_doc2vec_model(document_corpus)
    model.docvecs.most_similar()


if __name__ == "__main__":
    main()