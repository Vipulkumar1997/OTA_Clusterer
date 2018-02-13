import gensim.utils
from nltk.corpus import stopwords
from ota_clusterer import settings
from nltk.stem.snowball import SnowballStemmer
from ota_clusterer import logger

logger = logger.get_logger()
logger.name = __name__


def calculate_language_scores(document):
    """using stopwords to calculate language scores for each language in nltk stopwords corpus

    :param document: document to calculate language scores
    :return: language scores for each language
    """
    languages_scores = {}

    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(document)
        common_elements = words_set.intersection(stopwords_set)

        languages_scores[language] = len(common_elements)  # language "scoring"

    return languages_scores


def detect_language(document):
    """ detect language in given document
    :param document: document to analyze
    :return: most probable language

    """
    ratios = calculate_language_scores(document)
    most_rated_language = max(ratios, key=ratios.get)
    return most_rated_language


def stop_words_removal(document, document_language):
    """remove stopwords from given document

    :param document: document to clean stopwords
    :param document_language: used language in document
    :return: filtered document

    """

    stop_words = set(stopwords.words(document_language))

    filtered_document = []

    for word in document:
        if word not in stop_words:
            filtered_document.append(word)

    return filtered_document


def word_stemming(document, document_language):
    """using SnowBallStemmer to stemm words in document
    :param document: document to process
    :param document_language: used language in document
    :return: stemmed document

    """

    stemmer = SnowballStemmer(document_language)
    stemmed_document = []
    for word in document:
        stemmed_document.append(stemmer.stem(word))

    return stemmed_document


def preprocess_document(document, document_language):
    """perform document preprocessing (stop words removal and stemming)
    :param document: document to preprocess
    :param document_language: used language in document
    :return: preprocessed document

    """
    stop_words_cleaned_document = stop_words_removal(document, document_language)
    preprocessed_document = word_stemming(stop_words_cleaned_document, document_language)
    return preprocessed_document


if __name__ == '__main__':
    file_path = settings.DATA_DIR + '/crawling_data/evz.ch/www.evz.ch_en.txt'
    with open(file_path, 'r') as file:
        document = file.read()

    document = gensim.utils.simple_preprocess(document)
    language = detect_language(document)
    print(language)
