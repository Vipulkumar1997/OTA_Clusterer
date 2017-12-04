from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from ota_clusterer import settings
from nltk.stem.snowball import SnowballStemmer
from ota_clusterer import logger

logger = logger.get_logger()


# TODO Renaming all functions
def calculate_languages_ratios(document):
    languages_ratios = {}

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(document)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements)  # language "score"

    return languages_ratios


def detect_language(document):
    ratios = calculate_languages_ratios(document)

    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language


def stop_words_removal(document, document_language):
    stop_words = set(stopwords.words(document_language))

    filtered_document = []

    for word in document:
        if word not in stop_words:
            filtered_document.append(word)

    return filtered_document


def word_stemming(document, document_language):
   stemmer=  SnowballStemmer(document_language)
   stemmed_document = []
   for word in document:
       stemmed_document.append(stemmer.stem(word))

   return stemmed_document


def preprocess_document(document, document_language):
    document_language = detect_language(document)
    filtered_document = stop_words_removal(document, document_language)
    preprocessed_document = word_stemming(filtered_document, document_language)
    return preprocessed_document, document_language


if __name__ == '__main__':
    input_document = settings.DATA_DIR + '/doc2vec/data-27112017/www.alr-aerospace.ch.txt'
    with open(input_document, 'r') as file:
        document = file.read()
    language = detect_language(document)
    print(language)