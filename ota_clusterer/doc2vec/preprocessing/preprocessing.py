from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from ota_clusterer import settings
import sys

# TODO Renaming all functions
def calculate_languages_ratios(document):
    languages_ratios = {}
    tokens = wordpunct_tokenize(document)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements)  # language "score"

    return languages_ratios


def detect_language(document):
    ratios = calculate_languages_ratios(document)

    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language


if __name__ == '__main__':
    input_document = settings.DATA_DIR + '/doc2vec/data-27112017/www.alr-aerospace.ch.txt'
    with open(input_document, 'r') as file:
        document = file.read()
    language = detect_language(document)
    print(language)