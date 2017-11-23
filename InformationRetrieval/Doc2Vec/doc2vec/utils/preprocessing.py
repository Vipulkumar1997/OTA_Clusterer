import json
import os
import glob
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_documents_of_single_webpage(file_path_webpage):
    pattern = os.path.join(file_path_webpage, '*.json')

    cleaned_documents = []
    file_names = glob.glob(pattern)

    if not file_names:
        raise IOError

    else:
        for file_name in file_names:
            with open(file_name) as file:
                file_document = json.load(file)

            content = file_document['content']
            title = file_document['title']
            important_content = file_document['important_content']

            content_cleaned = clean_text(str(content))
            title_cleaned = clean_text(str(title))
            important_content_cleaned = clean_text(str(important_content))

            cleaned_documents.append(content_cleaned)
            cleaned_documents.append(title_cleaned)
            cleaned_documents.append(important_content_cleaned)

    return cleaned_documents


def clean_text(text):
    '''Remove all characters except letters'''
    clean = re.sub("[^a-zA-Z]", " ", text)
    words = clean.split()
    return words


def save_document(cleaned_documents, filename, file_path):
    final_document = ''
    for document_list in cleaned_documents:
        for entry in document_list:
            final_document += ' ' + entry

    with open(file_path + filename + ".txt", mode='w') as file:
        file.write(final_document)


def save_all_documents(directory_path_webpage_data):
    pattern = os.path.join(directory_path_webpage_data)
    folders_in_directory = glob.glob(pattern)

    if not folders_in_directory:
        raise IOError

    else:
        for folder_name in folders_in_directory:
            documents = get_documents_of_single_webpage(folder_name)
            save_document(documents, folder_name, '../data/')


def main():
    save_all_documents('../../../../WebCrawling/ElasticSearchClient/data/')


if __name__ == "__main__":
    main()
