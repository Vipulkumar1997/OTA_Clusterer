import json
import os
import glob
import re


def get_documents(document_path):
    pattern = os.path.join(document_path, '*.json')

    cleaned_documents = []
    for file_name in glob.glob(pattern):
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


def save_documents(cleaned_documents, file_path):
    final_document = ''
    for document_list in cleaned_documents:
        for entry in document_list:
            final_document += ' ' + entry

    with open(file_path + "cleaned_document.txt", mode='w') as file:
        file.write(final_document)


def clean_text(text):
    '''Remove all characters except letters'''
    clean = re.sub("[^a-zA-Z]", " ", text)
    words = clean.split()
    return words


def main():
    documents = get_documents(
        '../../WebCrawling/ElasticSearchClient/data/www.agoda.com/')
    save_documents(documents, '../data/')


if __name__ == "__main__":
    main()
