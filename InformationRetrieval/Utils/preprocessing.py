import json
import os
import glob
import pandas as pd
import re

def get_documents(document_path):

    pattern = os.path.join(document_path, '*.json')
    for file_name in glob.glob(pattern):
        with open(file_name) as file:
            file_document = json.load(file)

        content = file_document['content']
        title = file_document['title']
        important_content = file_document['important_content']

        content_cleaned = clean_text(str(content))
        print(content_cleaned)
        title_cleaned = clean_text(title)
        important_content_cleaned = clean_text(important_content)

        df = pd.DataFrame([content, title, important_content])

def clean_text(text):
    '''Remove all characters except letters'''
    clean = re.sub("[^a-zA-Z]"," ", text)
    words = clean.split()
    return words


def main():
    get_documents('/home/sandro/Dropbox/Study/Master of Science in Engineering/3. Semester/Vertiefungsarbeit 1/vm1-project-code/WebCrawling/PythonElasticSearchClient/Responses/www.agoda.com/')


if __name__ == "__main__":
    main()

