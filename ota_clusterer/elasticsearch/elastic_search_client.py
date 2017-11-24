from elasticsearch import Elasticsearch
import json
import time
import random
import os
from ota_clusterer import settings
import errno
import pandas as pd

# constants definition
ELASTIC_SERVER = 'http://mse-2017-wbcilurz.el.eee.intern:9200'
RESPONSE_FILE_PATH = settings.PROJECT_ROOT + '/data/crawling_data/'


def initialize_elasticsearch():
    es = Elasticsearch([ELASTIC_SERVER])
    return es


def create_response_file_directory(hostname):
    if not os.path.exists(hostname):
        try:
            os.makedirs(RESPONSE_FILE_PATH + '/' + hostname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def get_document_content():
    es = initialize_elasticsearch()
    result = es.search(index="fess.search",
                    body={
                        "query": {
                            "match_all": {

                            }
                        }
                    })

    print("Got %d Hits:" % result['hits']['total'])

    for entry in result['hits']['hits']:
        print(entry['_source']['content'])


def get_documents_of_host(hostname):
    es = initialize_elasticsearch()
    print("start fetching data from elasticsearch...")
    result = es.search(index="fess.search",
                       scroll='10m',
                       size=1000,
                       body={
                           "query": {

                               "constant_score": {

                                   "filter": {

                                       "term": {

                                           "host": hostname

                                       }

                                   }

                               }

                           }

                       })

    print("Got %d Hits:" % result['hits']['total'])
    print("start json dumping...")

    create_response_file_directory(hostname)

    for entry in result['hits']['hits']:
        filename = hostname + '_' + time.strftime("%Y%m%d") + str(random.randint(0, 1000)) + '_' + '.json'
        with open(RESPONSE_FILE_PATH + '/' + hostname + '/' + filename, 'w') as output:
            json.dump(entry['_source'], output)


def get_crawled_urls():
    file_path = settings.PROJECT_ROOT + '/data/prepared_urls/urls_prepared.csv'
    data_frame = pd.read_csv(file_path)
    return data_frame


def save_documents_of_crawled_hostnames():
    urls_data_frame = get_crawled_urls()
    for index, row in urls_data_frame.iterrows():
        hostname = row['www_url']
        get_documents_of_host(hostname)


def main():
    save_documents_of_crawled_hostnames()


if __name__ == "__main__":
    main()
