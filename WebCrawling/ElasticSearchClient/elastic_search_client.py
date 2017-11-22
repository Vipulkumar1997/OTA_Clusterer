from elasticsearch import Elasticsearch
import json
import time
import random
import os
import errno

# constants definition
ELASTIC_SERVER = 'http://mse-2017-wbcilurz.el.eee.intern:9200'
RESPONSE_FILE_PATH = 'data'


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


def main():
    get_documents_of_host(hostname="www.agoda.com")


if __name__ == "__main__":
    main()
