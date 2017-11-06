from elasticsearch import Elasticsearch
import json

#Initialization
ELASTIC_SERVER = 'http://mse-2017-wbcilurz.el.eee.intern:9200'
es = Elasticsearch([ELASTIC_SERVER])


def get_document_content():
    res = es.search(index="fess.search", body={"query": {"match_all": {}}})
    print("Got %d Hits:" % res['hits']['total'])

    for results in res['hits']['hits']:
        print(results['_source']['content'])


def get_documents_of_host(hostname):
    result = es.search(index="fess.search", body={
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

    i = 0
    for entry in result['hits']['hits']:
        filename = hostname + '_' + str(i) + '.json'
        with open('Responses/' + filename, 'w') as output:
            json.dump(entry['_source'], output)
        i += 1


get_documents_of_host("www.agoda.com")