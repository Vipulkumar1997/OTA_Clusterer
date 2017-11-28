from elasticsearch import Elasticsearch
import json
import time
import os
from ota_clusterer import settings
import pandas as pd
import glob
from ota_clusterer import logger

logger = logger.get_logger()
ELASTIC_SERVER = 'http://mse-2017-wbcilurz.el.eee.intern:9200'
RESPONSE_FILE_PATH = settings.PROJECT_ROOT + '/data/crawling_data/'
PREPARED_HOSTNAMES_FILE_PATH = settings.PROJECT_ROOT + '/data/prepared_urls/'


class ElasticSearchClient:

    def __init__(self):
        self.api = Elasticsearch([ELASTIC_SERVER])

    def persist_hostnames(self):
        hostnames = self.get_hostnames(self.get_file_paths())
        for hostname in hostnames:
            documents = self.get_documents_for_hostname(hostname)
            self.persist_hostname_documents(hostname, documents)

    def get_hostnames(self, file_paths):
        hostnames = []
        for file_name in file_paths:
            urls_data_frame = pd.read_csv(file_name)
            for index, row in urls_data_frame.iterrows():
                hostname = row['www_url']
                hostnames.append(hostname)

        return hostnames

    def get_file_paths(self):
        logger.info('get the file paths of the csv files')
        file_pattern = os.path.join(PREPARED_HOSTNAMES_FILE_PATH, '*.csv')
        file_paths = glob.glob(file_pattern)
        assert len(file_paths) > 0
        return file_paths

    def get_documents_for_hostname(self, hostname):
        logger.info("start fetching data from elasticsearch...from following host:" + hostname)
        result = self.api.search(index="fess.search",
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

        logger.info("Got %d Hits:" % result['hits']['total'])
        return result['hits']['hits']

    def persist_hostname_documents(self, hostname, documents):
        os.makedirs(RESPONSE_FILE_PATH + '/' + hostname)
        for document in documents:
            filename = hostname + '_' + time.strftime("%d-%b-%Y-%X") + '_' + '.json'
            with open(RESPONSE_FILE_PATH + '/' + hostname + '/' + filename, 'w') as json_file:
                json.dump(document['_source'], json_file)


def main():
    elastic_search_client = ElasticSearchClient()
    elastic_search_client.persist_hostnames()


if __name__ == "__main__":
    main()
