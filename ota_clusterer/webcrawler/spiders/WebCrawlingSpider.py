#!/usr/bin/env python

__author__ = 'Sandro Cilurzo'

import os
import errno
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
from bs4.element import Comment
from ota_clusterer import settings


class WebCrawlingSpider(CrawlSpider):
    """WebCrawlingSpider class is the blueprint for the crawling objects which do the web crawling job for a given
    hostname. One instance crawls the complete given domain, parses the results and persists all in a browser visible
    text information of the given webpage as .txt file.

    """

    name = 'webcrawler'
    hostname = ''
    allowed_domains = []
    start_urls = []
    RESPONSE_FILE_PATH = settings.DATA_DIR + 'crawling_data/'
    rules = [Rule(LinkExtractor(), callback='parse_page')]

    def __init__(self, hostname, start_urls, directory_path_to_save_results=None):
        self.hostname = hostname
        self.start_urls = start_urls
        if directory_path_to_save_results is not None:
            self.RESPONSE_FILE_PATH = directory_path_to_save_results

        self.create_data_directory()
        self.allowed_domains = [self.hostname, '*.' + self.hostname]
        super().__init__()

    def create_data_directory(self):
        """create for each hostname a director yto store results (data)

        """
        if not os.path.exists(self.RESPONSE_FILE_PATH + self.hostname):
            try:
                os.makedirs(self.RESPONSE_FILE_PATH + self. hostname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse_page(self, response):
        """parsing the html response and persist the extracted text
        :param response: html response from the spider

        """

        soup = BeautifulSoup(response.body, 'html.parser')
        visible_text = self.get_visible_text(soup)
        visible_text = visible_text.encode('utf-8')
        webpage_name = response.url.strip('https://')
        webpage_name = webpage_name.replace('/', '_')

        self.persist_webpage_text(webpage_name, visible_text)

    def persist_webpage_text(self, webpage_name, webpage_text):
        """ persisting the extracted text

        :param webpage_name: file name for persisting
        :param webpage_text: extracted text to persist

        """

        file_name = '%s.txt' % webpage_name
        file_path = self.RESPONSE_FILE_PATH + self.hostname + '/'
        with open(file_path + file_name, 'wb') as f:
            f.write(webpage_text)
            self.log('Saved file %s at %s' % (file_name, file_path))

    def get_visible_text(self, soup):
        """ extract all in a browser visible text from html response (BeautifulSoup object)
        :param soup: BeautifulSoup object
        :return: visible text


        """
        data = soup.findAll(text=True)
        visible_text = list(filter(self.check_if_text_is_visible, data))
        return u"\n".join(t.strip() for t in visible_text)

    def check_if_text_is_visible(self, element):
        """check if text is visible in browser
        :param element: html element
        :return: True or False if html element is visible or not

        """

        if element.parent.name in ['style', 'script', '[document]', 'head']:
            return False
        elif isinstance(element, Comment):
            return False
        return True
