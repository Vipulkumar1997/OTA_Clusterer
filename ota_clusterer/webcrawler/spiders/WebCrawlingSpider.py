import os
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
from bs4.element import Comment
import errno
from ota_clusterer import settings


class WebCrawlingSpider(CrawlSpider):
    name = 'webcrawler'
    hostname = ''
    allowed_domains = []
    start_urls = []
    RESPONSE_FILE_PATH = settings.DATA_DIR + 'crawling_data/'
    rules = [Rule(LinkExtractor(), callback='parse_page')]

    def __init__(self, hostname=None, start_urls=None):
        self.hostname = hostname
        self.create_data_directory()
        self.allowed_domains = [self.hostname, '*.' + self.hostname]
        self.start_urls = start_urls
        super().__init__()

    def create_data_directory(self):
        if not os.path.exists(self.RESPONSE_FILE_PATH + self.hostname):
            try:
                os.makedirs(self.RESPONSE_FILE_PATH + self. hostname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse_page(self, response):
        soup = BeautifulSoup(response.body, 'html.parser')
        visible_text = self.get_visible_text(soup)
        visible_text = visible_text.encode('utf-8')
        webpage_name = response.url.strip('https://')
        webpage_name = webpage_name.replace('/', '_')

        self.persist_webpage_text(webpage_name, visible_text)

    def persist_webpage_text(self, webpage_name, webpage_text):
        file_name = '%s.txt' % webpage_name
        file_path = self.RESPONSE_FILE_PATH + self.hostname + '/'
        with open(file_path + file_name, 'wb') as f:
            f.write(webpage_text)
            self.log('Saved file %s at %s' % (file_name, file_path))

    def get_visible_text(self, soup):
        data = soup.findAll(text=True)
        visible_text = list(filter(self.check_if_text_is_visible, data))
        return u"\n".join(t.strip() for t in visible_text)

    def check_if_text_is_visible(self, element):
        if element.parent.name in ['style', 'script', '[document]', 'head']:
            return False
        elif isinstance(element, Comment):
            return False
        return True
