from scrapy.crawler import CrawlerRunner
from ota_clusterer.webcrawler.spiders.WebCrawlingSpider import WebCrawlingSpider
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging
import ota_clusterer.webcrawler.settings as scrapy_settings
from twisted.internet import reactor
import csv
from ota_clusterer import settings
from ota_clusterer import logger

logger = logger.get_logger()
logger.name = __name__


class Crawler:

    def get_hostnames_from_list(self, file_path):
        hostnames = []
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                hostname = row[0].replace('http://www.', '')
                hostname = hostname.replace('https://www.', '')
                hostname = hostname.strip('/')
                hostnames.append(hostname)

        return hostnames

    def crawl_hostnames(self, hostnames, file_path):
        configure_logging()
        runner = CrawlerRunner(get_project_settings())
        for hostname in hostnames:
            runner.crawl(WebCrawlingSpider,
                         hostname=hostname,
                         start_urls=['http://' + hostname, 'https://' + hostname],
                         directory_path_to_save_results=file_path)
            d = runner.join()
            d.addBoth(lambda _: reactor.stop())
        reactor.run()

    def set_obey_robotstxt_false(self):
        scrapy_settings.ROBOTSTXT_OBEY = False


def crawl_given_urls(urls, directory_to_save_results):
    crawler = Crawler()
    logger.info('crawl following urls (hostnames): ' + str(urls))
    crawler.crawl_hostnames(urls, directory_to_save_results)


def crawl_list_of_hostnames(urls_list_file_path, directory_to_save_results):
    crawler = Crawler()
    logger.info('crawl following list of urls: ' + urls_list_file_path)
    hostnames = crawler.get_hostnames_from_list(urls_list_file_path)
    crawler.crawl_hostnames(hostnames, directory_to_save_results)


def main():
    #Example to crawl list of urls
    urls_list = settings.DATA_DIR + 'urls/urls-to-crawl.csv'
    directory_to_save_results = settings.DATA_DIR + 'crawling_data/'
    crawl_list_of_hostnames(urls_list_file_path=urls_list, directory_to_save_results=directory_to_save_results)

    #Example to crawl just specific urls
    hostnames = ['www.meissenberg.ch', 'www.klinik-zugersee.ch']
    crawl_given_urls(hostnames, directory_to_save_results)


if __name__ == "__main__":
    main()
