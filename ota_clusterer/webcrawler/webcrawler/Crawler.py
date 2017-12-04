from scrapy.crawler import CrawlerRunner
from ota_clusterer.webcrawler.webcrawler.spiders.WebCrawlingSpider import WebCrawlingSpider
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging
from twisted.internet import reactor
import csv
from ota_clusterer import settings


class Crawler:

    file_path = settings.DATA_DIR + 'urls/'

    def get_hostnames(self):
        hostnames = []
        with open(self.file_path + 'urls-to-crawl.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                hostname = row[0].replace('http://www.', '')
                hostname = hostname.replace('https://www.', '')
                hostnames.append(hostname)

        return hostnames

    def crawl_hostnames(self, hostnames):
        configure_logging()
        runner = CrawlerRunner(get_project_settings())
        for hostname in hostnames:
            runner.crawl(WebCrawlingSpider, hostname=hostname, start_urls=['http://' + hostname])
            d = runner.join()
            d.addBoth(lambda _: reactor.stop())
        reactor.run()


def main():
    crawler = Crawler()
    hostnames = crawler.get_hostnames()
    crawler.crawl_hostnames(hostnames)


main()
