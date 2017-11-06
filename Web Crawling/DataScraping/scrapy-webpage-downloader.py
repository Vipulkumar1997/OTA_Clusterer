import scrapy
from scrapy.crawler import CrawlerProcess


class WebPageSpider(scrapy.Spider):
    name = "Webpage Downloader"

    start_urls = [
        "https://www.ab-in-den-urlaub.de/",
    ]

    def parse(self, response):
        filename = response.url.split("/")[2] + '.html'
        with open(filename, 'wb') as f:
            f.write(response.body)


process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

process.crawl(WebPageSpider)
process.start()
