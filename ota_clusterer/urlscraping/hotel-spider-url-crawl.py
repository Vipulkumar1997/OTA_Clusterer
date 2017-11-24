from bs4 import BeautifulSoup
import urllib.request
import csv
from ota_clusterer import settings


def get_hotel_spider_urls():

    response = urllib.request.urlopen('http://www.hotel-spider.com/template/print_channel_manager.html')
    soup = BeautifulSoup(response, from_encoding=response.info().get_param('charset'))

    url_data = []
    for link in soup.find_all('a', href=True):
        print(link['href'])
        url_data.append(link['href'])

    file_path = settings.PROJECT_ROOT + "/data/urls/url_data_hotel-spider.csv"
    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter="\n")
        writer.writerow(url_data)


def main():
    get_hotel_spider_urls()


if __name__ == "__main__":
    main()


