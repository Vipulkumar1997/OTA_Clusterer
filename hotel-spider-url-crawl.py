from bs4 import BeautifulSoup
import urllib.request
import csv

response = urllib.request.urlopen('http://www.hotel-spider.com/template/print_channel_manager.html')
soup = BeautifulSoup(response, from_encoding=response.info().get_param('charset'))

url_data = []
for link in soup.find_all('a', href=True):
    print(link['href'])
    url_data.append(link['href'])

with open("url_data.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(url_data)


