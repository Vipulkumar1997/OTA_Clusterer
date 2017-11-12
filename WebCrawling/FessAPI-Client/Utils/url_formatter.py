import csv
import re
import pandas as pd


def prepare_urls():
    urls_list = []
    with open("Urls/url_data_hotel-spider-part1.csv", "r") as file:
        csv_file = csv.reader(file, delimiter="\n")
        for row in csv_file:
            urls_list.append(row)

    prepared_urls = []
    for url in urls_list:
        url = url[0]
        formatted_urls = check_url_format(url)
        prepared_urls.extend(formatted_urls)

    data_frame = pd.DataFrame(prepared_urls, columns = ['http_url', 'https_url', 'http_url_wildcard', 'https_wildcard_url', 'www_url'])
    data_frame.to_csv('Urls/urls_prepared.csv', sep='\t', encoding='utf-8')

def check_url_format(url):
    urls = []
    if re.match("^http?://", string=url):
        formatted_http_urls = [get_formatted_url_list(url, "http://")]
        urls = formatted_http_urls

    elif re.match("^https?://", string=url):
        formatted_https_urls = [get_formatted_url_list(url, "https://")]
        urls = formatted_https_urls

    return urls


def get_formatted_url_list(url, url_beginning):
    formatted_url_list = []
    if not check_if_url_ends_with_slash(url):
        url = get_url_with_slash(url)

    if url_beginning == "http://":
        https_url = replace_url_beginning(url, "http", url_beginning)
        wild_card_http_url = get_wildcard_url(url)
        wild_card_https_url = get_wildcard_url(https_url)
        www_url = replace_url_beginning(url, "www", url_beginning)

        formatted_url_list.append(url)
        formatted_url_list.append(https_url)
        formatted_url_list.append(wild_card_http_url)
        formatted_url_list.append(wild_card_https_url)
        formatted_url_list.append(www_url)

    elif url_beginning == "https://":
        http_url = replace_url_beginning(url, "https", url_beginning)
        wild_card_https_url = get_wildcard_url(url)
        wild_card_http_url = get_wildcard_url(http_url)
        www_url = replace_url_beginning(url, "www", url_beginning)

        formatted_url_list.append(http_url)
        formatted_url_list.append(url)
        formatted_url_list.append(wild_card_http_url)
        formatted_url_list.append(wild_card_https_url)
        formatted_url_list.append(www_url)

    return formatted_url_list


def check_if_url_ends_with_slash(url):
    if re.search(r"/$", url):
        return True


def replace_url_beginning(url, option, url_beginning):
    if option == "http":
        url = url.replace(url_beginning, "https://")

    elif option == "https":
        url = url.replace(url_beginning, "http://")

    elif option == "www":
        url = url.strip(url_beginning)

    return url


def get_url_with_slash(url):
    url_with_slash = "".join((url, "/"))
    return url_with_slash


def get_wildcard_url(url):
    url_wildcard = "".join((url, ".*"))
    return url_wildcard
