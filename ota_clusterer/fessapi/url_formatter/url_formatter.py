import csv
import re
import pandas as pd
from ota_clusterer import settings
import time
import os
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_urls(urls_file_path):
    logger.info('start preparing urls')
    pattern = os.path.join(urls_file_path, '*.csv')

    file_names = glob.glob(pattern)

    if not file_names:
        raise IOError

    else:
        logger.info('Start preparing urls')
        for counter, file_name in enumerate(file_names):
            urls_list = []
            with open(file_name, "r") as file:
                csv_file = csv.reader(file, delimiter="\n")
                for row in csv_file:
                    urls_list.append(row)

            prepared_urls = []
            for url in urls_list:
                url = url[0]
                formatted_urls = check_url_format(url)
                prepared_urls.extend(formatted_urls)

            data_frame = pd.DataFrame(prepared_urls, columns=['http_url',
                                                              'https_url',
                                                              'http_url_wildcard',
                                                              'https_url_wildcard',
                                                              'www_url'])

            filename = "urls_prepared-" + str(counter) + "-" + time.strftime("%d-%b-%Y-%X") + '.csv'
            save_to_file_path = settings.DATA_DIR + 'prepared_urls/' + filename
            data_frame.to_csv(save_to_file_path, encoding='utf-8')


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
        url = url.replace(url_beginning, '')
        url = url.strip('/')

    return url


def get_url_with_slash(url):
    url_with_slash = "".join((url, "/"))
    return url_with_slash


def get_wildcard_url(url):
    url_wildcard = "".join((url, ".*"))
    return url_wildcard
