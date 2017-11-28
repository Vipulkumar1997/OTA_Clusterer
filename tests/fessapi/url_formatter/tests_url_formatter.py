import pytest
from ota_clusterer.fessapi.url_formatter import url_formatter


def test_get_formatted_url_list_for_http():
    url = 'http://www.test.ch'
    url_beginning = 'http://'
    formatted_url_list = url_formatter.get_formatted_url_list(url, url_beginning)

    assert formatted_url_list == ['http://www.test.ch/',
                                  'https://www.test.ch/',
                                  'http://www.test.ch/.*',
                                  'https://www.test.ch/.*',
                                  'www.test.ch']
