import csv
from selenium import webdriver
from Utils import url_formatter
import os


# constant definition
AUTH_URL = 'http://mse-2017-wbcilurz.el.eee.intern:8080/login/'


def initialize_webdriver():
    # windows driver
    if os.name == 'nt':
        driver = webdriver.Firefox(executable_path='Geckodriver/geckodriver.exe')

    # linux driver
    elif os.name == 'posix':
        driver = webdriver.Firefox(executable_path='Geckodriver/geckodriver')
        return driver


def get_authenticated_driver():
    driver = initialize_webdriver()

    driver.get(AUTH_URL)
    assert "Login" in driver.title
    element_username = driver.find_element_by_id("username")
    element_password = driver.find_element_by_name("password")

    element_username.send_keys("admin")
    element_password.send_keys("admin")
    driver.find_element_by_name("login").click()
    return driver


def create_new_crawler():
    authenticated_driver = get_authenticated_driver()
    authenticated_driver.get(url='http://mse-2017-wbcilurz.el.eee.intern:8080/admin/webconfig/createnew/')

    # get the form fields
    element_name = authenticated_driver.find_element_by_id("name")
    element_urls = authenticated_driver.find_element_by_name("urls")
    element_urls_for_crawling = authenticated_driver.find_element_by_id("includedUrls")
    element_urls_for_indexing = authenticated_driver.find_element_by_id("includedDocUrls")
    element_depth = authenticated_driver.find_element_by_id("depth")
    element_max_access_count = authenticated_driver.find_element_by_id("maxAccessCount")

    urls_list = get_prepared_urls()

    for entry in urls_list:
        # get the fields from the list (based on the csv file formatting)
        http_url = entry[0]
        https_url = entry[1]
        http_url_wildcard = entry[2]
        https_url_wildcard = entry[3]

        basis_urls = http_url + "\n" + https_url
        wildcard_urls = http_url_wildcard + "\n" + https_url_wildcard

        # send the elements to the selenium objects (fill out the form)
        element_name.send_keys(https_url)
        element_urls.send_keys(basis_urls)
        element_urls_for_crawling.send_keys(wildcard_urls)
        element_urls_for_indexing.send_keys(wildcard_urls)
        element_depth.send_keys("10")
        element_max_access_count.send_keys("100")
        # submit the form
        authenticated_driver.find_element_by_name("create").click()

    authenticated_driver.close()


def get_prepared_urls():
    urls_list = []
    with open("Urls/urls_prepared.csv", "r") as file:
        csv_file = csv.reader(file, delimiter=",")
        for row in csv_file:
            urls_list.append(row)

    return urls_list


def main():
    url_formatter.prepare_urls()
    #create_new_crawler()


if __name__ == "__main__":
    main()
