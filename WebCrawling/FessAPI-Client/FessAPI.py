import csv
from selenium import webdriver
from .Utils import url_formatter


# constant definition
AUTH_URL = 'http://mse-2017-wbcilurz.el.eee.intern:8080/login/'


def initialize_webdriver():
    driver = webdriver.Firefox(executable_path='Geckodriver/geckodriver.exe')
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
        # include the parameters of the form
        crawler_name = entry
        element_name.send_keys(crawler_name)
        element_urls.send_keys(entry)
        element_urls_for_crawling.send_keys("https://booking.com")
        element_urls_for_indexing.send_keys("https://booking.com")
        element_depth.send_keys("10")
        element_max_access_count.send_keys("100")
        # submit the form
        authenticated_driver.find_element_by_name("create").click()

    authenticated_driver.close()


def get_prepared_urls():
    with open("URLS/urls_prepared.csv", "r") as file:
        urls_list = csv.reader(file, delimiter="\n")

    return urls_list


def main():
    url_formatter.prepare_urls()
    create_new_crawler()


if __name__ == "__main__":
    main()
