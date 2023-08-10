from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# from bs4 import BeautifulSoup

import os

def scrape_site():
    # Initialize the driver
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    # driver = webdriver.Chrome(service=Service(
    #     ChromeDriverManager().install()), options=options)
    try:
        # Visit the site
        driver.get(
            "https://www.courtlistener.com/?q=camp+lejeune&type=r&order_by=dateFiled+desc&filed_after=01%2F01%2F2023&court=nced&page=1#")

        # Find the documents
        data = []
        while True:
            documents = WebDriverWait(driver, 5).until(EC.visibility_of_all_elements_located(
                (By.CSS_SELECTOR, '.col-md-offset-half .visitable')))

            count = len(documents)
            # count = 1
            for i in range(0, count):
                print("begin : " + str(i))
                print(documents[i].get_attribute("href"))
                # Click on the element
                if documents[i].get_attribute("href") == "":
                    print("back to")
                    continue
                documents[i].click()
                # Wait for the text to load
                try:
                    WebDriverWait(driver, 5).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, '#pdf')))
                except TimeoutException:
                    driver.back()
                    documents = WebDriverWait(driver, 5).until(EC.visibility_of_all_elements_located(
                        (By.CSS_SELECTOR, '.col-md-offset-half .visitable')))
                    continue
                print("middle")
                # Get the content
                content = driver.find_element(
                    By.CSS_SELECTOR, '#opinion-content pre').get_attribute('innerHTML')

                print(content)

                data.append(content)

                # Go back to the initial page
                driver.back()

                documents = WebDriverWait(driver, 5).until(EC.visibility_of_all_elements_located(
                    (By.CSS_SELECTOR, '.col-md-offset-half .visitable')))
                print("len = " + str(len(documents)), "i = " + str(i))
                print("end")

            try:
                nexts = WebDriverWait(driver, 5).until(EC.visibility_of_all_elements_located(
                    (By.CSS_SELECTOR, '.col-xs-3.text-right .btn.btn-default')))
                print("len of nexts" + str(len(nexts)))
                print(nexts[-1].get_attribute("rel"))
                nexts[-1].click()
            except TimeoutException:
                break

        return data

    finally:
        # Close the driver
        driver.quit()


# Use the function

def save_data(data: str):
    with open("./data/filename.txt", "w", encoding="utf-8") as file:
        for d in data:
            # Write the content to the file
            print(d, file=file)
