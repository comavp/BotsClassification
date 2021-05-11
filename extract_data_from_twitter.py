import re
import csv
import time
from getpass import getpass
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

twitter_path = 'http://www.twitter.com'
login_path = 'login'

name_class_id = 'css-901oao r-18jsvk2 r-1qd0xha r-adyw6z r-1vr29t4 r-135wba7 r-bcqeeo r-1udh08x r-qvutc0'
screen_name_class_id = 'css-901oao css-bfa6kz r-m0bqgq r-18u37iz r-1qd0xha r-a023e6 r-16dba41 r-rjixqe r-bcqeeo r-qvutc0'
followers_class_id = 'css-1dbjc4n r-1mf7evn'
following_class_id = 'css-1dbjc4n'


def login_to_twitter(username, password, driver):
    driver.get(twitter_path + '/' + login_path)
    username_input = driver.find_element_by_xpath('//input[@name="session[username_or_email]"]')
    username_input.send_keys(username)
    password_input = driver.find_element_by_xpath('//input[@name="session[password]"]')
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)


def get_user_data(username, driver):
    driver.get(twitter_path + '/' + username)
    sleep(5)

    name_xpath = '//div[@class="' + str(name_class_id) + '"]'
    screen_name_xpath = '//div[@class="' + str(screen_name_class_id) + '"]'
    description_xpath = '//div[@data-testid="UserDescription"]'
    followers_xpath = '//div[@class="' + str(followers_class_id) + '"]/a/span[1]'
    following_xpath = '//div[@class="' + str(following_class_id) + '"]/a/span[1]'

    name = driver.find_element_by_xpath(name_xpath)
    screen_name = driver.find_element_by_xpath(screen_name_xpath)
    description = driver.find_element_by_xpath(description_xpath)
    followers = driver.find_element_by_xpath(followers_xpath)
    following = driver.find_element_by_xpath(following_xpath)

    print(name.text)
    print(screen_name.text)
    print(description.text)
    print(followers.text)
    print(following.text)


driver = webdriver.Chrome()
# username = 'thefivefifes@gmail.com'
# password = '*(FHB(_*233hf89aphefiweufhOIHEIuhf9238h'
# login_to_twitter(username, password, driver)
username1 = 'fillpackart'
username2 = 'norimyxxxo'
get_user_data(username2, driver)
# driver.quit()


