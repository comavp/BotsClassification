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
registration_date_id = 'css-901oao css-16my406 r-m0bqgq r-4qtqp9 r-poiln3 r-1b7u577 r-bcqeeo r-qvutc0'
tweets_count_class_id = 'css-901oao css-bfa6kz r-m0bqgq r-1qd0xha r-n6v787 r-16dba41 r-1cwl3u0 r-bcqeeo r-qvutc0'
verified_class_id = 'css-901oao css-16my406 r-18u37iz r-1q142lx r-poiln3 r-bcqeeo r-qvutc0'
background_image_class_id = 'css-1dbjc4n r-1p0dtai r-1mlwlqe r-1d2f490 r-1udh08x r-u8s1d r-zchlnj r-ipm5af r-417010'


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
    registration_date_xpath = '//span[contains(text(), "Joined") or contains(text(), "Регистрация")]'
    tweets_count_xpath = '//div[@class="' + str(tweets_count_class_id) + '"]'
    use_background_image_xpath = '//div[@class="' + str(background_image_class_id) + '"]/img'
    use_default_profile_image = '//a[@href ="/' + username +'/photo"]//img[contains(@src, "default")]'
    verified_xpath = '//span[@class="' + str(verified_class_id) + '"]//*[@aria-label="Verified account" ' \
                                                                  'or @aria-label="Подлинная учетная запись"]'

    name = driver.find_element_by_xpath(name_xpath)
    screen_name = driver.find_element_by_xpath(screen_name_xpath)
    try:
        description = driver.find_element_by_xpath(description_xpath).text
    except Exception:
        description = ""
    followers = driver.find_element_by_xpath(followers_xpath)
    following = driver.find_element_by_xpath(following_xpath)
    registration_date = driver.find_element_by_xpath(registration_date_xpath)
    tweets_count = driver.find_element_by_xpath(tweets_count_xpath)
    is_verified = True
    is_use_background_image = True
    is_use_default_profile_image = True
    try:
        driver.find_element_by_xpath(verified_xpath)
    except Exception:
        is_verified = False
    try:
        driver.find_element_by_xpath(use_background_image_xpath)
    except Exception:
        is_use_background_image = False
    try:
        driver.find_element_by_xpath(use_default_profile_image)
    except Exception:
        is_use_default_profile_image = False

    print('name: ' + name.text)
    print('screen_name: ' + screen_name.text)
    print('description: ' + description)
    print('followers: ' + followers.text)
    print('following: ' + following.text)
    print('Registration date: "' + registration_date.text + '"')
    print('tweets: ' + tweets_count.text)
    print('is_verified: ' + str(is_verified))
    print('is_use_background_image: ' + str(is_use_background_image))
    print('is_use_default_profile_image: ' + str(is_use_default_profile_image))


driver = webdriver.Chrome()
username = 'thefivefifes@gmail.com'
password = '*(FHB(_*233hf89aphefiweufhOIHEIuhf9238h'
#login_to_twitter(username, password, driver)
username1 = 'fillpackart'
username2 = 'lspolegi'
username3 = 'ColdSiemens'
username4 = 'schwarzenegger'
username5 = 'anekdotru'
username6 = 'sfdsa'
get_user_data(username1, driver)
#driver.quit()


