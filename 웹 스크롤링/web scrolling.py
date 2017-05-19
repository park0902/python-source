# # -*- coding:utf-8 -*-
# from pyvirtualdisplay import Display
# from selenium import webdriver
# from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait  # available since 2.4.0
# from selenium.webdriver.support import expected_conditions as EC  # available since 2.26.0
# from selenium.webdriver.common.keys import Keys
# from selenium.common.exceptions import NoSuchElementException
# from datetime import datetime, timedelta, date
# import hashlib
# import json
# import time, random
# import math
#
#
# # 1일 최대 4000건 지정한 기간 만큼 수집
# class DaumBlogCrawler:
#     def __init__(self):
#         pass
#
#     def crawl(self):
#         # 리눅스에선 디스플레이 못하니까 눈에 안 보여도 firefox를 가상으로 창을 띄우겠다.
#         # 왜냐하면 curl 로 소스코드를 받으면 훨씬 빠르지만 ajax를 실행 못함.
#         display = Display(visible=0, size=(800, 600))
#         display.start()
#
#         binary = FirefoxBinary(firefox_path="/usr/bin/firefox")
#         driver = webdriver.Firefox(firefox_binary=binary)
#         # 여기까지가 그 얘기
#         print
#         "driver create"
#         keyword = "빅데이터"
#         url = "http://search.daum.net/search?w=news&nil_search=btn&enc=utf8&cluster=n&cluster_page=1&q=" + keyword + "&sort=1&period=u&sd=20150101000000&ed=20150301235959&page=1&DA=STC"
#         driver.get(url)
#         print
#         "open web site"
#         nodes = driver.find_elements_by_xpath(".//*[@id='newsResultWrapper']/ul/li")
#         nodes2 = driver.find_elements_by_xpath(".//*[@id='newsResultUL']/li/div")
#         #  for node in nodes:
#         #   if nodes == driver.find_elements_by_xpath(".//*[@id='newsResultWrapper']/ul/li"):
#         #    print node.find_element_by_xpath("./div/div/div/a" + ".//*[@id='newsResultUL']/li/div/div/span").text
#         #   else:
#         #    print node.find_element_by_xpath(".//*[@id='newsResultUL']/li/div/div/span").text
#
#         # for node in nodes if nodes == driver.find_elements_by_xpath(".//*[@id='newsResultWrapper']/ul/li"):
#         #   print node.find_element_by_xpath("./div/div/div/a").text
#         # for node2 in nodes2 if nodes2 = driver.find_elements_by_xpath(".//*[@id='newsResultUL']/li/div"):
#         #   print node2.find_element_by_xpath("./div/span").text
#         for node in nodes:
#             try:
#                 print(node.find_element_by_xpath("./div/div/div/a").text)
#                 print(node.find_element_by_xpath("./div[2]/div/span").text)
#                 print(node.find_element_by_xpath("./div[2]/div/p").text)
#
#             except:
#                 print(node.find_element_by_xpath("./div/div/span[1]").text)
#                 print(node.find_element_by_xpath("./div/div/p").text)
#
#         driver.delete_all_cookies()
#         driver.quit()
#         display.stop()
#
#
# DaumBlogCrawler = DaumBlogCrawler()
# DaumBlogCrawler.crawl()




