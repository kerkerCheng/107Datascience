#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import time
import re

time_interval = 0.05
domain = 'https://www.ptt.cc'
homepage_url = 'https://www.ptt.cc/bbs/Beauty/index.html'
date_list = []
title_list = []
url_list = []

current_date = ''
flag = False

for i in range(1992, 2341):
    url = 'https://www.ptt.cc/bbs/Beauty/index' + str(i) + '.html'
    r = requests.get(url)

    while r.status_code is not 200:
        print("http request didn't complete, status code is " + str(r.status_code) + '.')
        print("retry after 1 second.")
        time.sleep(1)
        r = requests.get(url)

    if r.status_code is 200:
        print("http request completed, URL=" + url)
        content = r.text
        soup = BeautifulSoup(content, 'html.parser')

        article_list = soup.find_all('div', {'class': 'r-ent'})
        # previous_page = soup.find_all('div', {'class': 'btn-group btn-group-paging'})
        # previous_page_url = (previous_page[0].find_all('a', {'class:', 'btn wide'})[1]).get('href')
        # previous_page_int = int(''.join(str(i) for i in re.findall('\d', previous_page_url)))

        for article in article_list:
            obj = article.find_all('a')[0]
            art_title = obj.string
            art_url = obj.get('href')
            art_date = article.find_all('div', {'class': 'date'})[0].string

            if art_date is '1/01':
                flag = True
            elif art_date is '12/31' and flag is True:
                flag = False

            if flag is True:
                if art_title[0:4] != '[公告]':
                    date_list.append(art_date.replace('/', ''))
                    title_list.append(art_title)
                    url_list.append(domain + art_url)

            # print('title:   ' + art_title)
            # print('url:   ' + domain + art_url)
            # print('date:   ' + art_date)

    time.sleep(time_interval)
