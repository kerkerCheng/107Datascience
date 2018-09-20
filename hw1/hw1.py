#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import time
import csv
import os
import cProfile
import re


# def scrap():

time_interval = 0.05
domain = 'https://www.ptt.cc'
homepage_url = 'https://www.ptt.cc/bbs/Beauty/index.html'
date_list = []
title_list = []
url_list = []

last_date = ''
flag = False

for i in range(1992, 2342):
    url = 'https://www.ptt.cc/bbs/Beauty/index' + str(i) + '.html'
    r = requests.get(url, stream=True)

    while r.status_code != 200:
        print("http request didn't complete, status code is " + str(r.status_code) + '.')
        print("retry after 1 second.")
        time.sleep(1)
        r = requests.get(url)

    if r.status_code == 200:
        print("http request completed, URL=" + url)
        content = r.text
        soup = BeautifulSoup(content, 'html.parser')

        article_list = soup.find_all('div', {'class': 'r-ent'})
        # previous_page = soup.find_all('div', {'class': 'btn-group btn-group-paging'})
        # previous_page_url = (previous_page[0].find_all('a', {'class:', 'btn wide'})[1]).get('href')
        # previous_page_int = int(''.join(str(i) for i in re.findall('\d', previous_page_url)))

        for article in article_list:
            if not (not article.find_all('a')):
                obj = article.find_all('a')[0]
                art_title = obj.text
                art_url = obj.get('href')
                art_date = article.find_all('div', {'class': 'date'})[0].text.strip()

                # print("flag = " + str(flag))
                print('title:   ' + art_title)
                # print('url:   ' + domain + art_url)
                # print('date:   ' + art_date)

                # if flag is True and last_date == '12/31' and art_date != '12/31':
                #     flag = False
                if art_date == '1/01':
                    flag = True

                if flag is True:
                    if '[公告]' not in art_title:
                        date_list.append(art_date.strip().replace('/', ''))
                        title_list.append(art_title)
                        url_list.append(domain + art_url)

                last_date = art_date

    time.sleep(time_interval)

max_len = len(title_list)
for i in range(max_len-1, 0, -1):
    if date_list[i] != '1231':
        title_list = title_list[:-1]
        date_list = date_list[:-1]
        url_list = url_list[:-1]
    elif date_list[i] == '1231':
        break

with open(os.path.abspath('all_articles.txt'), 'w+', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for i in range(len(title_list)):
        writer.writerow([date_list[i], title_list[i], url_list[i]])
#
# if __name__ == '__main__':
#     scrap()
