#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from collections import Counter
from datetime import date
import bs4
import requests
import time
import os
import sys
import re

time_interval = 0.05


def to_date(s):
    return date(2017, int(s[:-2]), int(s[-2:]))


def load_text(filename):
    date_list = []
    title_list = []
    url_list = []

    with open(os.path.abspath(filename), 'r+', encoding='utf-8') as f:
        for line in f:
            date_list.append(line.split(',')[0])
            title_list.append(','.join(part for part in line.split(',')[1:-1]))
            url_list.append(line.split(',')[-1].replace('\n', ''))

    return date_list, title_list, url_list


def get_imgs(url):
    r = requests.get(url, stream=True)
    img_list = []

    while r.status_code != 200:
        print("http request didn't complete, status code is " + str(r.status_code) + '.')
        print("retry after 1 second.")
        time.sleep(1)
        r = requests.get(url, stream=True)

    if r.status_code == 200:
        print("http request completed, URL=" + url)
        content = r.text
        soup = BeautifulSoup(content, 'html.parser')

        for url in soup.find_all('a'):
            if '(' not in url.text[-4:] and ')' not in url.text[-4:]:
                if bool(re.search(url.text[-4:], '\.png|\.jpg|\.jpeg|\.gif', flags=re.IGNORECASE)):
                    if url.text != '':
                        img_list.append(url.text)

    return img_list


def crawl():
    domain = 'https://www.ptt.cc'
    homepage_url = 'https://www.ptt.cc/bbs/Beauty/index.html'
    date_list = []
    title_list = []
    url_list = []
    bomb_date_list = []
    bomb_title_list = []
    bomb_url_list = []
    ignore_url_list = ['https://www.ptt.cc/bbs/Beauty/M.1490936972.A.60D.html',
                       'https://www.ptt.cc/bbs/Beauty/M.1494776135.A.50A.html',
                       'https://www.ptt.cc/bbs/Beauty/M.1503194519.A.F4C.html',
                       'https://www.ptt.cc/bbs/Beauty/M.1504936945.A.313.html',
                       'https://www.ptt.cc/bbs/Beauty/M.1505973115.A.732.html',
                       'https://www.ptt.cc/bbs/Beauty/M.1507620395.A.27E.html',
                       'https://www.ptt.cc/bbs/Beauty/M.1510829546.A.D83.html',
                       'https://www.ptt.cc/bbs/Beauty/M.1512141143.A.D31.html']

    flag = False

    for i in range(1992, 2342):
        url = 'https://www.ptt.cc/bbs/Beauty/index' + str(i) + '.html'
        r = requests.get(url, stream=True)

        while r.status_code != 200:
            print("http request didn't complete, status code is " + str(r.status_code) + '.')
            print("retry after 1 second.")
            time.sleep(1)
            r = requests.get(url, stream=True)

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
                    art_good = ''
                    if article.find_all('span') != []:
                        art_good = article.find_all('span')[0].text

                    # print("flag = " + str(flag))
                    print('title:   ' + art_title)
                    # print('url:   ' + domain + art_url)
                    # print('date:   ' + art_date)

                    if (domain + art_url) in ignore_url_list:
                        continue

                    if art_date == '1/01':
                        flag = True

                    if flag is True:
                        if art_title[0:4] != '[公告]':
                            date_list.append(art_date.strip().replace('/', ''))
                            title_list.append(art_title)
                            url_list.append(domain + art_url)
                            if art_good == '爆':
                                bomb_date_list.append(art_date.strip().replace('/', ''))
                                bomb_title_list.append(art_title)
                                bomb_url_list.append(domain + art_url)

        time.sleep(time_interval)

    max_len = len(title_list)
    for i in range(max_len-1, 0, -1):
        if date_list[i] != '1231':
            title_list = title_list[:-1]
            date_list = date_list[:-1]
            url_list = url_list[:-1]
        elif date_list[i] == '1231':
            break

    max_len = len(bomb_title_list)
    for i in range(max_len-1, 0, -1):
        if bomb_date_list[i] != '1231':
            bomb_title_list = bomb_title_list[:-1]
            bomb_date_list = bomb_date_list[:-1]
            bomb_url_list = bomb_url_list[:-1]
        elif bomb_date_list[i] == '1231':
            break

    with open(os.path.abspath('all_articles.txt'), 'w+', encoding='utf-8') as f:
        for i in range(len(title_list)):
            f.write(date_list[i] + ',' + title_list[i] + ',' + url_list[i] + '\n')

    with open(os.path.abspath('all_popular.txt'), 'w+', encoding='utf-8') as f:
        for i in range(len(bomb_title_list)):
            f.write(bomb_date_list[i] + ',' + bomb_title_list[i] + ',' + bomb_url_list[i] + '\n')


def push(start, end):
    start_date = to_date(start)
    end_date = to_date(end)

    date_list, title_list, url_list = load_text('all_articles.txt')

    liker = []
    booer = []

    for i in range(len(title_list)):
        if start_date <= to_date(date_list[i]) <= end_date:
            url = url_list[i]
            r = requests.get(url, stream=True)

            while r.status_code != 200:
                print("http request didn't complete, status code is " + str(r.status_code) + '.')
                print("retry after 1 second.")
                time.sleep(1)
                r = requests.get(url, stream=True)

            if r.status_code == 200:
                print("http request completed, URL=" + url)
                content = r.text
                soup = BeautifulSoup(content, 'html.parser')
                reply_list = soup.find_all('div', {'class': 'push'})

                for reply in reply_list:
                    reply_id = reply.find('span', {'class': 'f3 hl push-userid'}).text
                    like_or_boo = None          # True:push ; False:boo

                    if reply.find('span', {'class': 'hl push-tag'}) is not None:
                        like_or_boo = True
                    elif reply.find('span', {'class': 'f1 hl push-tag'}) is not None:
                        if reply.find('span',  {'class': 'f1 hl push-tag'}).text.strip() == '噓':
                            like_or_boo = False

                    if like_or_boo is True:
                        liker.append(reply_id)
                    elif like_or_boo is False:
                        booer.append(reply_id)

            time.sleep(time_interval)

    liker_count = sorted(Counter(liker).items(), key=lambda t: (-t[1], t[0]), reverse=False)
    booer_count = sorted(Counter(booer).items(), key=lambda t: (-t[1], t[0]), reverse=False)
    num_of_like = len(liker)
    num_of_boo = len(booer)

    with open(os.path.abspath('push[' + start + '-' + end + '].txt'), 'w+', encoding='utf-8') as f:
        f.write('all like: ' + str(num_of_like) + '\n')
        f.write('all boo: ' + str(num_of_boo) + '\n')
        for i in range(10):
            f.write('like #' + str(i+1) + ': ' + liker_count[i][0] + ' ' + str(liker_count[i][1]) + '\n')
        for i in range(10):
            f.write('boo #' + str(i + 1) + ': ' + booer_count[i][0] + ' ' + str(booer_count[i][1]) + '\n')


def popular(start, end):
    start_date = to_date(start)
    end_date = to_date(end)
    img_list = []

    bomb_date_list, bomb_title_list, bomb_url_list = load_text('all_popular.txt')

    count = 0

    for i in range(len(bomb_title_list)):
        if start_date <= to_date(bomb_date_list[i]) <= end_date:
            count += 1
            url = bomb_url_list[i]
            img_list += get_imgs(url)
            time.sleep(time_interval)

    with open(os.path.abspath('popular[' + start + '-' + end + '].txt'), 'w+', encoding='utf-8') as f:
        f.write('number of popular articles: ' + str(count) + '\n')
        for img in img_list:
            f.write(img + '\n')


def keyword(start, end, keyword):
    start_date = to_date(start)
    end_date = to_date(end)
    date_list = []
    title_list = []
    url_list = []
    img_list = []
    all_keywords = []
    ans_urls = []

    with open(os.path.abspath('all_articles.txt'), 'r+', encoding='utf-8') as f:
        for line in f:
            date_list.append(line.split(',')[0])
            title_list.append(','.join(part for part in line.split(',')[1:-1]))
            url_list.append(line.split(',')[-1].replace('\n', ''))

    print('read file OK!!')

    for i in range(len(title_list)):
        if start_date <= to_date(date_list[i]) <= end_date:
            url = url_list[i]
            r = requests.get(url, stream=True)

            while r.status_code != 200:
                print("http request didn't complete, status code is " + str(r.status_code) + '.')
                print("retry after 1 second.")
                time.sleep(1)
                r = requests.get(url, stream=True)

            if r.status_code == 200:
                print("http request completed, URL=" + url)
                content = r.text
                soup = BeautifulSoup(content, 'html.parser')
                for part in soup.find_all('div', {'class': 'article-metaline'}):
                    all_keywords.append((part.find_all('span', {'class': 'article-meta-tag'})[0].text, url))
                    all_keywords.append((part.find_all('span', {'class': 'article-meta-value'})[0].text, url))
                for part in soup.find_all('div', {'class': 'article-metaline-right'}):
                    all_keywords.append((part.find_all('span', {'class': 'article-meta-tag'})[0].text, url))
                    all_keywords.append((part.find_all('span', {'class': 'article-meta-value'})[0].text, url))

                if soup.find_all('div', {'class': 'article-metaline'}) == []:
                    words = soup.find_all('div', {'id': 'main-content'})[0].text
                    words = words.split('--\n※ 發信站: 批踢踢實業坊(ptt.cc)')[0]
                    words = words.replace('\n', '')
                    all_keywords.append((words, url))
                    continue

                node = soup.find_all('div', {'class': 'article-metaline'})[-1].next_sibling
                last_node = soup.find_all('span', {'class': 'f2'})[0]
                while node != last_node:
                    if type(node) == bs4.element.NavigableString:
                        tmp = str(node).split('\n')
                        while '' in tmp:
                            tmp.remove('')
                        for j in range(len(tmp)):
                            tmp[j] = (tmp[j], url)
                        all_keywords += tmp
                    elif type(node) == bs4.element.Tag:
                        if node.text != '':
                            all_keywords.append((node.text, url))

                    node = node.next_sibling

            if all_keywords[-1][0] == '--':
                all_keywords[:] = all_keywords[:-1]

            time.sleep(time_interval)

    for item in all_keywords:
        if keyword in item[0]:
            ans_urls.append(item[1])

    ans_urls = list(set(ans_urls))

    print('Start to crawl images!!')

    for u in ans_urls:
        img_list += get_imgs(u)

    with open(os.path.abspath('keyword(' + keyword + ')[' + start + '-' + end + '].txt'), 'w+', encoding='utf-8') as f:
        for im in img_list:
            f.write(im + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'crawl':
        crawl()
    elif sys.argv[1] == 'push':
        start = sys.argv[2]
        end = sys.argv[3]
        push(start, end)
    elif sys.argv[1] == 'popular':
        start = sys.argv[2]
        end = sys.argv[3]
        popular(start, end)
    elif sys.argv[1] == 'keyword':
        key = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
        keyword(start, end, key)
