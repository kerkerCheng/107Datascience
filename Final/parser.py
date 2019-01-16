from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import time


tmp = []
ans_list = []
df = pd.read_csv('shot_logs.csv')
df = df[['player_name']].drop_duplicates()
names = df.values
for i in range(names.shape[0]):
    tmp.append(names[i][0].replace(',', '')
               .replace('nowtizski', 'nowitzki')
               .replace('jon ingles', 'joe ingles')
               .replace('danilo gallinai', 'danilo gallinari')
               .replace('mnta ellis', 'monta ellis')
               .replace('time hardaway jr', 'tim hardaway jr')
               .replace('alan crabbe', 'allen crabbe')
               .replace('nerles noel', 'nerlens noel')
               .replace('beno urdih', 'beno udrih')
               .replace('jimmer dredette', 'jimmer fredette')
               .lower())
player_list = list(set(tmp))

for player in player_list:
    url = "https://www.basketball-reference.com/search/search.fcgi?search=" + player
    r = requests.get(url, stream=True)

    while r.status_code != 200:
        print("http request didn't complete, status code is " + str(r.status_code) + '.')
        print("retry after 1 second.")
        time.sleep(1)
        r = requests.get(url, stream=True)

    if r.status_code == 200:
        # print("http request completed, URL=" + url)
        content = r.text
        soup = BeautifulSoup(content, 'html.parser')

        if soup.find('div', {'class': 'players', 'id': 'info'}) == None:
            url_new = 'https://www.basketball-reference.com' + \
                      soup.find('div', {'class': 'search-item-name'}).find('a')['href']
            r_new = requests.get(url_new, stream=True)
            while r_new.status_code != 200:
                print("http request didn't complete, status code is " + str(r.status_code) + '.')
                print("retry after 1 second.")
                time.sleep(1)
                r_new = requests.get(url_new, stream=True)

            if r_new.status_code == 200:
                # print("http request completed, URL=" + url_new)
                content = r_new.text
                soup = BeautifulSoup(content, 'html.parser')
                ans_parts = soup.find('div', {'class': 'players', 'id': 'info'}).find_all('strong')
                for i in ans_parts:
                    if "Position:" in i.text:
                        tu = (player,
                              i.next_sibling.replace('\n', '').replace(' and ', ',').replace(' ', '').replace('▪', ''))
                        print(tu[0] + ": " + tu[1])
                        ans_list.append(tu)

        else:
            ans_parts = soup.find('div', {'class': 'players', 'id': 'info'}).find_all('strong')
            for i in ans_parts:
                if "Position:" in i.text:
                    tu = (player, i.next_sibling.replace('\n', '').replace(' and ', ',').replace(' ', '').replace('▪', ''))
                    print(tu[0] + ": " + tu[1])
                    ans_list.append(tu)

