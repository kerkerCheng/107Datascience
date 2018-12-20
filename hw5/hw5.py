import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import text_to_word_sequence


def text_to_txt_file(X, txt_path='text_preprocessing.txt'):
    X = X.values
    ans = []
    for sentence in X:
        sentence = re.sub(r'http\S+', '', sentence)     # Remove url in the sentence
        sentence = re.sub(r'[\?\.\!\,]+(?=[\?\.\!\,])', '', sentence)       # Remove duplication of '?', '.', '!', ','
        ans.append(text_to_word_sequence(sentence, filters='"#$%&()*+-/<=>[\]^_`{|}~ '))
    with open(txt_path, 'w+', encoding='utf-8') as f:
        for sentence in ans:
            for word in sentence:
                tmp = ''
                sub_sen = []
                for ch in word:
                    if ch != '!' and ch != ',' and ch != '?' and ch != '.':
                        tmp = tmp + ch
                    elif ch == '!':
                        if tmp == '':
                            sub_sen += ['!']
                        else:
                            sub_sen += [tmp, '!']
                        tmp = ''
                    elif ch == ',':
                        if tmp == '':
                            sub_sen += [',']
                        else:
                            sub_sen += [tmp, ',']
                        tmp = ''
                    elif ch == '?':
                        if tmp == '':
                            sub_sen += ['?']
                        else:
                            sub_sen += [tmp, '?']
                        tmp = ''
                    elif ch == '.':
                        if tmp == '':
                            sub_sen += ['.']
                        else:
                            sub_sen += [tmp, '.']
                        tmp = ''
                if tmp != '':
                    sub_sen.append(tmp)

                for words in sub_sen:
                    if '@' not in words:
                        f.write(words + ' ')
            f.write('\n')


input_df = pd.read_csv('train.csv', sep=',')

X = input_df['text']
Y = input_df['sentiment']

text_to_txt_file(X)
