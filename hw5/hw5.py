import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import text_to_word_sequence


def text_to_txt_file(X, txt_path='text_preprocessing.txt'):
    ans = []
    for sentence in X:
        sentence = re.sub(r'http\S+', '', sentence)     # Remove url in the sentence
        ans.append(text_to_word_sequence(sentence, filters='"#$%&()*+-/<=>[\]^_`{|}~ '))
    with open(txt_path, 'w+', encoding='utf-8') as f:
        for sentence in ans:
            for word in sentence:
                if '@' not in word:
                    f.write(word + ' ')
            f.write('\n')


input_df = pd.read_csv('train.csv', sep=',')

X = input_df['text'].values
Y = input_df['sentiment'].values

# text_to_txt_file(X)
