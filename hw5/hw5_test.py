import os
import pandas as pd
import numpy as np
import re
from gensim.models import word2vec
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def input_tokenize(X):
    X = X.values
    ans = []
    matrix = []
    for sentence in X:
        sentence = re.sub(r'http\S+', '', sentence)     # Remove url in the sentence
        sentence = re.sub(r'[\?\.\!\,]+(?=[\?\.\!\,])', '', sentence)       # Remove duplication of '?', '.', '!', ','

        # Remove triple dot, message waiting
        sentence = sentence.replace('\x85', ' ').replace('\x95', ' ').replace('\x99', ' ').replace('\xa0', ' ')
        ##

        ans.append(text_to_word_sequence(sentence, filters='"#$%&()*+-/<=>[\]^_`{|}~ \t@'))
    for sentence in ans:
        sent = []
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
                    sent.append(words)
        matrix.append(sent)
    return np.array(matrix)

def sentence_to_index_matrix(X, word2vec_model, paddingsize):
    index_mat = []
    X_mat = input_tokenize(X)

    for i in range(X_mat.shape[0]):
        sentece_index = []
        print(str(i))
        for ind, word in enumerate(X_mat[i]):
            sentece_index.append(word2vec_model.wv.vocab[word].index + 1)
        index_mat.append(sentece_index)

    return np.array(pad_sequences(np.array(index_mat), maxlen=paddingsize))


def testing(X_test, word2vec_model, paddingsize, RNN_model_path, output_path):
    X_test_mat = sentence_to_index_matrix(X_test, word2vec_model, paddingsize)
    predictor = load_model(os.path.abspath(RNN_model_path))

    y_test_proba = predictor.predict(X_test_mat, verbose=1, batch_size=1024).squeeze()
    y = y_test_proba*4.0

    with open(os.path.abspath(output_path), 'w+') as f:
        f.write('ID,Sentiment\n')
        for ind, sent in enumerate(y):
            f.write(str(ind) + ',' + str(sent) + '\n')
        f.close()


def main():
    test_df = pd.read_csv('test.csv', sep=',')
    X_test = test_df['text']

    word2vec_model = word2vec.Word2Vec.load('word2vec_2018.12.22_15.36.model')
    RNN_model_path = 'model_2018.12.24_09.38_55_0.11.hdf5'
    output_path = RNN_model_path + '.csv'
    sentence_max_len = 150

    testing(X_test, word2vec_model, sentence_max_len, RNN_model_path, output_path)


if __name__ == '__main__':
    main()