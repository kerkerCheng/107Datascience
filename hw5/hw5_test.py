import hw5
import os
import pandas as pd
from gensim.models import word2vec


def testing(X_test, word2vec_model, paddingsize, RNN_model_path, output_path):
    X_test_mat = hw5.sentence_to_index_matrix(X_test, word2vec_model, paddingsize)
    predictor = hw5.load_model(os.path.abspath(RNN_model_path))

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
    RNN_model_path = '123'
    output_path = RNN_model_path + '.csv'
    sentence_max_len = 150

    testing(X_test, word2vec_model, sentence_max_len, RNN_model_path, output_path)


if __name__ == '__main__':
    testing()