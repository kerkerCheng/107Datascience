import pandas as pd
import numpy as np
import re
import time
import datetime
import os
import matplotlib.pyplot as plt
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Input, Dropout, Embedding, GRU, LSTM, Flatten, Dense, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adadelta, Adam
from keras.models import Model


timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M')


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.validation_loss = []
        self.train_acc = []
        self.validation_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.validation_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.validation_acc.append(logs.get('val_acc'))


def output_history(his, time):
    with open('./logs/' + time + '/train_loss', 'w+') as f:
        f.writelines("%f\n" % i for i in his.train_loss)
    with open('./logs/' + time + '/validation_loss', 'w+') as f:
        f.writelines("%f\n" % i for i in his.validation_loss)
    with open('./logs/' + time + '/train_acc', 'w+') as f:
        f.writelines("%f\n" % i for i in his.train_acc)
    with open('./logs/' + time + '/validation_acc', 'w+') as f:
        f.writelines("%f\n" % i for i in his.validation_acc)


def plot_acc(his):
    x = np.arange(0, len(his.train_acc))
    y1 = his.train_acc
    y2 = his.validation_acc
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend([' training acc ', ' validation acc '], fontsize=12)
    plt.xlabel('num. of epochs', fontsize=12)
    plt.ylabel('acc. (%)', fontsize=12)
    plt.show()


def text_to_txt_file(X, X_test, txt_path='text_preprocessing.txt'):
    X = np.concatenate((X.values, X_test.values))
    ans = []
    for sentence in X:
        sentence = re.sub(r'http\S+', '', sentence)     # Remove url in the sentence
        sentence = re.sub(r'[\?\.\!\,]+(?=[\?\.\!\,])', '', sentence)       # Remove duplication of '?', '.', '!', ','

        # Remove triple dot, message waiting
        sentence = sentence.replace('\x85', ' ').replace('\x95', ' ').replace('\x99', ' ').replace('\xa0', ' ')
        ##

        ans.append(text_to_word_sequence(sentence, filters='"#$%&()*+-/<=>[\]^_`{|}~ \t@'))
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
                        f.write(words + ' ')
            f.write('\n')


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


def word2vec_training(preprocessing_path, max_length, model_path='word2vec_' + timestamp + '.model'):
    preprocessing_path = os.path.abspath(preprocessing_path)
    sentences = word2vec.LineSentence(preprocessing_path)
    model = word2vec.Word2Vec(sentences, size=max_length, sg=1, min_count=1)
    model.save(model_path)
    return model


def sentence_to_index_matrix(X, word2vec_model, paddingsize):
    index_mat = []
    X_mat = input_tokenize(X)

    for i in range(X_mat.shape[0]):
        sentece_index = []
        print("幹你娘老雞掰" + str(i))
        for ind, word in enumerate(X_mat[i]):
            sentece_index.append(word2vec_model.wv.vocab[word].index + 1)
        index_mat.append(sentece_index)

    return np.array(pad_sequences(np.array(index_mat), maxlen=paddingsize))


def RNN(maxlen, num_words, wordvec_dim, word2vec_model):

    def pretrained_embedding_matrix(word2vec_model, num_words, wordvec_dim):
        embedding_matrix = np.zeros((num_words + 1, wordvec_dim))
        for i in range(1, num_words+1):
            embedding_matrix[i] = word2vec_model.wv[word2vec_model.wv.index2word[i-1]]
        return embedding_matrix

    inputs = Input(shape=(maxlen,))

    embedding_matrix = pretrained_embedding_matrix(word2vec_model, num_words, wordvec_dim)
    embed_in = Embedding(num_words+1,
                         wordvec_dim,
                         weights=[embedding_matrix],
                         input_length=maxlen,
                         trainable=False)(inputs)

    # RNN #
    hid_size = 384
    RNN_output = GRU(hid_size, return_sequences=True, dropout=0.3)(embed_in)

    # DNN #
    outputs = Flatten()(RNN_output)
    outputs = Dense(hid_size//2, activation='relu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.3)(outputs)
    outputs = Dense(hid_size//4, activation='relu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.3)(outputs)
    outputs = Dense(1, activation='relu')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adadelta()
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print('model compiled')
    return model


input_df = pd.read_csv('train.csv', sep=',')
test_df = pd.read_csv('test.csv', sep=',')

X = input_df['text']
X_test = test_df['text']
Y = input_df['sentiment']

# Parameters #
word_vec_size = 128
sentence_max_len = 150
verbose = 1

text_to_txt_file(X, X_test)
word2vec_model = word2vec_training(preprocessing_path='text_preprocessing.txt',
                                   max_length=word_vec_size)
# word2vec_model = word2vec.Word2Vec.load('word2vec_2018.12.22_15.11.model')
num_words = len(word2vec_model.wv.vocab)
print('number of words = %d' % num_words)
X_index = sentence_to_index_matrix(X, word2vec_model, sentence_max_len)

# Split Data #
X_train, X_val, y_train, y_val = train_test_split(X_index, Y.values, test_size=0.05, random_state=42)

# Training Parameters #
num_epo = 10000
batch_size = 64
patience = 6

# Training #
os.makedirs('./logs/'+timestamp)
model_names = 'model_' + timestamp + '_{epoch:02d}_{val_acc:.2f}.hdf5'

hist = LossHistory()
early_stop = EarlyStopping(monitor='val_acc', patience=patience, verbose=1)
model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.75, patience=8, verbose=1)
call_back = [hist, early_stop, model_checkpoint]

RNN_model = RNN(sentence_max_len, num_words, word_vec_size, word2vec_model)
RNN_model.summary()
RNN_model.fit(X_train, y_train,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              epochs=num_epo,
              callbacks=call_back,
              verbose=1)
RNN_model.save(timestamp + '_last.hdf5')
