from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, Input
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D, Add
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.applications import inception_resnet_v2
from keras import regularizers
from keras.initializers import glorot_normal
import numpy as np
import matplotlib.pyplot as plt


def build(l2_regularization=0.01, number_of_classes=10):
    inputs = Input(shape=(28, 28, 1))
    regulizer = regularizers.l2(l2_regularization)

    # CNN part
    outputs = Conv2D(filters=8, kernel_size=(3, 3), padding='same')(inputs)
    outputs = Conv2D(filters=8, kernel_size=(3, 3), padding='same')(outputs)
    outputs = Activation('selu')(outputs)
    outputs = BatchNormalization()(outputs)

    outputs = Conv2D(filters=8, kernel_size=(3, 3), padding='same')(outputs)
    outputs = Conv2D(filters=8, kernel_size=(3, 3), padding='same')(outputs)
    outputs = Activation('selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2), padding='same')(outputs)

    #

    outputs = Conv2D(filters=16, kernel_size=(4, 4), padding='same')(outputs)
    outputs = Conv2D(filters=16, kernel_size=(4, 4), padding='same')(outputs)
    outputs = Activation('selu')(outputs)
    outputs = BatchNormalization()(outputs)

    outputs = Conv2D(filters=16, kernel_size=(4, 4), padding='same')(outputs)
    outputs = Conv2D(filters=16, kernel_size=(4, 4), padding='same')(outputs)
    outputs = Activation('selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2), padding='same')(outputs)

    #

    outputs = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(outputs)
    outputs = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(outputs)
    outputs = Activation('selu')(outputs)
    outputs = BatchNormalization()(outputs)

    outputs = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(outputs)
    outputs = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(outputs)
    outputs = Activation('selu')(outputs)
    outputs = BatchNormalization()(outputs)

    outputs = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(outputs)
    outputs = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(outputs)
    outputs = Activation('selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2), padding='same')(outputs)

    outputs = Flatten()(outputs)

    outputs = Dense(256, activation='relu')(outputs)
    outputs = Dense(64, activation='relu')(outputs)
    outputs = Dense(number_of_classes, activation='softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    opt = Adam(0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    return model


def output_history(his, timestamp):
    with open('./logs/' + timestamp + '/train_loss', 'w+') as f:
        f.writelines("%f\n" % i for i in his.train_loss)
    with open('./logs/' + timestamp + '/validation_loss', 'w+') as f:
        f.writelines("%f\n" % i for i in his.validation_loss)
    with open('./logs/' + timestamp + '/train_acc', 'w+') as f:
        f.writelines("%f\n" % i for i in his.train_acc)
    with open('./logs/' + timestamp + '/validation_acc', 'w+') as f:
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
