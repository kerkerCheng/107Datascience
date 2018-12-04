from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, Input
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D, Add
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.initializers import glorot_normal
import numpy as np


def build(l2_regularization=0.01, number_of_classes=7):
    model = Sequential()
    regulizer = regularizers.l2(l2_regularization)

    # CNN part
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=number_of_classes, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    opt = Adam(0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    return model


def output_history(his, time):
    with open('./logs/' + time + '/train_loss', 'w+') as f:
        f.writelines("%f\n" % i for i in his.train_loss)
    with open('./logs/' + time + '/validation_loss', 'w+') as f:
        f.writelines("%f\n" % i for i in his.validation_loss)
    with open('./logs/' + time + '/train_acc', 'w+') as f:
        f.writelines("%f\n" % i for i in his.train_acc)
    with open('./logs/' + time + '/validation_acc', 'w+') as f:
        f.writelines("%f\n" % i for i in his.validation_acc)


# def plot_acc(his):
#     x = np.arange(0, len(his.train_acc))
#     y1 = his.train_acc
#     y2 = his.validation_acc
#     plt.plot(x, y1)
#     plt.plot(x, y2)
#     plt.legend([' training acc ', ' validation acc '], fontsize=12)
#     plt.xlabel('num. of epochs', fontsize=12)
#     plt.ylabel('acc. (%)', fontsize=12)
#     plt.show()


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