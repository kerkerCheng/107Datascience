from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, Input
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D, Add
from keras.models import Model
from keras.layers import advanced_activations
from keras.optimizers import Adam, SGD
from keras.applications import inception_resnet_v2
from keras.regularizers import l2
from keras.initializers import glorot_normal
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def three_conv_bn_relu(nb_filters, kernel_size=3):
    filters = nb_filters
    kernel_initializer = 'he_normal'
    padding = 'same'

    def f(inputs):
        out = Conv2D(filters=filters[0],
                     kernel_size=(1, 1),
                     kernel_initializer=kernel_initializer,
                     padding=padding)(inputs)
        out = BatchNormalization()(out)
        out = Activation('selu')(out)

        out = Conv2D(filters=filters[1],
                     kernel_size=(kernel_size, kernel_size),
                     kernel_initializer=kernel_initializer,
                     padding=padding)(out)
        out = BatchNormalization()(out)
        out = Activation('selu')(out)

        out = Conv2D(filters=filters[2],
                     kernel_size=(1, 1),
                     kernel_initializer=kernel_initializer,
                     padding=padding)(out)
        out = Dropout(0.5)(out)

        out = Add()([out, inputs])
        out = BatchNormalization()(out)
        out = Activation('selu')(out)
        return out
    return f


def conv_block(nb_filters, kernel_size=3):
    filters = nb_filters
    kernel_initializer = 'he_normal'
    padding = 'same'

    def f(inputs):
        out = Conv2D(filters=filters[0],
                     kernel_size=(1, 1),
                     kernel_initializer=kernel_initializer,
                     padding=padding)(inputs)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv2D(filters=filters[1],
                     kernel_size=(kernel_size, kernel_size),
                     kernel_initializer=kernel_initializer,
                     padding=padding)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv2D(filters=filters[2],
                     kernel_size=(1, 1),
                     kernel_initializer=kernel_initializer,
                     padding=padding)(out)
        out = Dropout(0.5)(out)

        inputs = Conv2D(filters=filters[2],
                        kernel_size=(1, 1),
                        kernel_initializer=kernel_initializer,
                        padding=padding)(inputs)

        out = Add()([inputs, out])
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        return out
    return f


def conv_block2(nb_filters, kernel_size=3):
    filters = nb_filters
    kernel_initializer = 'he_normal'
    padding = 'same'

    def f(inputs):
        out = Conv2D(filters=filters[0],
                     kernel_size=(kernel_size, kernel_size),
                     padding=padding)(inputs)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv2D(filters=filters[1],
                     kernel_size=(kernel_size, kernel_size),
                     padding=padding)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Add()([inputs, out])
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        return out

    return f


def build_Res(number_of_classes=10):
    inputs = Input(shape=(28, 28, 1))
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters=128, kernel_size=(1, 1), padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = conv_block2((128, 128))(out)
    out = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = conv_block2((256, 256))(out)
    out = MaxPooling2D((2, 2))(out)
    out = Conv2D(filters=512, kernel_size=(1, 1), padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = conv_block2((512, 512))(out)
    out = AveragePooling2D((3, 3))(out)

    out = Flatten()(out)

    # out = Dense(256)(out)
    # out = advanced_activations.LeakyReLU(alpha=0.5)(out)
    # out = BatchNormalization()(out)
    out = Dense(number_of_classes, activation='softmax')(out)

    opt = Adam(0.01)
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    return model


def model_2(number_of_classes=10):
    inputs = Input(shape=(28, 28, 1))
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(inputs)
    out = BatchNormalization()(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.2)(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2), padding='same')(out)
    out = Dropout(0.25)(out)
    out = advanced_activations.PReLU()(out)

    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2), padding='same')(out)
    out = Dropout(0.3)(out)
    out = advanced_activations.PReLU()(out)

    out = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.4)(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=512, kernel_size=(5, 5), padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.4)(out)
    out = advanced_activations.PReLU()(out)

    out = Flatten()(out)

    out = Dense(512)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.3)(out)
    out = advanced_activations.LeakyReLU(alpha=0.5)(out)
    out = Dense(512)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.3)(out)
    out = advanced_activations.LeakyReLU(alpha=0.5)(out)
    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = advanced_activations.LeakyReLU(alpha=0.5)(out)
    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.6)(out)
    out = advanced_activations.LeakyReLU(alpha=0.5)(out)
    out = Dense(number_of_classes, activation='softmax')(out)

    model = Model(inputs=inputs, outputs=out)

    opt = Adam(0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    return model


def model_3(number_of_classes=10):
    inputs = Input(shape=(28, 28, 1))
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(inputs)
    out = BatchNormalization()(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.2)(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2), padding='same')(out)
    out = Dropout(0.25)(out)
    out = advanced_activations.PReLU()(out)

    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=192, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2), padding='same')(out)
    out = advanced_activations.PReLU()(out)

    out = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.4)(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.4)(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.4)(out)
    out = advanced_activations.PReLU()(out)

    out = Flatten()(out)

    out = Dense(512)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.3)(out)
    out = advanced_activations.LeakyReLU(alpha=0.4)(out)
    out = Dense(512)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.3)(out)
    out = advanced_activations.LeakyReLU(alpha=0.4)(out)
    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = advanced_activations.LeakyReLU(alpha=0.4)(out)
    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = Dropout(0.6)(out)
    out = advanced_activations.LeakyReLU(alpha=0.4)(out)
    out = Dense(number_of_classes, activation='softmax')(out)

    model = Model(inputs=inputs, outputs=out)

    opt = Adam(0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    return model


def build(number_of_classes=10):
    inputs = Input(shape=(28, 28, 1))

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


def plot_acc(his, timestamp):
    fig = plt.figure()
    x = np.arange(0, len(his.train_acc))
    y1 = his.train_acc
    y2 = his.validation_acc
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend([' training acc ', ' validation acc '], fontsize=12)
    plt.xlabel('num. of epochs', fontsize=12)
    plt.ylabel('acc. (%)', fontsize=12)
    plt.show()
    fig.savefig('./logs/' + timestamp + '/' + timestamp + '.png')


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
