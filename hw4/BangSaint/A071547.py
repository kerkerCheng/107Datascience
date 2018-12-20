import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, Input
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D, Add
from keras.models import Model
from keras.layers import advanced_activations
from keras.optimizers import Adam, SGD, Adadelta
from scipy import stats
import time
import datetime
import os
import sys


def model_1(number_of_classes=10):
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

    opt = Adadelta()
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
    out = Dropout(0.4)(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.4)(out)
    out = advanced_activations.PReLU()(out)

    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = advanced_activations.PReLU()(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Dropout(0.4)(out)
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

    opt = Adadelta()
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


def show_image(img):
    plt.imshow(img.reshape(28, 28), cmap='Greys')
    plt.show()


def main():
    csv_path = os.path.abspath('./train.csv')
    csv_arr = np.genfromtxt(csv_path, delimiter=',', dtype='str', skip_header=1)

    X = np.zeros((csv_arr.shape[0], 28*28))
    y = np.zeros(csv_arr.shape[0])
    for i in range(0, csv_arr.shape[0]):
        X[i][:] = csv_arr[i][1:]
        y[i] = csv_arr[i][0]

    X = X.reshape((csv_arr.shape[0], 28, 28, 1))

    X = X/225.0         # Re-scale
    y = np_utils.to_categorical(y)

    # Split Train and Validation #
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

    # Parameters #
    num_epo = 10000
    batch_size = 32
    patience = 20
    verbose = 2

    # Image augmentation #
    data_gen = image.ImageDataGenerator(horizontal_flip=True,
                                        rotation_range=10,
                                        width_shift_range=0.05,
                                        height_shift_range=0.05)
    data_gen.fit(X_train, augment=True)
    img_generator = data_gen.flow(X_train, y_train, batch_size=batch_size)

    # Callbacks #
    early_stop = EarlyStopping(monitor='val_acc', patience=patience)
    model_checkpoint1 = ModelCheckpoint('A071547_model1.hdf5',
                                        monitor='val_acc',
                                        save_best_only=True,
                                        verbose=1)
    model_checkpoint2 = ModelCheckpoint('A071547_model2.hdf5',
                                        monitor='val_acc',
                                        save_best_only=True,
                                        verbose=1)
    model_checkpoint3 = ModelCheckpoint('A071547_model3.hdf5',
                                        monitor='val_acc',
                                        save_best_only=True,
                                        verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.75, patience=8, verbose=1)
    call_back1 = [early_stop, model_checkpoint1, reduce_lr]
    call_back2 = [early_stop, model_checkpoint2, reduce_lr]
    call_back3 = [early_stop, model_checkpoint3, reduce_lr]

    # Train #
    classifier1 = model_1()
    classifier1.fit_generator(img_generator,
                              epochs=num_epo,
                              verbose=verbose,
                              validation_data=(X_val, y_val),
                              callbacks=call_back1)

    classifier2 = model_2()
    classifier2.fit_generator(img_generator,
                              epochs=num_epo,
                              verbose=verbose,
                              validation_data=(X_val, y_val),
                              callbacks=call_back2)

    classifier3 = model_3()
    classifier3.fit_generator(img_generator,
                              epochs=num_epo,
                              verbose=verbose,
                              validation_data=(X_val, y_val),
                              callbacks=call_back3)

    ########################################

    model_path1 = './A071547_model1.hdf5'
    model_path2 = './A071547_model2.hdf5'
    model_path3 = './A071547_model3.hdf5'
    test_path = os.path.abspath('./test.csv')
    test_arr = np.genfromtxt(test_path, delimiter=',', dtype='str', skip_header=1)
    output_path = 'answer.csv'

    X = np.zeros((test_arr.shape[0], 28 * 28))
    for i in range(0, test_arr.shape[0]):
        X[i][:] = test_arr[i][1:]

    X = X.reshape((test_arr.shape[0], 28, 28, 1))
    X = X / 255.0

    num_of_test = test_arr.shape[0]

    classifier1 = load_model(model_path1)
    classifier2 = load_model(model_path2)
    classifier3 = load_model(model_path3)

    ans_proba1 = classifier1.predict(X, batch_size=256, verbose=1)
    ans_proba2 = classifier2.predict(X, batch_size=256, verbose=1)
    ans_proba3 = classifier3.predict(X, batch_size=256, verbose=1)

    ans_class1 = ans_proba1.argmax(axis=-1)
    ans_class2 = ans_proba2.argmax(axis=-1)
    ans_class3 = ans_proba3.argmax(axis=-1)

    ans_class = []
    for i in range(num_of_test):
        nominee = [ans_class1[i], ans_class2[i], ans_class3[i]]
        ans = stats.mode(nominee)[0][0]
        ans_class.append(ans)

    with open(output_path, 'w+', encoding='utf-8') as f:
        f.write('id,label' + '\n')
        for i in range(len(ans_class)):
            f.write(str(i) + ',' + str(int(ans_class[i])) + '\n')
        f.close()


if __name__ == "__main__":
    main()
