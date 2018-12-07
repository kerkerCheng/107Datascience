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
from keras.optimizers import Adam, SGD
import time
import datetime
import os
import sys
import cv2


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
    model_name = 'a071547.hdf5'

    # Image augmentation #
    data_gen = image.ImageDataGenerator(rotation_range=15,
                                        shear_range=0.08,
                                        width_shift_range=0.08,
                                        height_shift_range=0.08,
                                        horizontal_flip=True)
    data_gen.fit(X_train, augment=True)
    img_generator = data_gen.flow(X_train, y_train, batch_size=batch_size)

    # Callbacks #
    early_stop = EarlyStopping(monitor='val_acc', patience=patience)
    model_checkpoint = ModelCheckpoint(model_name,
                                       monitor='val_acc',
                                       save_best_only=True,
                                       verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.75, patience=8, verbose=1)
    call_back = [early_stop, model_checkpoint, reduce_lr]

    # Train #
    classifier = model_2()
    classifier.fit_generator(img_generator,
                             epochs=num_epo,
                             verbose=verbose,
                             validation_data=(X_val, y_val),
                             callbacks=call_back)

    ########################################

    model_path = './' + model_name
    test_path = os.path.abspath('./test.csv')
    test_arr = np.genfromtxt(test_path, delimiter=',', dtype='str', skip_header=1)
    output_path = 'answer.csv'

    X = np.zeros((test_arr.shape[0], 28 * 28))
    for i in range(0, test_arr.shape[0]):
        X[i][:] = test_arr[i][1:]

    X = X.reshape((test_arr.shape[0], 28, 28, 1))
    X = X / 255.0

    classifier = load_model(model_path)

    ans_proba = classifier.predict(X, batch_size=256, verbose=1)
    ans_class = ans_proba.argmax(axis=-1)

    with open(output_path, 'w+', encoding='utf-8') as f:
        f.write('id,label' + '\n')
        for i in range(len(ans_class)):
            f.write(str(i) + ',' + str(int(ans_class[i])) + '\n')
        f.close()


if __name__ == "__main__":
    main()
