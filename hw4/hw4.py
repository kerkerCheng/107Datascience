import numpy as np
import matplotlib.pyplot as plt
import hw4_model as md
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing import image
import time
import datetime
import os
import sys
import cv2


def show_image(img):
    plt.imshow(img.reshape(28, 28), cmap='Greys')
    plt.show()


def main():
    which_model = sys.argv[1]

    csv_path = os.path.abspath('./train.csv')
    csv_arr = np.genfromtxt(csv_path, delimiter=',', dtype='str', skip_header=1)

    X = np.zeros((csv_arr.shape[0], 28*28))
    X_s = np.zeros((csv_arr.shape[0], 299, 299, 3))
    y = np.zeros(csv_arr.shape[0])
    for i in range(0, csv_arr.shape[0]):
        X[i][:] = csv_arr[i][1:]
        y[i] = csv_arr[i][0]

    X = X.reshape((csv_arr.shape[0], 28, 28, 1))

    for i in range(0, X_s.shape[0]):
        t = cv2.resize(X[i], dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        X_s[i] = cv2.merge((t, t, t))

    X = X/225.0         # Re-scale
    X_s = X_s/255.0
    y = np_utils.to_categorical(y)

    # Split Train and Validation #
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
    X_s_train, X_s_val, y_train, y_val = train_test_split(X_s, y, test_size=0.05, random_state=42)

    # Parameters #
    num_epo = 10000
    batch_size = 64
    patience = 20
    verbose = 2
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M')
    os.makedirs('./logs/'+timestamp)
    model_names = 'model_' + timestamp + '_{epoch:02d}_{val_acc:.2f}.hdf5'

    # Image augmentation #
    data_gen = image.ImageDataGenerator(rotation_range=15,
                                        shear_range=0.08,
                                        width_shift_range=0.08,
                                        height_shift_range=0.08,
                                        horizontal_flip=True)
    data_gen_s = image.ImageDataGenerator(rotation_range=15,
                                          shear_range=0.08,
                                          width_shift_range=0.08,
                                          height_shift_range=0.08,
                                          horizontal_flip=True)
    data_gen.fit(X_train, augment=True)
    data_gen_s.fit(X_s_train, augment=True)
    img_generator = data_gen.flow(X_train, y_train, batch_size=batch_size)
    img_s_generator = data_gen_s.flow(X_s_train, y_train, batch_size=batch_size)

    # Callbacks #
    early_stop = EarlyStopping(monitor='val_acc', patience=patience)
    hist = md.LossHistory()
    model_checkpoint = ModelCheckpoint('./logs/' + timestamp + '/' + model_names,
                                       monitor='val_acc',
                                       save_best_only=True,
                                       verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.75, patience=8, verbose=1)
    call_back = [hist, early_stop, model_checkpoint, reduce_lr]

    # Train #
    classifier = None
    if which_model == '1':
        classifier = md.build_Res()
    elif which_model == '2':
        classifier = md.model_2()
    elif which_model == '3':
        classifier = md.model_3()
    elif which_model == '4':
        classifier = md.model_4()

    if which_model != '4':
        classifier.fit_generator(img_generator,
                                 epochs=num_epo,
                                 verbose=verbose,
                                 validation_data=(X_val, y_val),
                                 callbacks=call_back)

        classifier.save('model_last_' + timestamp + '.hdf5')
        md.output_history(hist, timestamp)
        md.plot_acc(hist, timestamp)
    elif which_model == '4':
        classifier.fit_generator(img_s_generator,
                                 epochs=num_epo,
                                 verbose=verbose,
                                 validation_data=(X_s_val, y_val),
                                 callbacks=call_back)

        classifier.save('model_last_' + timestamp + '.hdf5')
        md.output_history(hist, timestamp)
        md.plot_acc(hist, timestamp)


if __name__ == "__main__":
    main()
