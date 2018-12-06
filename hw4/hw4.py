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
    data_gen.fit(X_train, augment=True)
    img_generator = data_gen.flow(X_train, y_train, batch_size=batch_size)

    # Callbacks #
    early_stop = EarlyStopping(monitor='val_acc', patience=patience)
    hist = md.LossHistory()
    model_checkpoint = ModelCheckpoint('./logs/' + timestamp + '/' + model_names,
                                       monitor='val_acc',
                                       save_best_only=True,
                                       verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.75, patience=4, verbose=1)
    call_back = [hist, early_stop, model_checkpoint, reduce_lr]

    # Train #
    classifier = md.model_2()
    classifier.fit_generator(img_generator,
                             epochs=num_epo,
                             verbose=verbose,
                             validation_data=(X_val, y_val),
                             callbacks=call_back)

    classifier.save('model_last_' + timestamp + '.hdf5')
    md.output_history(hist, timestamp)
    md.plot_acc(hist)


if __name__ == "__main__":
    main()
