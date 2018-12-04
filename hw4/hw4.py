import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from keras.utils import np_utils
# import model as md
# from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.preprocessing import image
import time
import datetime
import os
import sys


def show_image(img):
    plt.imshow(img.reshape(28, 28), cmap='Greys')
    plt.show()


# def main():
csv_path = os.path.abspath('./train.csv')
csv_arr = np.genfromtxt(csv_path, delimiter=',', dtype='str', skip_header=1)

X = np.zeros((csv_arr.shape[0], 28*28))
y = np.zeros(csv_arr.shape[0])
for i in range(0, csv_arr.shape[0]):
    X[i][:] = csv_arr[i][1:]
    y[i] = csv_arr[i][0]

X = X.reshape((csv_arr.shape[0], 28, 28))

# Split Train and Validation #
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

# Parameters #
num_epo = 10000
num_batch = 32
patience = 50
verbose = 2
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M')
log_file_path = 'logs/'+timestamp
model_names = 'predict_model.hdf5'

# Image augmentation #
data_gen = image.ImageDataGenerator(rotation_range=39,
                                    shear_range=0.08,
                                    width_shift_range=0.08,
                                    height_shift_range=0.08)
data_gen.fit(X_train, augment=True, seed=1)
img_generator = data_gen.flow(X_train, y_train, batch_size=num_batch)


# Callbacks #
# early_stop = EarlyStopping(monitor='val_acc', patience=patience)
# tsb = TensorBoard(log_dir=log_file_path, histogram_freq=0, write_graph=True, write_images=True)
# hist = md.LossHistory()
# model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=int(patience/4), verbose=1)
# call_back = [hist, tsb, early_stop, model_checkpoint, reduce_lr]

# Train #
# classifier = md.build()
# classifier.fit_generator(img_generator,
#                          steps_per_epoch=cut/num_batch,
#                          epochs=num_epo,
#                          verbose=verbose,
#                          validation_data=(X_validation, Y_validation),
#                          callbacks=call_back)

# tsb.set_model(classifier)
# classifier.save(model_names)
# md.output_history(hist, timestamp)
# md.plot_acc(hist)

#
# if __name__ == "__main__":
#     main()