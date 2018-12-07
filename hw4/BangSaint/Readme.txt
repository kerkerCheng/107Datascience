引入函式如下:

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