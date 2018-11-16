import xgboost as xgb
import numpy as np
import datetime
import time
import pandas as pd
import os
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

train_path = 'train.csv'
test_path = 'test.csv'
mask = [2, 5]
le = preprocessing.LabelEncoder()

df_train = pd.read_csv(os.path.abspath(train_path), header=None)
df_test = pd.read_csv(os.path.abspath(test_path), header=None)

# Remove useless columns
df_train.drop(df_train.columns[mask], axis=1, inplace=True)
df_test.drop(df_test.columns[mask], axis=1, inplace=True)
y = df_train.values[:, -1].astype(int)
df_train.drop(df_train.columns[-1], axis=1, inplace=True)

training_size, testing_size = df_train.shape[0], df_test.shape[0]
frames = [df_train, df_test]
df_all = pd.concat(frames)
X_all = df_all.values

# Label Encoding
for i in range(X_all.shape[1]):
    if type(X_all[0][i]) is str:
        X_all[:, i] = le.fit_transform(X_all[:, i])

train_X = (X_all[:training_size, :])
test_X = (X_all[training_size:, :])



# Convert to one-hot encoding
# df_all = pd.get_dummies(df_all)



# Normalize
# X[:, 0:5] = scale(X[:, 0:5], axis=1, copy=True)