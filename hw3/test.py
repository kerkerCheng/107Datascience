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
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

train_path = 'train.csv'
test_path = 'test.csv'
mask = []
le = preprocessing.LabelEncoder()

df_train = pd.read_csv(os.path.abspath(train_path), header=None, engine='python')
df_test = pd.read_csv(os.path.abspath(test_path), header=None, engine='python')

# Remove useless columns
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

classifier_xgb = xgb.XGBClassifier(silent=True,
                                   max_depth=4,
                                   learning_rate=0.1,
                                   n_estimators=210,
                                   n_jobs=4,
                                   objective='reg:logistic')
kf = KFold(n_splits=3)
result = []

for i in range(train_X.shape[1]):
    print(i)
    x = np.delete(train_X, i, 1)
    acc = []
    for train_ind, test_ind in kf.split(x):
        x_tr, x_ts = x[train_ind], x[test_ind]
        y_tr, y_ts = y[train_ind], y[test_ind]
        classifier_xgb.fit(x_tr, y_tr)
        acc.append(accuracy_score(y_ts, classifier_xgb.predict(x_ts)))
    result.append(acc)
