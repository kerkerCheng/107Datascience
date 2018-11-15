import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import scale, OneHotEncoder

mask = [1, 2, 5, 13]


def read_file(train_path='train.csv', test_path='test.csv'):
    df_train = pd.read_csv(os.path.abspath(train_path), header=None)
    df_test = pd.read_csv(os.path.abspath(test_path), header=None)

    # Remove useless columns
    df_train.drop(df_train.columns[mask], axis=1, inplace=True)
    df_test.drop(df_test.columns[mask], axis=1, inplace=True)
    y = df_train.values[:, -1].astype(int)
    df_train.drop(df_train.columns[-1], axis=1, inplace=True)

    training_size, testing_size = df_train.shape[0], df_test.shape[0]
    print(df_train.shape)
    print(df_test.shape)

    # Convert to one-hot encoding
    frames = [df_train, df_test]
    df_all = pd.concat(frames)
    df_all = pd.get_dummies(df_all)

    train_X = (df_all.values[:training_size, :]).astype(float)
    test_X = (df_all.values[training_size:, :]).astype(float)

    # Normalize
    # X[:, 0:5] = scale(X[:, 0:5], axis=1, copy=True)

    return train_X, test_X, y


train_X, test_X, y = read_file()
print(train_X.shape)
print(test_X.shape)
print(y.shape)

# Training
classifier = xgb.XGBClassifier(max_depth=9,
                               learning_rate=0.1,
                               n_estimators=150,
                               silent=True,
                               n_jobs=4)

num_splits = 3
# Cross-Validation
cv_results = cross_validate(classifier, train_X, y, cv=num_splits, return_train_score=False)
print(cv_results['test_score'])


# Prediction
# classifier.fit(train_X, y)
# ans = classifier.predict(test_X)
#
# with open(os.path.abspath('sub.csv'), 'w+') as f:
#     f.write('ID,ans\n')
#     for index, element in enumerate(ans):
#         f.write(str(index) + ',' + str(element) + '\n')

