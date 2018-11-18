import xgboost as xgb
import numpy as np
import datetime
import time
import pandas as pd
import os
import sys
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.feature_selection import RFE


timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m.%d_%H.%M')
mask = [13]


def read_file(train_path='train.csv', test_path='test.csv'):
    le = preprocessing.LabelEncoder()

    df_train = pd.read_csv(os.path.abspath(train_path), header=None, engine='python')
    df_test = pd.read_csv(os.path.abspath(test_path), header=None, engine='python')

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

    train_X = (X_all[:training_size, :]).astype(int)
    test_X = (X_all[training_size:, :]).astype(int)

    # Convert to one-hot encoding
    # frames = [df_train, df_test]
    # df_all = pd.concat(frames)
    # df_all = pd.get_dummies(df_all)
    #
    # train_X = (df_all.values[:training_size, :]).astype(float)
    # test_X = (df_all.values[training_size:, :]).astype(float)

    # Normalize
    # X[:, 0:5] = scale(X[:, 0:5], axis=1, copy=True)

    return train_X, test_X, y


def grid_search(train_X, y):

    # Cross-Validation

    classifier_xgb = xgb.XGBClassifier()
    classifier_svm = svm.SVC()
    classifier_ada = AdaBoostClassifier()
    classifier_usvm = svm.NuSVC()

    parameters = {'C': [1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0], 'kernel': ['rbf']}
    # parameters = {'n_estimators': [160, 180, 200, 220, 240, 260, 280], 'learning_rate': [1.5, 1.7, 1.9, 2.1, 2.3]}
    # parameters = {'max_depth': [6, 8, 10, 12],
    #               'learning_rate': [0.08, 0.1, 0.12],
    #               'n_estimators': [180, 210, 240],
    #               'n_jobs': [4],
    #               'objective': ['reg:logistic', 'reg:linear', 'binary:hinge'],
    #               'max_delta_step': [0, 2, 4, 6, 8],
    #               'min_child_weight': [1, 3, 5, 7, 9],
    #               'gamma': [0, 2, 4, 6]}
    # parameters = {'nu': [0.3, 0.5, 0.7],
    #               'degree': [3, 5, 7]}

    num_splits = 3
    clf_grid = GridSearchCV(classifier_svm, parameters, cv=num_splits, n_jobs=4, verbose=2)
    clf_grid.fit(train_X, y)
    grid_result = pd.DataFrame.from_dict(clf_grid.cv_results_)
    grid_result.to_csv(timestamp + '_svm_grid.csv', sep=',', encoding='utf-8')

    # cv_results = cross_validate(classifier_xgb, train_X, y, cv=num_splits, return_train_score=False)
    # cv_results = cross_validate(classifier_ada, train_X, y, cv=num_splits, return_train_score=False)
    # print(cv_results['test_score'])


def rfecv(train_X, y):
    classifier_xgb = xgb.XGBClassifier(silent=True,
                                       max_depth=4,
                                       learning_rate=0.1,
                                       n_estimators=210,
                                       n_jobs=4,
                                       objective='reg:logistic')
    selector = RFE(classifier_xgb, step=1, verbose=2, n_features_to_select=9)
    selector = selector.fit(train_X, y)
    np.save('feature_ranking.npy', selector.grid_scores_)


def main():
    train = sys.argv[1]
    test = sys.argv[2]
    train_X, test_X, y = read_file(train, test)

    # Training
    classifier_xgb = xgb.XGBClassifier(silent=True,
                                       max_depth=6,
                                       learning_rate=0.1,
                                       n_estimators=210,
                                       objective='reg:logistic')
    classifier_svm = svm.SVC(C=1.95)
    classifier_ada = AdaBoostClassifier(n_estimators=260, learning_rate=1.5)

    # grid_search(train_X, y)
    # rfecv(train_X, y)

    # Prediction
    voting_models = [('xgb', classifier_xgb), ('svm', classifier_svm), ('ada', classifier_ada)]
    clf = VotingClassifier(estimators=voting_models, n_jobs=None)
    clf.fit(train_X, y)
    ans = clf.predict(test_X)

    with open(os.path.abspath('answer.csv'), 'w+') as f:
        f.write('ID,ans\n')
        for index, element in enumerate(ans):
            f.write(str(index) + ',' + str(element) + '\n')


if __name__ == '__main__':
    main()
