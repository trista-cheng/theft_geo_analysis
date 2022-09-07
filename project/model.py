import numpy as np
import pandas as pd
import seaborn as sns
# import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.utils import shuffle


def read_data(data_path, data_name):
    df = pd.read_csv(data_path + data_name + '.csv')
    y_df = df[['output']]
    # drop 'address', 'district', 'output'
    x_df = df.drop(columns=['output'])

    # 定義 X, y
    X = np.array(x_df)
    y = np.array(y_df).reshape(-1)
    # 洗牌
    X, y = shuffle(X, y, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

def cal_score(y_test, y_pred):

    # metric = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    return f1, accuracy, precision, recall, auc


def model_knn(data_name, X_train, X_test, y_train):
    # create new a knn model
    knn = KNeighborsClassifier()
    #create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(2, 10)}
    #use gridsearch to test all values for n_neighbors
    knn_cv = GridSearchCV(knn, param_grid, scoring=['accuracy', 'f1', 'recall', 'precision', 'roc_auc'],
                            refit='f1', cv=5, return_train_score=True)
    # fit model to data
    y_pred = knn_cv.fit(X_train, y_train).predict(X_test)
    best_k = knn_cv.best_params_['n_neighbors']

    return data_name, y_pred, best_k

def model_dt(data_name, X_train, X_test, y_train):
    depth_pair = {'bicycle': 13, 'home': 11}
    best_depth = depth_pair[data_name]
    clf_cv = DecisionTreeClassifier(criterion='gini', max_depth=best_depth)
    clf_cv.fit(X_train, y_train)
    # 預測結果
    y_pred = clf_cv.predict(X_test)

    return data_name, y_pred, best_depth


def randomforest(data_name, X_train, X_test, y_train):
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train, y_train)
    # predict
    y_pred = forest.predict(X_test)

    return data_name, y_pred


def svm(data_name, X_train, X_test, y_train):
    kernels = ['rbf', 'poly']
    c_range = 1 / np.geomspace(1e-4, 1, 15)
    parameters = {'kernel': kernels, 'C': c_range}
    gs = GridSearchCV(SVC(random_state=1), param_grid=parameters, scoring=['accuracy', 'f1', 'recall', 'precision', 'roc_auc'],
                    refit='f1', cv=5, return_train_score=True)
    gs.fit(X_train, y_train.to_numpy().ravel())
    cv_dict = {}
    for key in gs.cv_results_:
        if type(gs.cv_results_[key]) == list:
            cv_dict[key] = gs.cv_results_[key]
        else:
            cv_dict[key] = gs.cv_results_[key].tolist()

    y_pred = gs.predict(X_test)

    return data_name, y_pred


def naivebayes(data_name, X_train, X_test, y_train, case=1):
    if case == 1:
        clf = GaussianNB()
    else:
        clf = MultinomialNB()
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    return data_name, y_pred