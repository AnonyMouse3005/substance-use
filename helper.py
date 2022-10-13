from itertools import chain, combinations
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import json


def powerset(iterable):
    # Usage: list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def loocv(X, y, model):
    loo = LeaveOneOut()
    scores = []
    for train_index, test_index in loo.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = model.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # print(score)
        scores.append(score)

    return np.mean(scores)

def standard_scale(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def json_dump(filepath, var):
    with open(filepath, 'w') as f:
        json.dump(var, f)