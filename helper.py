from itertools import chain, combinations
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import json
import scipy.stats as ss


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


def cramers_V(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    err_flag = True if min( (kcorr-1), (rcorr-1)) == 0 else False  # Debug: division of zero flag
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1))), err_flag