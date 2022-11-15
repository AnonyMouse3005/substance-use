from itertools import chain, combinations
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (SelectFromModel, SelectPercentile, chi2)
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings

from helper import *
from genetic_selection import GeneticSelectionCV
sys.path.insert(0, '../sklearn-genetic-mod')
from genetic_selection_mod import GeneticSelectionCV_mod


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

def norm_scale(X):
    scaler = Normalizer().fit(X)
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


'''
See imputing notes from data_nonet_analysis.ipynb
'''
def impute_MARs(vars, df):

    for v in vars:
        col = df[v]
        if v == 'TB2':
            for idx, i in enumerate(col):
                if i == 5:
                    df.at[idx, 'TB2_4_TEXT'] = -1
                    df.at[idx, 'TB3'] = 1
                    df.at[idx, 'TB4'] = 0
        if v == 'TB3':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'TB4'] = 0
        if v == 'TB5':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'TB6'] = -1
                    df.at[idx, 'TB7'] = 1
                    df.at[idx, 'TB8'] = 0
        if v == 'TB7':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'TB8'] = 0
        if v == 'TB9':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'TB10'] = -1
                    df.at[idx, 'TB11'] = 1
                    df.at[idx, 'TB12'] = 0
        if v == 'TB11':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'TB12'] = 0
        if v == 'AL1':
            for idx, i in enumerate(col):
                if i == 2:
                    df.at[idx, 'AL1_4_TEXT'] = -1
                    df.at[idx, 'AL2_1_TEXT'] = -1
                    df.at[idx, 'AL3_1_TEXT'] = -1
                    df.at[idx, 'AL4'] = 1
                    df.at[idx, 'AL5'] = 0
                    df.at[idx, 'AL6A'] = 0
                    df.at[idx, 'AL6B'] = 0
                elif i == 3:
                    df.at[idx, 'AL1_4_TEXT'] = -2
        if v == 'AL2':
            for idx, i in enumerate(col):
                if i == 2:
                    df.at[idx, 'AL2_1_TEXT'] = -2
        if v == 'AL3':
            for idx, i in enumerate(col):
                if i == 2:
                    df.at[idx, 'AL3_1_TEXT'] = -2
        if v == 'AL5':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'AL6A'] = 0
                    df.at[idx, 'AL6B'] = 0
        if v == 'AL6A':
            for idx, i in enumerate(col):
                if pd.isnull(df.loc[idx, v]) and not pd.isnull(df.loc[idx, 'AL6B']):
                    df.at[idx, v] = df.at[idx, 'AL6B']
        if v == 'ID1':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'ID2'] = -1
                    for j in range(3,13):
                        df.at[idx, f'ID{j}'] = 0
                    for j in range(15,21):
                        df.at[idx, f'ID{j}'] = -1
        if v == 'ID3':
            for idx, i in enumerate(col):
                if i == 1:
                    for j in range(4,13):
                        df.at[idx, f'ID{j}'] = 0
                    for j in range(15,21):
                        df.at[idx, f'ID{j}'] = -1
        if v == 'ID17':
            for idx, i in enumerate(col):
                if i == 1:
                    for j in range(18,21):
                        df.at[idx, f'ID{j}'] = -1
        if v == 'ND1':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'ND2'] = -1
        if v == 'OD1':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'OD2'] = 0
        if v == 'OD6':
            for idx, i in enumerate(col):
                if i == 2:
                    for j in range(7,12):
                        df.at[idx, f'OD{j}'] = 0
        if v == 'OD8':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'OD9'] = 0
        if v == 'OD10':
            for idx, i in enumerate(col):
                if i == 1:
                    df.at[idx, 'OD11'] = 0
        if v == 'CJ3':
            for idx, i in enumerate(col):
                if i == 1:
                    for j in range(4,8):
                        df.at[idx, f'CJ{j}'] = -1
        if v == 'DM12':
            for idx, i in enumerate(col):
                if i != 1:
                    df.at[idx, 'DM13'] = -1

    vars_mixed = ['DM1','TB2_4_TEXT','TB6','TB10','AL1_4_TEXT','AL2_1_TEXT','AL3_1_TEXT','ID2','ND2']
    for v in vars_mixed:
        col = df[v]
        if v[-4:] == 'TEXT':  # e.g., modify TB2 column instead of TB2_4_TEXT
            v = v.split('_')[0]
        for idx, i in enumerate(col):
            if 0 <= i <= 14:    df.at[idx, v] = 0  # children
            elif 15 <= i <= 24: df.at[idx, v] = 1  # youth
            elif 25 <= i <= 64: df.at[idx, v] = 2  # adult
            elif i >= 65:       df.at[idx, v] = 3  # senior
            elif i == -1:       df.at[idx, v] = -1  # never
            elif i == -2 or (np.isnan(i) and df.at[idx, v] == 4):       df.at[idx, v] = np.nan  # don't know

    v = 'SC1'  # numerical variable (# of years)
    col = df[v]
    for idx, i in enumerate(col):
        if i < 0.5:    df.at[idx, v] = 0  # less than 6 months
        elif 0.5 <= i < 1:  df.at[idx, v] = 1  # 
        elif 1 <= i < 2:    df.at[idx, v] = 2  #
        elif 2 <= i < 5:    df.at[idx, v] = 3  #
        elif 5 <= i < 10:    df.at[idx, v] = 4  #
        elif i >= 10:       df.at[idx, v] = 5  # more than 10 years

    return df


def plot_learning_curve(  # Originally from sklearn's doc
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
    score=None, baseline=1
):
    """
    Generate ONE plots: the test and training learning curves
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    if score is not None:
        axes.plot([], [], ' ', label=f"CV acc/baseline acc: {round(score, 3)}/{round(baseline, 3)}")

    axes.legend(loc="lower right")

    return plt


#---------------------------------------------------------------------------------------------------------------------
def plot_learning_curve_v2(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
    score=None, baseline=1
):
    """
    Generate ONE plot: the test and training learning curves
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))

    plt.rc('axes', titlesize=10) 
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    if score is not None:
        axes.plot([], [], ' ', label=f"CV acc/baseline acc: {round(score, 3)}/{round(baseline, 3)}")

    axes.legend(loc="lower right")

    fig.tight_layout()

    return plt


@ignore_warnings(category=ConvergenceWarning)
def genetic_alg_mod(clf, clf_name, hparams, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    print(f'classifier: {clf_name}')
    X = standard_scale(X_raw) if clf_name != 'DT' else X_raw
    i = 0
    for train_idx, test_idx in cv_outer.split(X):
        i += 1
        n_splits = cv_outer.n_splits if hasattr(cv_outer, 'n_splits') else X.shape[0]
        print(f'At fold {i}/{n_splits} (outer CV)')
        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        model = GeneticSelectionCV_mod(
            clf, cv=cv, verbose=0,
            scoring="accuracy", max_features=None,
            n_population=300, crossover_proba=0.5,
            mutation_proba=0.2, n_generations=40,
            crossover_independent_proba=0.1,
            mutation_independent_proba=0.05,
            tournament_size=3, n_gen_no_change=10,
            hparams=hparams,
            caching=False, n_jobs=10)
        model = model.fit(X_train, y_train)

        X_train_new, X_test_new = X_train[:, model.support_], X_test[:, model.support_]
        best_params = model.best_params_
        if hparams:
            for k, v in best_params.items():
                if k in ['max_depth', 'min_samples_split']:      v = int(round(v, 0))
                setattr(clf, k, v)

        results.append(clf.fit(X_train_new, y_train).score(X_test_new, y_test))
    
    return np.mean(results)


@ignore_warnings(category=ConvergenceWarning)
def genetic_alg(clf, clf_name, hparams_grid, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    print(f'classifier: {clf_name}')
    X = standard_scale(X_raw) if clf_name != 'DT' else X_raw
    i = 0
    for train_idx, test_idx in cv_outer.split(X):
        i += 1
        n_splits = cv_outer.n_splits if hasattr(cv_outer, 'n_splits') else X.shape[0]
        print(f'At fold {i}/{n_splits} (outer CV)')
        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        model = GeneticSelectionCV(
            clf, cv=cv, verbose=0,
            scoring="accuracy", max_features=None,
            n_population=300, crossover_proba=0.5,
            mutation_proba=0.2, n_generations=40,
            crossover_independent_proba=0.1,
            mutation_independent_proba=0.05,
            tournament_size=3, n_gen_no_change=10,
            caching=False, n_jobs=10)
        model = model.fit(X_train, y_train)

        X_train_new, X_test_new = X_train[:, model.support_], X_test[:, model.support_]
        search = GridSearchCV(clf, param_grid=hparams_grid, cv=cv, refit=True)
        search.fit(X_train_new, y_train)

        results.append(search.score(X_test_new, y_test))
    
    return np.mean(results)


@ignore_warnings(category=ConvergenceWarning)
def pca(clf, clf_name, hparams_grid, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    n_components = np.linspace(2, 20, 10, dtype=np.int32)
    X = standard_scale(X_raw)
    for train_idx, test_idx in cv_outer.split(X):

        X_train, y_train, X_test, y_test = X_raw[train_idx], y[train_idx], X_raw[test_idx], y[test_idx]
        cv_scores = {}
        for j, n_pc in enumerate(n_components):
            model = PCA(n_components=n_pc)
            X_train_new = model.fit_transform(X_train)
            X_test_new = model.transform(X_test)
            cv_scores[j] = {'score': np.mean(cross_val_score(clf, X_train_new, y_train, scoring='accuracy', cv=cv)),
                            'X_train_new': X_train_new, 'X_test_new': X_test_new}

        best_pc_idx = np.argmax([i['score'] for i in cv_scores.values()])
        X_train_best, X_test_best = cv_scores[best_pc_idx]['X_train_new'], cv_scores[best_pc_idx]['X_test_new']
        search = GridSearchCV(clf, param_grid=hparams_grid, cv=cv, refit=True)
        search.fit(X_train_best, y_train)

        results.append(search.score(X_test_best, y_test))

    return np.mean(results)


@ignore_warnings(category=ConvergenceWarning)
def chi2_filter(clf, clf_name, hparams_grid, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    percentiles = [3, 6, 10, 15, 20, 30, 40]
    X = standard_scale(X_raw) if clf_name != 'DT' else X_raw
    for train_idx, test_idx in cv_outer.split(X):

        X_train, y_train, X_test, y_test = X_raw[train_idx], y[train_idx], X_raw[test_idx], y[test_idx]
        cv_scores = {}
        for j, pc in enumerate(percentiles):
            model = SelectPercentile(chi2, percentile=pc)
            X_train_new = model.fit_transform(X_train, y_train)
            X_test_new = model.transform(X_test)
            cv_scores[j] = {'score': np.mean(cross_val_score(clf, X_train_new, y_train, scoring='accuracy', cv=cv)),
                            'X_train_new': X_train_new, 'X_test_new': X_test_new}

        best_pc_idx = np.argmax([i['score'] for i in cv_scores.values()])
        X_train_best, X_test_best = cv_scores[best_pc_idx]['X_train_new'], cv_scores[best_pc_idx]['X_test_new']
        search = GridSearchCV(clf, param_grid=hparams_grid, cv=cv, refit=True)
        search.fit(X_train_best, y_train)

        results.append(search.score(X_test_best, y_test))

    return np.mean(results)


@ignore_warnings(category=ConvergenceWarning)
def thresholding(clf, clf_name, hparams_grid, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    thresholds = [f"{scale}*mean" for scale in [0.1, 0.5, 0.75, 1, 1.25, 1.5, 2]]
    X = standard_scale(X_raw) if clf_name != 'DT' else X_raw
    for train_idx, test_idx in cv_outer.split(X):
        
        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        if clf_name == 'SVM':       clf.kernel = 'linear'  # only linear kernel allows SVM to have coef_
        importances = 'auto'
        cv_scores = {}
        for j, th in enumerate(thresholds):
            model = SelectFromModel(clf.fit(X_train, y_train), prefit=True, importance_getter=importances, threshold=th)
            X_train_new, X_test_new = model.transform(X_train), model.transform(X_test)
            cv_scores[j] = {'score': np.mean(cross_val_score(clf, X_train_new, y_train, scoring='accuracy', cv=cv)),
                            'X_train_new': X_train_new, 'X_test_new': X_test_new}

        best_pc_idx = np.argmax([i['score'] for i in cv_scores.values()])
        X_train_best, X_test_best = cv_scores[best_pc_idx]['X_train_new'], cv_scores[best_pc_idx]['X_test_new']
        search = GridSearchCV(clf, param_grid=hparams_grid, cv=cv, refit=True)
        search.fit(X_train_best, y_train)

        results.append(search.score(X_test_best, y_test))

    return np.mean(results)


def handpick_features(drug, X_df):
    features_idx = [X_df.columns.get_loc(c) for c in X_df if c.startswith('AC') or c in ['DM1','DM8','DM9','DM23','SC5','SC6','SC9']]
    if drug == 'marijuana':
        return sorted(features_idx + [X_df.columns.get_loc('ND1')])
    elif drug == 'meth':
        return sorted(features_idx + [X_df.columns.get_loc('ND7')])