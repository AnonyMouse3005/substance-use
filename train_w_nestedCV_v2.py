# %% This version test different imputation methods
import ast
import json
import multiprocessing
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import (SelectFromModel, SelectPercentile, chi2,
                                       mutual_info_classif)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import *
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from helper import *


# %% load data and impute
def load_data(cohort, drug):

    if cohort == 1:
        nonet_vars, nonet_df = C1W1nonet_vars, C1W1nonet_df
        pred_df = C1pred_df
        pred_var = zip(pred_df['ND1'], pred_df['Q68']) if drug == 'marijuana' else zip(pred_df['ND7'], pred_df['Q75'])
    elif cohort == 2:
        nonet_vars, nonet_df = C2W1nonet_vars, C2W1nonet_df
        pred_df = C2pred_df
        pred_var = zip(pred_df['ND1'], pred_df['W2_ND1']) if drug == 'marijuana' else zip(pred_df['ND7'], pred_df['W2_ND7'])
    elif cohort == '1+2':
        nonet_df = pd.concat([C1W1nonet_df, C2W1nonet_df], ignore_index=True)
        nonet_vars = C1W1nonet_vars  # same set of columns for both cohorts
        colname_map = {}
        C2pred_keys = list(C2pred_df.columns)
        for i, c in enumerate(list(C1pred_df.columns)):  # map column names of C2pred_df to C1pred_df (since C1W2 has different varnames)
            colname_map[C2pred_keys[i]] = c
        pred_df = pd.concat([C1pred_df, C2pred_df.rename(columns=colname_map)], ignore_index=True)
        pred_var = zip(pred_df['ND1'], pred_df['Q68']) if drug == 'marijuana' else zip(pred_df['ND7'], pred_df['Q75'])
        
    df = impute_MARs(nonet_vars, nonet_df)
    discarded_vars = ['PID','PID2','AL6B','ID13','ID14_4','ID14_5','ID14_6','ID14_7','ND13','ND15_4','ND15_5','ND15_6','ND15_7',
                'DA5','DA6','DA7','DA7a','DA7b','DA7c','DA7d','DA8','DA8a','DA8b','DA8c','DA8d'] + [v for v in list(df.columns) if 'TEXT' in v]

    dep_var_full = []
    for a, b in pred_var:
        if not np.isnan(a) and not np.isnan(b):
            y = 1 if b - a > 0 else 0
            # y = 1 if b - a > 1 else 0
            # y = b - a
            dep_var_full.append(y)
        else:   dep_var_full.append(np.nan)

    df = pd.concat([df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
    y = np.array(dep_var_full)
    y = y[~np.isnan(y)]

    X_df = df[df['pred'].notna()].drop(discarded_vars+['pred'], axis=1)
    X_df.reset_index(drop=True, inplace=True)

    return X_df, y, dep_var_full


def case_var_delete(cohort, drug, thresh=0.9):  # delete case and/or variable if too many missing values

    X_df, y, dep_var_full = load_data(cohort, drug)
    X_dropcol_df = X_df.dropna(axis=1, thresh=len(X_df)*thresh)  # drop column (variable) if missing 10% out of all participants
    X_drop_df = X_df.dropna(axis=0, thresh=len(X_dropcol_df.columns)*thresh)  # drop row (partcipant) if missing 10% out of all features
    if len(X_drop_df) < len(X_df):  y = y[X_drop_df.index]

    nominal_vars = [v for v in X_drop_df.columns if v in ['DM8','DM10','DM12','DM13']]
    X_ordinal_df = X_drop_df.drop(nominal_vars, axis=1)
    X_nominal_df = X_drop_df[nominal_vars]
    # Encode
    Xenc_ordinal_df = X_ordinal_df.astype('str').apply(LabelEncoder().fit_transform)
    Xenc_ordinal_df = Xenc_ordinal_df.where(~X_ordinal_df.isna(), X_ordinal_df)  # Do not encode the NaNs

    nominal_cols = []
    for v in nominal_vars:
        nominal_cols.append(pd.get_dummies(X_nominal_df[v], prefix=v))
    Xenc_nominal_df = pd.concat(nominal_cols, axis=1)
    Xenc_df = pd.concat([Xenc_ordinal_df, Xenc_nominal_df], axis=1)
    # Impute
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_imp = imp.fit_transform(Xenc_df)

    return X_imp, y, Xenc_df, list(X_drop_df.index), dep_var_full


def missforest(cohort, drug):
    X_df, y, dep_var_full = load_data(cohort, drug)
    nominal_vars = ['DM8','DM10','DM12','DM13']
    X_ordinal_df = X_df.drop(nominal_vars, axis=1)
    X_nominal_df = X_df[nominal_vars]
    Xenc_ordinal_df = X_ordinal_df.astype('str').apply(LabelEncoder().fit_transform)
    Xenc_ordinal_df = Xenc_ordinal_df.where(~X_ordinal_df.isna(), X_ordinal_df)  # Do not encode the NaNs
    Xenc_df = pd.concat([Xenc_ordinal_df, X_nominal_df], axis=1)
    imp = IterativeImputer(estimator=RandomForestClassifier(), 
                    initial_strategy='most_frequent',
                    max_iter=10, random_state=0)
    X = imp.fit_transform(Xenc_df)
    X_nominal_df = pd.DataFrame(X[:, -len(nominal_vars):], columns=nominal_vars)
    nominal_cols = []
    for v in nominal_vars:
        nominal_cols.append(pd.get_dummies(X_nominal_df[v], prefix=v))
    Xenc_nominal_df = pd.concat(nominal_cols, axis=1)
    Xenc_df = pd.concat([Xenc_ordinal_df, Xenc_nominal_df], axis=1)
    X = np.concatenate((X[:, :-len(nominal_vars)], Xenc_nominal_df.to_numpy()), axis=1)
    return X, y, Xenc_df, dep_var_full


# %% which feature type to use for training?
def run_nonnetwork():
    scores_dict = {}
    for cohort in cohorts:
        for drug in drugs:

            print(f'At Cohort {cohort}, {drug} using non-network features')

            if impute_method == 'case-var-del':
                X_imp, y, Xenc_df, _, _ = case_var_delete(cohort, drug)  # df with dropped rows and columns if too much missingness
            elif impute_method == 'missforest':
                X_imp, y, Xenc_df, _ = missforest(cohort, drug)

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            for clf_name in clf_dict.keys():
                scores_dict[f'{cohort}-{drug}-{clf_name}'] = []

            # Group features
            f_dict = {}
            fgroups = ['SC', 'DM', 'TB', 'AL', 'ID', 'ND', 'DA', 'OD', 'TX', 'AC', 'CJ']  # feature groups
            for g in fgroups:
                f_indices = [Xenc_df.columns.get_loc(c) for c in Xenc_df if c.startswith(g)]  # column indices of the group's features
                f_dict[g] = f_indices

            baseline = stats.mode(y)[1][0]/len(y)

            # Domain-specific feature grouping
            if 'manual' in methods:
                for gname, fsubs in fsubs_nonnet.items():
                    scores_dict[f'{cohort}-{drug}-fgroup'].append(gname)
                    X = X_imp[:, [f for fgroup in [f_dict[s] for s in fsubs] for f in fgroup]]
                    for clf_name, clf in clf_dict.items():
                        print(f'Cohort {cohort}, {drug}: start running manual grouping {gname} for {clf_name}')
                        scores = []
                        for train_idx, test_idx in cv_outer.split(X):
                            X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                            if clf_name != 'DT':
                                X_train = standard_scale(X_train)
                                X_test = standard_scale(X_test)
                            search = GridSearchCV(clf, param_grid=clf_param_grid[clf_name], cv=cv_inner, refit=True, n_jobs=5)
                            search.fit(X_train, y_train)
                            scores.append(search.score(X_test, y_test))
                        scores_dict[f'{cohort}-{drug}-{clf_name}'].append(np.mean(scores))

            # various feature selection methods
            for m_name in methods:
                if m_name == 'manual':  continue
                scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                for clf_name, clf in clf_dict.items():
                    print(f'Cohort {cohort}, {drug}: start running {m_name} for {clf_name}')
                    if m_name == 'thresholding':    final_score = thresholding(clf, clf_name, clf_param_grid[clf_name], X_imp, y, cv_inner, cv_outer)
                    if m_name == 'chi2':    final_score = chi2_filter(clf, clf_name, clf_param_grid[clf_name], X_imp, y, cv_inner, cv_outer)
                    if m_name == 'pca':    final_score = pca(clf, clf_name, clf_param_grid[clf_name], X_imp, y, cv_inner, cv_outer)
                    if m_name == "GA":     final_score = genetic_alg(clf, clf_name, clf_param_grid[clf_name], X_imp, y, cv_inner, cv_outer)
                    if m_name == "GAmod":   final_score = genetic_alg_mod(clf, clf_name, clf_params[clf_name], X_imp, y, cv_inner, cv_outer)
                    scores_dict[f'{cohort}-{drug}-{clf_name}'].append(final_score)

            scores_dict[f'{cohort}-{drug}-baseline'] = [baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

    pd.DataFrame.from_dict(scores_dict).to_csv(f'results/nestedCV_scores_nonnetwork_{goals_code}.csv', index=False)


def run_network():
    scores_dict = {}
    for cohort in cohorts:
        for drug in drugs:

            print(f'At Cohort {cohort}, {drug} using network features')

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            for clf_name in clf_dict.keys():
                scores_dict[f'{cohort}-{drug}-{clf_name}'] = []

            # csv generated from data_net_analysis.ipynb (Cramer's V section)
            df = pd.read_csv(f"saved-vars/C{''.join(str(cohort).split('+'))}_network_221114-processed.csv")
            if cohort == 1:
                pred_df = C1pred_df
                pred_var = zip(pred_df['ND1'], pred_df['Q68']) if drug == 'marijuana' else zip(pred_df['ND7'], pred_df['Q75'])
            elif cohort == 2:
                pred_df = C2pred_df
                pred_var = zip(pred_df['ND1'], pred_df['W2_ND1']) if drug == 'marijuana' else zip(pred_df['ND7'], pred_df['W2_ND7'])
            elif cohort == '1+2':
                colname_map = {}
                C2pred_keys = list(C2pred_df.columns)
                for i, c in enumerate(list(C1pred_df.columns)):  # map column names of C2pred_df to C1pred_df (since C1W2 has different varnames)
                    colname_map[C2pred_keys[i]] = c

                pred_df = pd.concat([C1pred_df, C2pred_df.rename(columns=colname_map)], ignore_index=True)
                pred_var = zip(pred_df['ND1'], pred_df['Q68']) if drug == 'marijuana' else zip(pred_df['ND7'], pred_df['Q75'])
            
            dep_var_full = []
            for a, b in pred_var:
                if not np.isnan(a) and not np.isnan(b):
                    y = 0 if a <= b else 1
                    # y = a - b
                    dep_var_full.append(y)
                else:   dep_var_full.append(np.nan)

            df = pd.concat([df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
            X_df = df[df['pred'].notna()].drop('pred', axis=1)

            # Group features
            f_dict = {}
            fgroups = ['NSX','NDX']  # feature groups
            for g in fgroups:
                f_indices = [X_df.columns.get_loc(c) for c in X_df if c.startswith(g)]  # column indices of the group's features
                f_dict[g] = f_indices

            X_np = X_df.to_numpy()

            # Learning curve
            y = np.array(dep_var_full)
            y = y[~np.isnan(y)]
            baseline = stats.mode(y)[1][0]/len(y)

            # Domain-specific feature grouping
            if 'manual' in methods:
                for gname, fsubs in fsubs_net.items():
                    scores_dict[f'{cohort}-{drug}-fgroup'].append(gname)
                    X = X_np[:, [f for fgroup in [f_dict[s] for s in fsubs] for f in fgroup]]
                    for clf_name, clf in clf_dict.items():
                        print(f'Cohort {cohort}, {drug}: start running manual grouping {gname} for {clf_name}')
                        scores = []
                        for train_idx, test_idx in cv_outer.split(X):
                            X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                            if clf_name != 'DT':
                                X_train = standard_scale(X_train)
                                X_test = standard_scale(X_test)
                            search = GridSearchCV(clf, param_grid=clf_param_grid[clf_name], cv=cv_inner, refit=True, n_jobs=5)
                            search.fit(X_train, y_train)
                            scores.append(search.score(X_test, y_test))
                        scores_dict[f'{cohort}-{drug}-{clf_name}'].append(np.mean(scores))

            # various feature selection methods
            for m_name in methods:
                if m_name == 'manual':  continue
                scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                for clf_name, clf in clf_dict.items():
                    print(f'Cohort {cohort}, {drug}: start running {m_name} for {clf_name}')
                    if m_name == 'thresholding':    final_score = thresholding(clf, clf_name, clf_param_grid[clf_name], X_np, y, cv_inner, cv_outer)
                    if m_name == 'chi2':    final_score = chi2_filter(clf, clf_name, clf_param_grid[clf_name], X_np, y, cv_inner, cv_outer)
                    if m_name == 'pca':    final_score = pca(clf, clf_name, clf_param_grid[clf_name], X_np, y, cv_inner, cv_outer)
                    if m_name == "GA":     final_score = genetic_alg(clf, clf_name, clf_param_grid[clf_name], X_np, y, cv_inner, cv_outer)
                    if m_name == "GAmod":   final_score = genetic_alg_mod(clf, clf_name, clf_params[clf_name], X_np, y, cv_inner, cv_outer)
                    scores_dict[f'{cohort}-{drug}-{clf_name}'].append(final_score)

            scores_dict[f'{cohort}-{drug}-baseline'] = [baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

    pd.DataFrame.from_dict(scores_dict).to_csv(f'results/nestedCV_scores_network_{goals_code}.csv', index=False)


def run_netnonnetwork():
    scores_dict = {}
    for cohort in cohorts:
        for drug in drugs:

            print(f'At Cohort {cohort}, {drug} using network + non-network features')

            #---------------------------------- Non-network ------------------------------------------------------
            if impute_method == 'case-var-del':
                X_imp, y, Xenc_df, drop_idx, dep_var_full = case_var_delete(cohort, drug)  # df with dropped rows and columns if too much missingness
            elif impute_method == 'missforest':
                X_imp, y, Xenc_df, dep_var_full = missforest(cohort, drug)

            #---------------------------------- Network ------------------------------------------------------
            net_df = pd.read_csv(f"saved-vars/C{''.join(str(cohort).split('+'))}_network_221114-processed.csv")
            net_df = pd.concat([net_df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
            Xnet_df = net_df[net_df['pred'].notna()].drop('pred', axis=1)
            if impute_method == 'case-var-del':     Xnet_df = Xnet_df.iloc[drop_idx]  # drop rows that were dropped during case-var-del of nonnetwork data

            X_np = Xnet_df.to_numpy()

            #-------------------------------------------------------------------------------------------
            X_all = np.concatenate((X_imp, X_np), axis=1)
            X_df = pd.concat([Xenc_df, Xnet_df], axis=1)
            

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            for clf_name in clf_dict.keys():
                scores_dict[f'{cohort}-{drug}-{clf_name}'] = []

            # Group non-network features
            f_nonet_dict = {}
            fgroups_nonet = ['SC', 'DM', 'TB', 'AL', 'ID', 'ND', 'DA', 'OD', 'TX', 'AC', 'CJ']
            for g in fgroups_nonet:
                f_indices = [Xenc_df.columns.get_loc(c) for c in Xenc_df if c.startswith(g)]  # column indices of the group's features
                f_nonet_dict[g] = f_indices
            # Group network features
            f_net_dict = {}
            fgroups_net = ['NSX','NDX']
            for g in fgroups_net:
                f_indices = [Xnet_df.columns.get_loc(c) for c in Xnet_df if c.startswith(g)]  # column indices of the group's features
                f_net_dict[g] = f_indices

            baseline = stats.mode(y)[1][0]/len(y)

            # Domain-specific feature grouping
            if 'manual' in methods:
                for gname, fsubs in fsubs_netnonnet.items():
                    scores_dict[f'{cohort}-{drug}-fgroup'].append(gname)
                    X_nonet = X_imp[:, [f for fgroup in [f_nonet_dict[s] for s in fsubs if s in fgroups_nonet] for f in fgroup]]
                    X_net = X_np[:, [f for fgroup in [f_net_dict[s] for s in fsubs if s in fgroups_net] for f in fgroup]]
                    X = np.concatenate((X_nonet, X_net), axis=1)
                    for clf_name, clf in clf_dict.items():
                        print(f'Cohort {cohort}, {drug}: start running manual grouping {gname} for {clf_name}')
                        scores = []
                        for train_idx, test_idx in cv_outer.split(X):
                            X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                            if clf_name != 'DT':
                                X_train = standard_scale(X_train)
                                X_test = standard_scale(X_test)
                            search = GridSearchCV(clf, param_grid=clf_param_grid[clf_name], cv=cv_inner, refit=True, n_jobs=5)
                            search.fit(X_train, y_train)
                            scores.append(search.score(X_test, y_test))
                        scores_dict[f'{cohort}-{drug}-{clf_name}'].append(np.mean(scores))

            # various feature selection methods
            for m_name in methods:
                if m_name == 'manual':  continue
                scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                for clf_name, clf in clf_dict.items():
                    print(f'Cohort {cohort}, {drug}: start running {m_name} for {clf_name}')
                    if m_name == 'thresholding':    final_score = thresholding(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer)
                    if m_name == 'chi2':    final_score = chi2_filter(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer)
                    if m_name == 'pca':    final_score = pca(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer)
                    if m_name == "GA":     final_score = genetic_alg(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer)
                    if m_name == "GAmod":   final_score = genetic_alg_mod(clf, clf_name, clf_params[clf_name], X_all, y, cv_inner, cv_outer)
                    scores_dict[f'{cohort}-{drug}-{clf_name}'].append(final_score)

            scores_dict[f'{cohort}-{drug}-baseline'] = [baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

    pd.DataFrame.from_dict(scores_dict).to_csv(f'results/nestedCV_scores_nonnetwork+network_{goals_code}.csv', index=False)


if __name__ == '__main__':

    # %% boilerplate
    if len(sys.argv) != 5:
        print('Usage: python3 train_w_nestedCV.py [non-network?] [network?] [non-network+network?] [goal_dict.json]')
        sys.exit(1)


    print(f'Number of available CPUs: {multiprocessing.cpu_count()}')

    # %% load dataset
    datapath = 'data/original/pre-imputed/'

    C1W1nonet_df = pd.read_csv(datapath + 'C1W1_nonnetwork_preimputed.csv')
    C1pred_df = pd.read_csv(datapath + 'C1_nonnetwork_pred.csv')
    C1W1nonet_vars = list(C1W1nonet_df.columns)

    C2W1nonet_df = pd.read_csv(datapath + '221114/C2W1_nonnetwork_preimputed.csv')
    C2pred_df = pd.read_csv(datapath + '221114/C2_nonnetwork_pred.csv')
    C2W1nonet_vars = list(C2W1nonet_df.columns)

    with open(sys.argv[4], 'r') as f:  # load dict containing lists of cohorts, drugs, and methods to be investigated
        goals = json.load(f)
    cohorts, drugs, methods, clf_list, impute_method = goals["cohorts"], goals["drugs"], goals["methods"], goals["clf_list"], goals["imputation"]


    # %% domain-specific feature groupings
    fsubs_1, fsubs_2, fsubs_3, fsubs_4 = ['SC'], ['DM'], ['TB','AL','ID','ND','DA','OD'], ['TX','AC','CJ']
    fsubs_5 = fsubs_1 + fsubs_2
    fsubs_6 = fsubs_1 + fsubs_2 + fsubs_3
    fsubs_7 = fsubs_1 + fsubs_2 + fsubs_4
    fsubs_8 = fsubs_1 + fsubs_2 + fsubs_3 + fsubs_4
    fsubs_nonnet = {'g1': fsubs_1, 'g2': fsubs_2, 'g3': fsubs_3, 'g4': fsubs_4, 'g5': fsubs_5, 'g6': fsubs_6, 'g7': fsubs_7, 'g8': fsubs_8}

    fsubs_net = {'g9': ['NSX'], 'g10': ['NDX'], 'g11': ['NSX','NDX']}

    fsubs_netnonnet = { 'g12': fsubs_nonnet['g3']+fsubs_net['g10'],
                        'g13': fsubs_nonnet['g6']+fsubs_net['g10'],
                        'g14': fsubs_nonnet['g8']+fsubs_net['g11']}


    # %% classifiers considered
    clf_choices = {
        'LG_L1': LogisticRegression(solver='saga', penalty='l1'),
        'LG_L2': LogisticRegression(solver='saga', penalty='l2'),
        'LG_EN': LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5),
        'DT': DecisionTreeClassifier(),
        'SVM': SVC(cache_size=1000)
    }
    clf_params = {
        'LG_L1': {'names': ['C'], 'range': [(0.001, 1000)], 'bitwidth': 8},
        'LG_L2': {'names': ['C'], 'range': [(0.001, 1000)], 'bitwidth': 8},
        'LG_EN': {'names': ['C'], 'range': [(0.001, 1000)], 'bitwidth': 8},
        'DT': {'names': ['max_depth','min_samples_split'], 'range': [(3, 10), (5,15)], 'bitwidth': 4},
        'SVM': {'names': ['gamma','C'], 'range': [(0.0001, 100), (0.001, 1000)], 'bitwidth': 8}
    }
    clf_param_grid = {
        'LG_L1': dict(C=np.logspace(-3,3,num=7)),
        'LG_L2': dict(C=np.logspace(-3,3,num=7)),
        'LG_EN': dict(C=np.logspace(-3,3,num=7)),
        'DT': dict(max_depth=range(3,11), min_samples_split=[5, 10, 15]),
        'SVM': dict(gamma=np.logspace(-4,2,num=7), C=np.logspace(-3,3,num=7))
    }
    clf_dict = {clf_name: clf for clf_name, clf in clf_choices.items() if clf_name in clf_list}
    cv_inner = int(goals["cv_inner"])
    cv_outer = LeaveOneOut() if goals["cv_outer"] == "LOO" else KFold(int(goals["cv_outer"]))
    goals_code = f"{'-'.join([str(c) for c in cohorts])}_{'-'.join([d[:4] for d in drugs])}_{'-'.join([m[:3] for m in methods])}_{'-'.join([clf_name for clf_name in clf_dict.keys()])}_{impute_method}"

    print(f'Process missing non-network data with {impute_method} imputation')

    # %% non-network features
    if int(sys.argv[1]):
        run_nonnetwork()

    # %% network features
    if int(sys.argv[2]):
        run_network()

    # %% network+nonnetwork features
    if int(sys.argv[3]):
        run_netnonnetwork()
        