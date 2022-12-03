# %% This version report AUROC (also objective AUROC while cv-ing), also fixing data
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import (SelectFromModel, SelectPercentile, chi2,
                                       mutual_info_classif)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import RocCurveDisplay, roc_auc_score, recall_score
from sklearn.model_selection import *
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, Normalizer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from helper import *


# %% load data and impute
def load_data_v2(cohort, drug):

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
    discarded_vars = [v for v in list(df.columns) if 'TEXT' in v] + \
                ['PID','PID2','AL6B','ID13','ID14_4','ID14_5','ID14_6','ID14_7','ND13','ND15_4','ND15_5','ND15_6','ND15_7',
                'DA5','DA6','DA7','DA7a','DA7b','DA7c','DA7d','DA8','DA8a','DA8b','DA8c','DA8d',
                # 'ID1','OD1','OD6','OD8','OD10','ID17','ID18','ID19','ID20','CJ3'
                ]

    dep_var_full = []
    for a, b in pred_var:
        if not np.isnan(a) and not np.isnan(b):
            y = 1 if b - a > 0 else 0
            # y = 1 if b - a > 1 else 0
            # y = a - b
            dep_var_full.append(y)
        else:   dep_var_full.append(np.nan)

    df = pd.concat([df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
    y = np.array(dep_var_full)
    y = y[~np.isnan(y)]

    X_df = df[df['pred'].notna()].drop(discarded_vars+['pred'], axis=1)
    X_df.reset_index(drop=True, inplace=True)

    return X_df, y, dep_var_full


def merge_vars(df):

    var_lists = [
        ['TB1','TB5','TB9'],
        ['TB2','TB6','TB10'],
        ['TB3','TB7','TB11'],
        ['TB4','TB8','TB12'],
        [f'ID{i}' for i in range(3,13)],
        [f'ND{i}' for i in range(1,13) if i not in [1, 2]],
        [f'DA1_{i}' for i in range(1,8)],
        [f'DA2_{i}' for i in range(1,8)],
        [f'TX1_{i}' for i in range(1,8)],
        [f'TX2_{i}' for i in range(1,8)],
        ['OD7','OD9','OD11'],
        ['AC1A','AC1B','AC2A','AC2B','AC7A','AC7B','AC11','AC12'],
        ['AC3A','AC3B','AC3C','AC3D','AC6','AC7C','AC8A','AC8B','AC9A','AC9B','AC10','AC13','AC14'],
        ['AC4A','AC4B','AC5A','AC5B']
    ]

    for var_list in var_lists:
        df['-'.join(var_list)] = df[var_list].sum(axis=1)
        df.drop(var_list, axis=1, inplace=True)
    return df


def case_var_delete_v3(cohort, drug, thresh=0.9):  # delete case and/or variable if too many missing values

    X_df, y, dep_var_full = load_data_v2(cohort, drug)
    X_dropcol_df = X_df.dropna(axis=1, thresh=len(X_df)*thresh)  # drop column (variable) if missing 10% out of all participants
    X_drop_df = X_df.dropna(axis=0, thresh=len(X_dropcol_df.columns)*thresh)  # drop row (partcipant) if missing 10% out of all features
    if len(X_drop_df) < len(X_df):  y = y[X_drop_df.index]

    nominal_vars = ['DM8','DM10','DM12','DM13','CJ4','CJ6','CJ7','ID17','ID18','ID19','ID20','OD2','OD7','OD8','OD9','OD10','OD11']
    X_ordinal_df = X_drop_df.drop(nominal_vars, axis=1)
    X_nominal_df = X_drop_df[nominal_vars]
    # Encode
    categories = [list(map(int, nnw_labelings[v][1].keys())) for v in X_ordinal_df.columns]
    encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=np.nan)
    Xenc_ordinal_df = pd.DataFrame(encoder.fit_transform(X_ordinal_df), columns=X_ordinal_df.columns)

    nominal_cols = []
    for v in nominal_vars:
        nominal_cols.append(pd.get_dummies(X_nominal_df[v], prefix=v))
    Xenc_nominal_df = pd.concat(nominal_cols, axis=1)
    Xenc_df = pd.concat([Xenc_ordinal_df.reset_index(drop=True), Xenc_nominal_df.reset_index(drop=True)], axis=1)
    # Impute
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent', verbose=100)
    X_imp = imp.fit_transform(Xenc_df)
    X_imp_df = pd.DataFrame(X_imp, columns=Xenc_df.columns)
    print(X_imp_df.shape)

    return X_imp_df.to_numpy(), y, X_imp_df, dep_var_full, list(X_drop_df.index)  # last 2 outputs only needed when including network features



# %% which feature type to use for training?
def run_nonnetwork():
    scores_dict = {}
    for cohort in cohorts:
        for drug in drugs:

            print(f'At Cohort {cohort}, {drug} using non-network features')

            X_imp, y, Xenc_df, _, _ = case_var_delete_v3(cohort, drug)  # df with dropped rows and columns if too much missingness

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            for clf_name in clf_dict.keys():
                scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'] = []
                scores_dict[f'{cohort}-{drug}-{clf_name}-recall'] = []

            # Group features
            f_dict = {}
            fgroups = ['SC', 'DM', 'TB', 'AL', 'ID', 'ND', 'DA', 'OD', 'TX', 'AC', 'CJ']  # feature groups
            for g in fgroups:
                f_indices = [Xenc_df.columns.get_loc(c) for c in Xenc_df if c.startswith(g)]  # column indices of the group's features
                f_dict[g] = f_indices

            acc_baseline = stats.mode(y)[1][0]/len(y)

            # Domain-specific feature grouping
            if 'manual' in methods:
                for gname, fsubs in fsubs_nonnet.items():
                    scores_dict[f'{cohort}-{drug}-fgroup'].append(gname)
                    X = X_imp[:, [f for fgroup in [f_dict[s] for s in fsubs] for f in fgroup]]
                    for clf_name, clf in clf_dict.items():
                        print(f'Cohort {cohort}, {drug}: start running manual grouping {gname} for {clf_name}')
                        preds, preds_proba = [], []
                        for train_idx, test_idx in cv_outer.split(X):
                            X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                            # if clf_name not in ['DT','RF','GB']:    X_train, X_test = norm_scale(X_train, X_test)
                            X_train, X_test = norm_scale(X_train, X_test)
                            search = GridSearchCV(clf, param_grid=clf_param_grid[clf_name], cv=cv_inner, refit=True, n_jobs=5, scoring=scoring)
                            search.fit(X_train, y_train)
                            if clf_name=='SVM':     preds_proba.append(search.decision_function(X_test)[0])
                            else:   preds_proba.append(search.predict_proba(X_test)[:,1][0])
                            preds.append(search.predict(X_test)[0])
                        scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'].append(roc_auc_score(y, np.array(preds_proba)))
                        scores_dict[f'{cohort}-{drug}-{clf_name}-recall'].append(recall_score(y, np.array(preds)))

            # various feature selection methods
            for m_name in methods:
                if m_name == 'manual':  continue
                scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                for clf_name, clf in clf_dict.items():
                    print(f'Cohort {cohort}, {drug}: start running {m_name} for {clf_name}')
                    if m_name == 'thresholding':    auroc, recall = thresholding(clf, clf_name, clf_param_grid[clf_name], X_imp, y, cv_inner, cv_outer, scoring)
                    if m_name == 'chi2':    auroc, recall = chi2_filter(clf, clf_name, clf_param_grid[clf_name], X_imp, y, cv_inner, cv_outer, scoring)
                    if m_name == 'pca':    auroc, recall = pca(clf, clf_name, clf_param_grid[clf_name], X_imp, y, cv_inner, cv_outer, scoring)
                    if m_name == "GA":     auroc, recall = genetic_alg(clf, clf_name, clf_param_grid[clf_name], X_imp, y, cv_inner, cv_outer, scoring)
                    if m_name == "GAmod":   auroc, recall = genetic_alg_mod(clf, clf_name, clf_params[clf_name], X_imp, y, cv_inner, cv_outer, scoring)
                    scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'].append(auroc)
                    scores_dict[f'{cohort}-{drug}-{clf_name}-recall'].append(recall)

            scores_dict[f'{cohort}-{drug}-baseline'] = [acc_baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

    pd.DataFrame.from_dict(scores_dict).to_csv(f'results/nestedCV_scores_nonnetwork_{goals_code}.csv', index=False)


def run_network():
    scores_dict = {}
    for cohort in cohorts:
        for drug in drugs:

            print(f'At Cohort {cohort}, {drug} using network features')

            _, y, _, dep_var_full, rows_idx = case_var_delete_v3(cohort, drug)  # df with dropped rows and columns if too much missingness
            net_df = pd.read_csv(f"saved-vars/C{''.join(str(cohort).split('+'))}_network_221114-processed.csv")
            net_df = pd.concat([net_df, pd.DataFrame({'pred': dep_var_full})], axis=1)
            Xnet_df = net_df[net_df['pred'].notna()].drop('pred', axis=1)  # drop rows where prediction var is missing
            Xnet_df.reset_index(drop=True, inplace=True)
            Xnet_df = Xnet_df.loc[rows_idx]  # drop rows that have been dropped by case_var_delete
            X_np = Xnet_df.to_numpy()
            

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            for clf_name in clf_dict.keys():
                scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'] = []
                scores_dict[f'{cohort}-{drug}-{clf_name}-recall'] = []

            # Group features
            f_dict = {}
            fgroups = ['NSX','NDX']  # feature groups
            for g in fgroups:
                f_indices = [Xnet_df.columns.get_loc(c) for c in Xnet_df if c.startswith(g)]  # column indices of the group's features
                f_dict[g] = f_indices

            acc_baseline = stats.mode(y)[1][0]/len(y)

            # Domain-specific feature grouping
            if 'manual' in methods:
                for gname, fsubs in fsubs_net.items():
                    scores_dict[f'{cohort}-{drug}-fgroup'].append(gname)
                    X = X_np[:, [f for fgroup in [f_dict[s] for s in fsubs] for f in fgroup]]
                    for clf_name, clf in clf_dict.items():
                        print(f'Cohort {cohort}, {drug}: start running manual grouping {gname} for {clf_name}')
                        preds, preds_proba = [], []
                        for train_idx, test_idx in cv_outer.split(X):
                            X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                            # if clf_name not in ['DT','RF','GB']:    X_train, X_test = norm_scale(X_train, X_test)
                            X_train, X_test = norm_scale(X_train, X_test)
                            search = GridSearchCV(clf, param_grid=clf_param_grid[clf_name], cv=cv_inner, refit=True, n_jobs=5, scoring=scoring)
                            search.fit(X_train, y_train)
                            if clf_name=='SVM':     preds_proba.append(search.decision_function(X_test)[0])
                            else:   preds_proba.append(search.predict_proba(X_test)[:,1][0])
                            preds.append(search.predict(X_test)[0])
                        scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'].append(roc_auc_score(y, np.array(preds_proba)))
                        scores_dict[f'{cohort}-{drug}-{clf_name}-recall'].append(recall_score(y, np.array(preds)))

            # various feature selection methods
            for m_name in methods:
                if m_name == 'manual':  continue
                scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                for clf_name, clf in clf_dict.items():
                    print(f'Cohort {cohort}, {drug}: start running {m_name} for {clf_name}')
                    if m_name == 'thresholding':    auroc, recall = thresholding(clf, clf_name, clf_param_grid[clf_name], X_np, y, cv_inner, cv_outer, scoring)
                    if m_name == 'chi2':    auroc, recall = chi2_filter(clf, clf_name, clf_param_grid[clf_name], X_np, y, cv_inner, cv_outer, scoring)
                    if m_name == 'pca':    auroc, recall = pca(clf, clf_name, clf_param_grid[clf_name], X_np, y, cv_inner, cv_outer, scoring)
                    if m_name == "GA":     auroc, recall = genetic_alg(clf, clf_name, clf_param_grid[clf_name], X_np, y, cv_inner, cv_outer, scoring)
                    if m_name == "GAmod":   auroc, recall = genetic_alg_mod(clf, clf_name, clf_params[clf_name], X_np, y, cv_inner, cv_outer, scoring)
                    scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'].append(auroc)
                    scores_dict[f'{cohort}-{drug}-{clf_name}-recall'].append(recall)

            scores_dict[f'{cohort}-{drug}-baseline'] = [acc_baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

    pd.DataFrame.from_dict(scores_dict).to_csv(f'results/nestedCV_scores_network_{goals_code}.csv', index=False)


def run_netnonnetwork():
    scores_dict = {}
    for cohort in cohorts:
        for drug in drugs:

            print(f'At Cohort {cohort}, {drug} using network + non-network features')

            #---------------------------------- Non-network ------------------------------------------------------
            X_imp, y, Xenc_df, dep_var_full, rows_idx = case_var_delete_v3(cohort, drug)  # df with dropped rows and columns if too much missingness

            #---------------------------------- Network ------------------------------------------------------
            net_df = pd.read_csv(f"saved-vars/C{''.join(str(cohort).split('+'))}_network_221114-processed.csv")
            net_df = pd.concat([net_df, pd.DataFrame({'pred': dep_var_full})], axis=1)
            Xnet_df = net_df[net_df['pred'].notna()].drop('pred', axis=1)  # drop rows where prediction var is missing
            Xnet_df.reset_index(drop=True, inplace=True)
            Xnet_df = Xnet_df.loc[rows_idx]  # drop rows that have been dropped by case_var_delete
            X_np = Xnet_df.to_numpy()

            #-------------------------------------------------------------------------------------------
            X_all = np.concatenate((X_imp, X_np), axis=1)
            

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            for clf_name in clf_dict.keys():
                scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'] = []
                scores_dict[f'{cohort}-{drug}-{clf_name}-recall'] = []

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

            acc_baseline = stats.mode(y)[1][0]/len(y)

            # Domain-specific feature grouping
            if 'manual' in methods:
                for gname, fsubs in fsubs_netnonnet.items():
                    scores_dict[f'{cohort}-{drug}-fgroup'].append(gname)
                    X_nonet = X_imp[:, [f for fgroup in [f_nonet_dict[s] for s in fsubs if s in fgroups_nonet] for f in fgroup]]
                    X_net = X_np[:, [f for fgroup in [f_net_dict[s] for s in fsubs if s in fgroups_net] for f in fgroup]]
                    X = np.concatenate((X_nonet, X_net), axis=1)
                    for clf_name, clf in clf_dict.items():
                        print(f'Cohort {cohort}, {drug}: start running manual grouping {gname} for {clf_name}')
                        preds, preds_proba = [], []
                        for train_idx, test_idx in cv_outer.split(X):
                            X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                            # if clf_name not in ['DT','RF','GB']:    X_train, X_test = norm_scale(X_train, X_test)
                            X_train, X_test = norm_scale(X_train, X_test)
                            search = GridSearchCV(clf, param_grid=clf_param_grid[clf_name], cv=cv_inner, refit=True, n_jobs=5, scoring=scoring)
                            search.fit(X_train, y_train)
                            if clf_name=='SVM':     preds_proba.append(search.decision_function(X_test)[0])
                            else:   preds_proba.append(search.predict_proba(X_test)[:,1][0])
                            preds.append(search.predict(X_test)[0])
                        scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'].append(roc_auc_score(y, np.array(preds_proba)))
                        scores_dict[f'{cohort}-{drug}-{clf_name}-recall'].append(recall_score(y, np.array(preds)))

            # various feature selection methods
            for m_name in methods:
                if m_name == 'manual':  continue
                scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                for clf_name, clf in clf_dict.items():
                    print(f'Cohort {cohort}, {drug}: start running {m_name} for {clf_name}')
                    if m_name == 'thresholding':    auroc, recall = thresholding(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer, scoring)
                    if m_name == 'chi2':    auroc, recall = chi2_filter(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer, scoring)
                    if m_name == 'pca':    auroc, recall = pca(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer, scoring)
                    if m_name == "GA":     auroc, recall = genetic_alg(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer, scoring)
                    if m_name == "GAmod":   auroc, recall = genetic_alg_mod(clf, clf_name, clf_params[clf_name], X_all, y, cv_inner, cv_outer, scoring)
                    scores_dict[f'{cohort}-{drug}-{clf_name}-AUROC'].append(auroc)
                    scores_dict[f'{cohort}-{drug}-{clf_name}-recall'].append(recall)

            scores_dict[f'{cohort}-{drug}-baseline'] = [acc_baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

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
    cohorts, drugs, methods, clf_list, scoring = goals["cohorts"], goals["drugs"], goals["methods"], goals["clf_list"], goals["scoring"]

    with open('saved-vars/labelings_non-network.json', 'r') as f:
        nnw_labelings = json.load(f)


    # %% domain-specific feature groupings
    fsubs_1, fsubs_2, fsubs_3, fsubs_4 = ['SC'], ['DM'], ['TB','AL','ID','ND','DA','OD'], ['TX','AC','CJ']
    fsubs_5 = fsubs_1 + fsubs_2
    fsubs_6 = fsubs_1 + fsubs_2 + fsubs_3
    fsubs_7 = fsubs_1 + fsubs_2 + fsubs_4
    fsubs_8 = fsubs_1 + fsubs_2 + fsubs_3 + fsubs_4
    fsubs_nonnet = {
        'g1': fsubs_1,
        'g2': fsubs_2,
        'g3': fsubs_3,
        'g4': fsubs_4,
        'g5': fsubs_5,
        'g6': fsubs_6,
        'g7': fsubs_7,
        'g8': fsubs_8
        }

    fsubs_net = {'g9': ['NSX'], 'g10': ['NDX'], 'g11': ['NSX','NDX']}

    fsubs_netnonnet = { 'g12': fsubs_3+fsubs_net['g10'],
                        'g13': fsubs_6+fsubs_net['g10'],
                        'g14': fsubs_8+fsubs_net['g11']}


    # %% classifiers considered
    clf_choices = {
        'LG_L1': LogisticRegression(solver='saga', penalty='l1'),
        'LG_L2': LogisticRegression(solver='saga', penalty='l2'),
        'LG_EN': LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5),
        'DT': DecisionTreeClassifier(),
        'SVM': SVC(cache_size=1000),
        'RF': RandomForestClassifier(),
        'GB': GradientBoostingClassifier(),
    }
    clf_params = {
        'LG_L1': {'names': ['C'], 'range': [(0.001, 1000)], 'bitwidth': 8},
        'LG_L2': {'names': ['C'], 'range': [(0.001, 1000)], 'bitwidth': 8},
        'LG_EN': {'names': ['C'], 'range': [(0.001, 1000)], 'bitwidth': 8},
        'DT': {'names': ['max_depth','min_samples_split'], 'range': [(3, 10), (5,15)], 'bitwidth': 4},
        'SVM': {'names': ['gamma','C'], 'range': [(0.0001, 100), (0.001, 1000)], 'bitwidth': 8},
    }
    clf_param_grid = {
        'LG_L1': dict(C=np.logspace(-3,3,num=7)),
        'LG_L2': dict(C=np.logspace(-3,3,num=7)),
        'LG_EN': dict(C=np.logspace(-3,3,num=7)),
        'DT': dict(max_depth=range(3,11), min_samples_split=[5, 10, 15]),
        'SVM': dict(gamma=np.logspace(-4,2,num=7), C=np.logspace(-3,3,num=7)),
        'RF': dict(max_depth=[3,5,7,10,None]),
        'GB': dict(max_depth=[3,5,7,10]),
    }
    clf_dict = {clf_name: clf for clf_name, clf in clf_choices.items() if clf_name in clf_list}
    cv_inner = int(goals["cv_inner"])
    cv_outer = LeaveOneOut() if goals["cv_outer"] == "LOO" else KFold(int(goals["cv_outer"]))
    goals_code = f"{'-'.join([str(c) for c in cohorts])}_{'-'.join([d[:4] for d in drugs])}_{'-'.join([m[:3] for m in methods])}_{'-'.join([clf_name for clf_name in clf_dict.keys()])}_{scoring}"


    # %% non-network features
    if int(sys.argv[1]):
        run_nonnetwork()

    # %% network features
    if int(sys.argv[2]):
        run_network()

    # %% network+nonnetwork features
    if int(sys.argv[3]):
        run_netnonnetwork()
        
        