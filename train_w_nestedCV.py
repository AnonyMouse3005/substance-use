# %%
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
from sklearn.feature_selection import (SelectFromModel, SelectPercentile, chi2,
                                       mutual_info_classif)
from sklearn.impute import SimpleImputer
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
from genetic_selection import GeneticSelectionCV
sys.path.insert(0, '/home/nthach17/repo/sklearn-genetic-mod')
from genetic_selection_mod import GeneticSelectionCV_mod


# %% boilerplate
if len(sys.argv) != 5:
    print('Usage: python3 train_w_nestedCV.py [non-network?] [network?] [non-network+network?] [method_list]')
    sys.exit(1)


# %%
@ignore_warnings(category=ConvergenceWarning)
def genetic_alg_mod(clf, clf_name, hparams, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    print(f'classifier: {clf_name}')
    X = standard_scale(X_raw) if clf_name != 'decision tree' else X_raw
    for train_idx, test_idx in cv_outer.split(X):

        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        model = GeneticSelectionCV_mod(
            clf, cv=cv, verbose=2,
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
                if k in ['max_depth']:      v = int(round(v, 0))
                setattr(clf, k, v)

        results.append(clf.fit(X_train_new, y_train).score(X_test_new, y_test))
    
    return np.mean(results)


@ignore_warnings(category=ConvergenceWarning)
def genetic_alg(clf, clf_name, hparams_grid, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    print(f'classifier: {clf_name}')
    X = standard_scale(X_raw) if clf_name != 'decision tree' else X_raw
    for train_idx, test_idx in cv_outer.split(X):

        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        model = GeneticSelectionCV(
            clf, cv=cv, verbose=2,
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
def pca(clf, clf_name, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    for train_idx, test_idx in cv_outer.split(X_raw):
        X_train, y_train, X_test, y_test = X_raw[train_idx], y[train_idx], X_raw[test_idx], y[test_idx]
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("pca", PCA()), (clf_name, clf)])
        param_grid = {"pca__n_components": np.linspace(2, 30, 10, dtype=np.int32)}
        search = GridSearchCV(pipe, param_grid, cv=cv, refit=True)
        search.fit(X_train, y_train)

        results.append(search.score(X_test, y_test))

    return np.mean(results)


@ignore_warnings(category=ConvergenceWarning)
def chi2_filter(clf, clf_name, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    for train_idx, test_idx in cv_outer.split(X_raw):
        X_train, y_train, X_test, y_test = X_raw[train_idx], y[train_idx], X_raw[test_idx], y[test_idx]
        pipe = Pipeline(steps=[("anova", SelectPercentile(chi2)), (clf_name, clf)]) if clf_name == 'decision tree' \
            else Pipeline(steps=[("anova", SelectPercentile(chi2)), ("scaler", StandardScaler()), (clf_name, clf)])
        param_grid = {"anova__percentile": [3, 6, 10, 15, 20, 30, 40]}
        search = GridSearchCV(pipe, param_grid, cv=cv, refit=True)
        search.fit(X_train, y_train)
        
        if clf_name != 'decision tree':     X_test = standard_scale(X_test)
        X_test_new = X_test[:,search.best_estimator_['anova'].get_support()]

        results.append(search.best_estimator_[clf_name].score(X_test_new, y_test))

    return np.mean(results)


@ignore_warnings(category=ConvergenceWarning)
def thresholding(clf, clf_name, X_raw, y, cv=10, cv_outer=LeaveOneOut()):

    results = []
    thresholds = [f"{scale}*mean" for scale in [0.1, 0.5, 0.75, 1, 1.25, 1.5, 2]]
    X = standard_scale(X_raw) if clf_name != 'decision tree' else X_raw
    for train_idx, test_idx in cv_outer.split(X_raw):
        
        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        if clf_name == 'SVM':       clf.kernel = 'linear'  # only linear kernel allows SVM to have coef_
        importances = 'auto'
        cv_scores = {}
        for j, th in enumerate(thresholds):
            model = SelectFromModel(clf.fit(X_train, y_train), prefit=True, importance_getter=importances, threshold=th)
            X_train_new, X_test_new = model.transform(X_train), model.transform(X_test)
            cv_scores[j] = {'score': np.mean(cross_val_score(clf, X_train_new, y_train, scoring='accuracy', cv=cv)),
                            'X_train_new': X_train_new, 'X_test_new': X_test_new}

        best_threshold_idx = np.argmax([i['score'] for i in cv_scores.values()])
        X_train_best, X_test_best = cv_scores[best_threshold_idx]['X_train_new'], cv_scores[best_threshold_idx]['X_test_new']
        results.append(clf.fit(X_train_best, y_train).score(X_test_best, y_test))

    return np.mean(results)


# regularization method


# %% load dataset
datapath = 'data/original/pre-imputed/'

C1W1nonet_df = pd.read_csv(datapath + 'C1W1_nonnetwork_preimputed.csv')
C1pred_df = pd.read_csv(datapath + 'C1_nonnetwork_pred.csv')
C1W1nonet_vars = list(C1W1nonet_df.columns)

C2W1nonet_df = pd.read_csv(datapath + 'C2W1_nonnetwork_preimputed.csv')
C2pred_df = pd.read_csv(datapath + 'C2_nonnetwork_pred.csv')
C2W1nonet_vars = list(C2W1nonet_df.columns)

with open(sys.argv[4], 'r') as f:  # load list containing names of model building methods
    method_list = json.load(f)


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


# %% non-network features
if int(sys.argv[1]):

    clf_dict = {
        # 'logistic regression': LogisticRegression(),
        # 'decision tree': DecisionTreeClassifier(),
        'SVM': SVC()}
    scores_dict = {}
    for cohort in ['1+2']:
        for drug in ['meth']:

            print(f'At Cohort {cohort}, {drug}')

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            scores_dict[f'{cohort}-{drug}-LG'], scores_dict[f'{cohort}-{drug}-DT'], scores_dict[f'{cohort}-{drug}-SVM'] = [], [], []

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
            nominal_vars = ['DM8','DM10','DM12','DM13']
            indep_vars = list(df.drop(columns=discarded_vars).columns)
            
            dep_var_full = []
            for a, b in pred_var:
                if not np.isnan(a) and not np.isnan(b):
                    y = 0 if a <= b else 1
                    # y = a - b
                    dep_var_full.append(y)
                else:   dep_var_full.append(np.nan)

            df = pd.concat([df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
            X_df = df[df['pred'].notna()].drop(discarded_vars+['pred'], axis=1)
            X_ordinal_df = X_df.drop(nominal_vars, axis=1)
            X_nominal_df = X_df[nominal_vars]

            # Encode
            Xenc_ordinal_df = X_ordinal_df.astype('str').apply(LabelEncoder().fit_transform)
            Xenc_ordinal_df = Xenc_ordinal_df.where(~X_ordinal_df.isna(), X_ordinal_df)  # Do not encode the NaNs

            nominal_cols =[]
            for v in nominal_vars:
                nominal_cols.append(pd.get_dummies(X_nominal_df[v], prefix=v))
            Xenc_nominal_df = pd.concat(nominal_cols, axis=1)

            Xenc_df = pd.concat([Xenc_ordinal_df, Xenc_nominal_df], axis=1)

            # Group features
            f_dict = {}
            fgroups = ['SC', 'DM', 'TB', 'AL', 'ID', 'ND', 'DA', 'OD', 'TX', 'AC', 'CJ']  # feature groups
            for g in fgroups:
                features = [c for c in Xenc_df if c.startswith(g)]
                f_indices = [Xenc_df.columns.get_loc(c) for c in Xenc_df if c.startswith(g)]  # column indices of the group's features
                f_dict[g] = f_indices

            # Impute
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_imp = imp.fit_transform(Xenc_df)

            # Learning curve
            y = np.array(dep_var_full)
            y = y[~np.isnan(y)]
            baseline = stats.mode(y)[1][0]/len(y)

            # various feature selection methods
            for m_name, fselect_method in {'genetic alg': genetic_alg}.items():
                print(f'Start running {m_name}')
                results = fselect_method(clf_dict, Xenc_df, X_imp, y, cohort, drug, baseline, 'plots/analysis/learning_curve/feature_selection')
                
                if m_name != 'pca':
                    # scores_dict[f'{cohort}-{drug}-fgroup'].extend([m_name, np.nan])
                    # scores_dict[f'{cohort}-{drug}-LG'].extend([results['logistic regression']['cv_score'], results['logistic regression']['features']])
                    # scores_dict[f'{cohort}-{drug}-DT'].extend([results['decision tree']['cv_score'], results['decision tree']['features']])
                    scores_dict[f'{cohort}-{drug}-SVM'].extend([results['SVM']['cv_score'], results['SVM']['features']])
                else:
                    # scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                    # scores_dict[f'{cohort}-{drug}-LG'].append(results['logistic regression']['cv_score'])
                    # scores_dict[f'{cohort}-{drug}-DT'].append(results['decision tree']['cv_score'])
                    scores_dict[f'{cohort}-{drug}-SVM'].append(results['SVM']['cv_score'])

            scores_dict[f'{cohort}-{drug}-baseline'] = [baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

    # pd.DataFrame.from_dict(scores_dict).to_csv('results/scores_nonnetwork.csv', index=False)


# %% network features
if int(sys.argv[2]):

    clf_dict = {'logistic regression': LogisticRegression(),
                'decision tree': DecisionTreeClassifier(),
                'SVM': SVC()}
    scores_dict = {}
    for cohort in [1, 2, '1+2']:
        for drug in ['marijuana', 'meth']:

            print(f'At Cohort {cohort}, {drug}')

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            scores_dict[f'{cohort}-{drug}-LG'], scores_dict[f'{cohort}-{drug}-DT'], scores_dict[f'{cohort}-{drug}-SVM'] = [], [], []

            # csv generated from data_net_analysis.ipynb (Cramer's V section)
            df = pd.read_csv(f"saved-vars/C{''.join(str(cohort).split('+'))}_network-processed.csv")
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
                features = [c for c in X_df if c.startswith(g) or g[:2] in c]
                f_indices = [X_df.columns.get_loc(c) for c in X_df if c.startswith(g)]  # column indices of the group's features
                f_dict[g] = f_indices

            X_np = X_df.to_numpy()

            # Learning curve
            y = np.array(dep_var_full)
            y = y[~np.isnan(y)]
            baseline = stats.mode(y)[1][0]/len(y)

            # Domain-specific feature grouping
            # for gname, fsubs in fsubs_net.items():
            #     X = X_np[:, [f for fgroup in [f_dict[s] for s in fsubs] for f in fgroup]]
            #     savefile = f'plots/analysis/learning_curve/net/C{cohort}-{drug}/C{cohort}-{drug}-{gname}_learningCurve.pdf'
            #     lg_score, dt_score, svm_score = plot_LC_tuned(X, y, cohort, drug, gname, savefile=savefile, baseline=baseline)
            #     scores_dict[f'{cohort}-{drug}-fgroup'].append(gname)
            #     scores_dict[f'{cohort}-{drug}-LG'].append(lg_score)
            #     scores_dict[f'{cohort}-{drug}-DT'].append(dt_score)
            #     scores_dict[f'{cohort}-{drug}-SVM'].append(svm_score)

            # various feature selection methods
            for m_name, fselect_method in {'pca': pca, 'chi2': chi2_filter, 'thresholding': thresholding, 'genetic alg': genetic_alg}.items():
                print(f'Start running {m_name}')
                results = fselect_method(clf_dict, X_df, X_np, y, cohort, drug, baseline, 'plots/analysis/learning_curve/net/feature_selection')
                
                if m_name != 'pca':
                    scores_dict[f'{cohort}-{drug}-fgroup'].extend([m_name, np.nan])
                    scores_dict[f'{cohort}-{drug}-LG'].extend([results['logistic regression']['cv_score'], results['logistic regression']['features']])
                    scores_dict[f'{cohort}-{drug}-DT'].extend([results['decision tree']['cv_score'], results['decision tree']['features']])
                    scores_dict[f'{cohort}-{drug}-SVM'].extend([results['SVM']['cv_score'], results['SVM']['features']])
                else:
                    scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                    scores_dict[f'{cohort}-{drug}-LG'].append(results['logistic regression']['cv_score'])
                    scores_dict[f'{cohort}-{drug}-DT'].append(results['decision tree']['cv_score'])
                    scores_dict[f'{cohort}-{drug}-SVM'].append(results['SVM']['cv_score'])

            scores_dict[f'{cohort}-{drug}-baseline'] = [baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

    pd.DataFrame.from_dict(scores_dict).to_csv('results/scores_network.csv', index=False)


# %% network+nonnetwork features
if int(sys.argv[3]):

    clf_dict = {
        'LG': LogisticRegression(),
        'DT': DecisionTreeClassifier(),
        'SVM': SVC()
    }
    clf_params = {
        'LG': {'names': ['C'], 'range': [(0.01, 10000)], 'bitwidth': 8},
        'DT': {'names': ['max_depth'], 'range': [(3, 10)], 'bitwidth': 4},
        'SVM': {'names': ['gamma','C'], 'range': [(0.0001, 10), (1, 1000)], 'bitwidth': 8}
    }
    clf_param_grid = {
        'LG': dict(C=np.logspace(-4,4,10)),
        'DT': dict(max_depth=range(3,10)),
        'SVM': dict(kernel='rbf', C=np.logspace(-4,4,10))
    }
    scores_dict = {}
    for cohort in [1,2,'1+2']:
        for drug in ['marijuana','meth']:

            print(f'At Cohort {cohort}, {drug}')

            scores_dict[f'{cohort}-{drug}-fgroup'] = []
            scores_dict[f'{cohort}-{drug}-LG'], scores_dict[f'{cohort}-{drug}-DT'], scores_dict[f'{cohort}-{drug}-SVM'] = [], [], []

            net_df = pd.read_csv(f"saved-vars/C{''.join(str(cohort).split('+'))}_network-processed.csv")
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

            nonet_df = impute_MARs(nonet_vars, nonet_df)
            discarded_vars = ['PID','PID2','AL6B','ID13','ID14_4','ID14_5','ID14_6','ID14_7','ND13','ND15_4','ND15_5','ND15_6','ND15_7',
                        'DA5','DA6','DA7','DA7a','DA7b','DA7c','DA7d','DA8','DA8a','DA8b','DA8c','DA8d'] + [v for v in list(nonet_df.columns) if 'TEXT' in v]
            nominal_vars = ['DM8','DM10','DM12','DM13']
            
            dep_var_full = []
            for a, b in pred_var:
                if not np.isnan(a) and not np.isnan(b):
                    y = 0 if a <= b else 1
                    # y = a - b
                    dep_var_full.append(y)
                else:   dep_var_full.append(np.nan)

            #---------------------------------- Non-network ------------------------------------------------------
            nonet_df = pd.concat([nonet_df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
            Xnonet_df = nonet_df[nonet_df['pred'].notna()].drop(discarded_vars+['pred'], axis=1)
            X_ordinal_df = Xnonet_df.drop(nominal_vars, axis=1)
            X_nominal_df = Xnonet_df[nominal_vars]

            # Encode
            Xenc_ordinal_df = X_ordinal_df.astype('str').apply(LabelEncoder().fit_transform)
            Xenc_ordinal_df = Xenc_ordinal_df.where(~X_ordinal_df.isna(), X_ordinal_df)  # Do not encode the NaNs

            nominal_cols =[]
            for v in nominal_vars:
                nominal_cols.append(pd.get_dummies(X_nominal_df[v], prefix=v))
            Xenc_nominal_df = pd.concat(nominal_cols, axis=1)

            Xenc_df = pd.concat([Xenc_ordinal_df, Xenc_nominal_df], axis=1)

            # Group non-network features
            f_nonet_dict = {}
            fgroups_nonet = ['SC', 'DM', 'TB', 'AL', 'ID', 'ND', 'DA', 'OD', 'TX', 'AC', 'CJ']  # feature groups
            for g in fgroups_nonet:
                features = [c for c in Xenc_df if c.startswith(g)]
                f_indices = [Xenc_df.columns.get_loc(c) for c in Xenc_df if c.startswith(g)]  # column indices of the group's features
                f_nonet_dict[g] = f_indices

            # Impute
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_imp = imp.fit_transform(Xenc_df)

            #---------------------------------- Network ------------------------------------------------------
            net_df = pd.concat([net_df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
            Xnet_df = net_df[net_df['pred'].notna()].drop('pred', axis=1)

            # Group network features
            f_net_dict = {}
            fgroups_net = ['NSX','NDX']  # feature groups
            for g in fgroups_net:
                features = [c for c in Xnet_df if c.startswith(g) or g[:2] in c]
                f_indices = [Xnet_df.columns.get_loc(c) for c in Xnet_df if c.startswith(g)]  # column indices of the group's features
                f_net_dict[g] = f_indices

            X_np = Xnet_df.to_numpy()

            #-------------------------------------------------------------------------------------------
            X_all = np.concatenate((X_imp, X_np), axis=1)
            X_df = pd.concat([Xenc_df, Xnet_df], axis=1)

            # Nested CV
            y = np.array(dep_var_full)
            y = y[~np.isnan(y)]
            baseline = stats.mode(y)[1][0]/len(y)
            cv_inner, cv_outer = 10, LeaveOneOut()

            # Domain-specific feature grouping
            for gname, fsubs in fsubs_netnonnet.items():
                X_nonet = X_imp[:, [f for fgroup in [f_nonet_dict[s] for s in fsubs if s in fgroups_nonet] for f in fgroup]]
                X_net = X_np[:, [f for fgroup in [f_net_dict[s] for s in fsubs if s in fgroups_net] for f in fgroup]]
                X = np.concatenate((X_nonet, X_net), axis=1)
                for clf_name, clf in clf_dict.items():
                    scores = []
                    for train_idx, test_idx in cv_outer.split(X):
                        X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                        if clf_name != 'decision tree':
                            X_train = standard_scale(X_train)
                            X_test = standard_scale(X_test)
                        clf.fit(X_train, y_train)
                        scores.append(clf.score(X_test, y_test))
                    scores_dict[f'{cohort}-{drug}-{clf_name}'].append(np.mean(scores))

            # various feature selection methods
            for m_name in method_list:
                scores_dict[f'{cohort}-{drug}-fgroup'].append(m_name)
                for clf_name, clf in clf_dict.items():
                    print(f'Start running {m_name} for {clf_name}')
                    if m_name == 'thresholding':    final_score = thresholding(clf, clf_name, X_all, y, cv_inner, cv_outer)
                    if m_name == 'chi2':    final_score = chi2_filter(clf, clf_name, X_all, y, cv_inner, cv_outer)
                    if m_name == 'pca':    final_score = pca(clf, clf_name, X_all, y, cv_inner, cv_outer)
                    if m_name == "genetic alg":     final_score = genetic_alg(clf, clf_name, clf_param_grid[clf_name], X_all, y, cv_inner, cv_outer)
                    if m_name == "genetic alg (mod)":   final_score = genetic_alg_mod(clf, clf_name, clf_params[clf_name], X_all, y, cv_inner, cv_outer)
                    scores_dict[f'{cohort}-{drug}-{clf_name}'].append(final_score)

            scores_dict[f'{cohort}-{drug}-baseline'] = [baseline] * len(scores_dict[f'{cohort}-{drug}-fgroup'])

    pd.DataFrame.from_dict(scores_dict).to_csv('results/nested_CV_scores_nonnetwork+network.csv', index=False)