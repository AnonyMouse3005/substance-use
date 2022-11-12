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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import *
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from genetic_selection import GeneticSelectionCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from helper import *


# %%
def load_and_impute(cohort, encode=True, impute=True, drug='marijuana'):

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
        
    df = impute_MARs(nonet_vars, nonet_df)
    discarded_vars = ['PID','PID2','AL6B','ID13','ID14_4','ID14_5','ID14_6','ID14_7','ND13','ND15_4','ND15_5','ND15_6','ND15_7',
                'DA5','DA6','DA7','DA7a','DA7b','DA7c','DA7d','DA8','DA8a','DA8b','DA8c','DA8d'] + [v for v in list(df.columns) if 'TEXT' in v]
    nominal_vars = ['DM8','DM10','DM12','DM13']

    dep_var_full = []
    for a, b in pred_var:
        if not np.isnan(a) and not np.isnan(b):
            y = 0 if a <= b else 1
            # y = a - b
            dep_var_full.append(y)
        else:   dep_var_full.append(np.nan)

    df = pd.concat([df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
    y = np.array(dep_var_full)
    y = y[~np.isnan(y)]

    X_df = df.drop(discarded_vars, axis=1)
    X_ordinal_df = X_df.drop(nominal_vars, axis=1)
    X_nominal_df = X_df[nominal_vars]

    # Encode
    Xenc_ordinal_df = X_ordinal_df.astype('str').apply(LabelEncoder().fit_transform)
    Xenc_ordinal_df = Xenc_ordinal_df.where(~X_ordinal_df.isna(), X_ordinal_df)  # Do not encode the NaNs

    nominal_cols = []
    for v in nominal_vars:
        nominal_cols.append(pd.get_dummies(X_nominal_df[v], prefix=v))
    Xenc_nominal_df = pd.concat(nominal_cols, axis=1)

    Xenc_df = pd.concat([Xenc_ordinal_df, Xenc_nominal_df], axis=1)

    if impute:
        # Impute
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_imp = imp.fit_transform(Xenc_df)
        return X_imp, y
    else:
        if encode:      return Xenc_df, y
        else:           return X_df, y


def plot_missingness(cohort, save=False):

    X_df, _ = load_and_impute(cohort, encode=False, impute=False)

    plt.subplots(figsize=(50,15), tight_layout=True)
    ax = sns.heatmap(X_df.isna(), cbar=False)
    plt.xlabel('Non-network variable', fontsize=30)
    plt.ylabel('Participant', fontsize=30)
    if cohort == "1+2":     ax.hlines([35], *ax.get_xlim(), color='r')
    plt.xticks(rotation=90)
    
    if save:
        plt.savefig(f'plots/analysis/visualization/missingness/{cohort}_missingness.pdf', facecolor='white')


def plot_distribution(save=False):

    with open('saved-vars/labelings_non-network.json', 'r') as f:  # load dict containing lists of cohorts, drugs, and methods to be investigated
        mappings = json.load(f)

    X_c1_df, _ = load_and_impute(cohort=1, encode=False, impute=False)
    X_c2_df, _ = load_and_impute(cohort=2, encode=False, impute=False)

    num_dict = {'DM1': {"0": "Children [0-14]", "1": "Youth [15-24]", "2": "Adult [25-64]", "3": "Senior (>65)"},
            'SC1': {"0": "Less than 6 months", "1": "6 months to less than a year", "2": "A year to less than 2 years",
                "3": "2 years to less than 5 years", "4": "5 years to less than 10 years", "5": "10 years or more"}}
    for v_name, v_info in mappings.items():

        v_label = v_info[0]
        v_cate = num_dict[v_name] if v_name in ['SC1', 'DM1'] else v_info[1]

        X_c1 = X_c1_df.loc[:,v_name]
        X_c2 = X_c2_df.loc[:,v_name]

        X_c1 = [v_cate[str(int(entry))] for entry in sorted(X_c1) if not np.isnan(entry)]
        X_c2 = [v_cate[str(int(entry))] for entry in sorted(X_c2) if not np.isnan(entry)]

        c1_df = pd.DataFrame({'C1': pd.Categorical(X_c1, categories=list(v_cate.values()))}).value_counts(sort=False, normalize=True)
        c2_df = pd.DataFrame({'C2': pd.Categorical(X_c2, categories=list(v_cate.values()))}).value_counts(sort=False, normalize=True)
        df = pd.concat([c1_df, c2_df], keys=['C1', 'C2'], axis=1)
        ax = df.plot.bar(figsize=(10,7))
        ax.set_ylabel("Frequency")
        plt.title(v_label, fontdict={'fontsize' : 10})
        plt.tight_layout()
        if save:
            plt.savefig(f'plots/analysis/visualization/distribution/{v_name}_distribution.pdf', bbox_inches='tight', facecolor='white')


def case_var_delete(cohort, thresh=0.9, drug='marijuana'):  # delete case or variable if too many missing values
    X_df, y = load_and_impute(cohort, encode=False, impute=False, drug=drug)
    X_droprow_df = X_df.dropna(axis=0, thresh=len(X_df.columns)*thresh)  # drop row (partcipant) if missing 10% out of all features
    X_drop_df = X_droprow_df.dropna(axis=1, thresh=len(X_df)*thresh)  # drop column (variable) if missing 10% out of all participants
    return X_drop_df, y


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


# %% main
if __name__ == '__main__':

    datapath = 'data/original/pre-imputed/'
    C1W1nonet_df = pd.read_csv(datapath + 'C1W1_nonnetwork_preimputed.csv')
    C1pred_df = pd.read_csv(datapath + 'C1_nonnetwork_pred.csv')
    C1W1nonet_vars = list(C1W1nonet_df.columns)
    C2W1nonet_df = pd.read_csv(datapath + 'C2W1_nonnetwork_preimputed.csv')
    C2pred_df = pd.read_csv(datapath + 'C2_nonnetwork_pred.csv')
    C2W1nonet_vars = list(C2W1nonet_df.columns)

    # %% plot missingness of (non-network) data
    cohorts = [1, 2, "1+2"]
    for cohort in cohorts:
        plot_missingness(cohort)


    # %% plot distribution of C1 vs C2 (non-network data)
    plot_distribution()


    # %% case and variable deletion
    X_drop_df, y = case_var_delete(cohort=1, drug='marijuana')
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
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X = imp.fit_transform(Xenc_df)

    clf_dict = {
        # 'LG_L1': LogisticRegression(solver='saga', penalty='l1'),
        # 'LG_L2': LogisticRegression(solver='saga', penalty='l2'),
        # 'LG_EN': LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5),
        'DT': DecisionTreeClassifier(),
        'SVM': SVC(cache_size=1000)
    }
    clf_param_grid = {
        'LG_L1': dict(C=np.logspace(-3,3,num=7)),
        'LG_L2': dict(C=np.logspace(-3,3,num=7)),
        'LG_EN': dict(C=np.logspace(-3,3,num=7)),
        'DT': dict(max_depth=range(3,11), min_samples_split=[5, 10, 15]),
        'SVM': dict(gamma=np.logspace(-4,2,num=7), C=np.logspace(-3,3,num=7))
    }
    cv_inner, cv_outer = 10, LeaveOneOut()

    # results = {}
    # for clf_name, clf in clf_dict.items():
    #     scores = []
    #     for train_idx, test_idx in cv_outer.split(X):
    #         X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    #         if clf_name != 'DT':
    #             X_train = standard_scale(X_train)
    #             X_test = standard_scale(X_test)
    #         search = GridSearchCV(clf, param_grid=clf_param_grid[clf_name], cv=cv_inner, refit=True, n_jobs=5)
    #         search.fit(X_train, y_train)
    #         scores.append(search.score(X_test, y_test))
    #     results[clf_name] = np.mean(scores)
    # results['X_shape'] = X.shape

    results = {}
    for clf_name, clf in clf_dict.items():
        results[clf_name] = genetic_alg(clf, clf_name, clf_param_grid[clf_name], X, y, cv_inner, cv_outer)

# %%
