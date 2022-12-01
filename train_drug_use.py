# %% train using C2W1 data to predict current [drug] use only, instead of future use
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
from sklearn.metrics import RocCurveDisplay, recall_score, accuracy_score
from sklearn.model_selection import *
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from helper import *


# %% load data and impute
def load_data_v3(cohort, drug, threshold):

    drug_key = 'ND1' if drug == 'marijuana' else 'ND7'
    if cohort == 1:
        nonet_vars, nonet_df = C1W1nonet_vars, C1W1nonet_df
    elif cohort == 2:
        nonet_vars, nonet_df = C2W1nonet_vars, C2W1nonet_df
    elif cohort == '1+2':
        nonet_df = pd.concat([C1W1nonet_df, C2W1nonet_df], ignore_index=True)
        nonet_vars = C1W1nonet_vars  # same set of columns for both cohorts
    pred_var = nonet_df[drug_key]
        
    df = impute_MARs(nonet_vars, nonet_df)
    discarded_vars = [v for v in list(df.columns) if 'TEXT' in v] + \
                ['PID','PID2','AL6B','ID13','ID14_4','ID14_5','ID14_6','ID14_7','ND13','ND15_4','ND15_5','ND15_6','ND15_7',
                'DA5','DA6','DA7','DA7a','DA7b','DA7c','DA7d','DA8','DA8a','DA8b','DA8c','DA8d',
                ]

    y = np.array(pred_var)
    y = y[~np.isnan(y)]
    lbl_enc = LabelEncoder().fit(list(map(int, nnw_labelings[drug_key][1].keys())))
    y = lbl_enc.transform(y)

    if threshold is not None:  # threshold for binary classification
        for idx, ele in enumerate(y):
            y[idx] = 1 if ele > threshold else 0

    X_df = df[df[drug_key].notna()].drop(discarded_vars+[drug_key], axis=1)  # drop rows where target var is missing
    X_df.reset_index(drop=True, inplace=True)

    return X_df, y


def case_var_delete_v3(cohort, drug, thresh=0.9, threshold=None):  # delete case and/or variable if too many missing values

    X_df, y = load_data_v3(cohort, drug, threshold)
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

    return X_imp_df.to_numpy(), y, X_imp_df


# %%
datapath = 'data/original/pre-imputed/'
C1W1nonet_df = pd.read_csv(datapath + 'C1W1_nonnetwork_preimputed.csv')
C1pred_df = pd.read_csv(datapath + 'C1_nonnetwork_pred.csv')
C1W1nonet_vars = list(C1W1nonet_df.columns)
C2W1nonet_df = pd.read_csv(datapath + '221121/C2W1_nonnetwork_preimputed.csv')
C2pred_df = pd.read_csv(datapath + '221114/C2_nonnetwork_pred.csv')
C2W1nonet_vars = list(C2W1nonet_df.columns)

with open('saved-vars/labelings_non-network.json', 'r') as f:
    nnw_labelings = json.load(f)

cohort, drug = 2, 'marijuana'
X_imp, y, Xenc_df = case_var_delete_v3(cohort, drug, threshold=4)


f = open('saved-vars/trained_vars_v4.txt', 'w')
for v in Xenc_df.columns:
    if len(v.split('.')) > 1:
        key = v.split('_')[0]
        f.write(f"{v}: {nnw_labelings[key][0]}: {str(nnw_labelings[key][1][str(int(float(v.split('_')[1])))])}\n")
    else:
        f.write(f"{v}: {nnw_labelings[v][0]}: {str(nnw_labelings[v][1])}\n")
f.close()

clf_choices = {
    'LG_L1': LogisticRegression(solver='saga', penalty='l1'),
    'LG_L2': LogisticRegression(solver='saga', penalty='l2'),
    'LG_EN': LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5),
    'DT': DecisionTreeClassifier(),
    'SVM': SVC(cache_size=1000),
    'RF': RandomForestClassifier(),
    'GB': GradientBoostingClassifier(),
}
clf_param_grid = {
    'LG_L1': dict(C=np.logspace(-3,3,num=7)),
    'LG_L2': dict(C=np.logspace(-3,3,num=7)),
    'LG_EN': dict(C=np.logspace(-3,3,num=7)),
    'DT': dict(max_depth=range(3,11), min_samples_split=range(2,11)),
    'SVM': dict(gamma=np.logspace(-4,2,num=7), C=np.logspace(-3,3,num=7)),
    'RF': dict(max_depth=[3,5,7,10,None]),
    'GB': dict(max_depth=[3,5,7,10]),
}
cv_inner, cv_outer = 10, LeaveOneOut()


# %% binary classification (with different thresholds when defining labels)

# clf_name = 'RF'
# objective = 'roc_auc'
for clf_name in ['GB','DT','LG_L1','LG_L2','LG_EN']:
    for objective in ['recall', 'roc_auc']:

        clf = clf_choices[clf_name]
        X = X_imp
        acc_baseline = stats.mode(y)[1][0]/len(y)
        hard_preds = []
        soft_preds = []
        for train_idx, test_idx in cv_outer.split(X):
            X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            if clf_name not in ['DT','RF','GB']:    X_train, X_test = norm_scale(X_train, X_test)
            search = GridSearchCV(clf, param_grid=clf_param_grid[clf_name], cv=cv_inner, refit=True, n_jobs=5, scoring=objective)
            search.fit(X_train, y_train)
            hard_pred, soft_pred = search.predict(X_test)[0], search.decision_function(X_test)[0] if clf_name=='SVM' else search.predict_proba(X_test)[:,1][0]
            print(hard_pred, y_test[0], soft_pred)
            hard_preds.append(hard_pred)
            soft_preds.append(soft_pred)
        print(f'{clf_name}: ACC: {accuracy_score(y, np.array(hard_preds)):.2f}/{acc_baseline:.2f}, recall: {recall_score(y, np.array(hard_preds)):.2f}/0.5, AUROC: {roc_auc_score(y, np.array(soft_preds)):.2f}/0.5')




# %% ordinal classification (multiclass)
# clf = DecisionTreeClassifier(max_depth=10)
clf = GradientBoostingClassifier(max_depth=3)
ordinal_clf = OrdinalClassifier(clf)
X = X_imp
results = []
for train_idx, test_idx in LeaveOneOut().split(X):
    X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    ordinal_clf.fit(X_train, y_train)
    print(ordinal_clf.predict(X_test)[0], y_test[0])
    results.append(ordinal_clf.predict(X_test)[0] == y_test[0])
print(np.mean(results))
# %%
