# %% this is lightweight sandbox file that functions like a notebook, DO NOT run from terminal
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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from helper import *


# %%
reg_or_classf = 0  # 1 if reg, classf otherwise


# %%
with open('saved-vars/labelings_non-network.json', 'r') as f:
    nnw_labelings = json.load(f)

def find_range(ls):

    diff = []
    for e in product(ls, ls):
        diff.append(e[0]-e[1])
        
    return sorted(set(diff))


def get_score(pred, gt, zero_pt):
    if (pred>zero_pt and gt <= zero_pt) or (pred <= zero_pt and gt > zero_pt):
        return 0
    else: return 1

def get_baseline(y, zero_pt):
    neg, pos = y[np.where(y <= zero_pt)], y[np.where(y > zero_pt)]
    return np.amax([len(neg), len(pos)])/len(y)


def load_data(cohort, drug):

    drug_key = 'ND1' if drug == 'marijuana' else 'ND7'
    pred_categories = list(map(int, nnw_labelings[drug_key][1].keys()))
    le = LabelEncoder()  # encode labels
    le.fit(find_range(pred_categories))
    # print(find_range(pred_categories), le.transform(find_range(pred_categories)))

    if cohort == 1:
        nonet_vars, nonet_df = C1W1nonet_vars, C1W1nonet_df
        pred_df = C1pred_df
        pred_var = zip(pred_df[drug_key], pred_df['Q68']) if drug == 'marijuana' else zip(pred_df[drug_key], pred_df['Q75'])
    elif cohort == 2:
        nonet_vars, nonet_df = C2W1nonet_vars, C2W1nonet_df
        pred_df = C2pred_df
        pred_var = zip(pred_df[drug_key], pred_df[f'W2_{drug_key}'])
    elif cohort == '1+2':
        nonet_df = pd.concat([C1W1nonet_df, C2W1nonet_df], ignore_index=True)
        nonet_vars = C1W1nonet_vars  # same set of columns for both cohorts
        colname_map = {}
        C2pred_keys = list(C2pred_df.columns)
        for i, c in enumerate(list(C1pred_df.columns)):  # map column names of C2pred_df to C1pred_df (since C1W2 has different varnames)
            colname_map[C2pred_keys[i]] = c
        pred_df = pd.concat([C1pred_df, C2pred_df.rename(columns=colname_map)], ignore_index=True)
        pred_var = zip(pred_df[drug_key], pred_df['Q68']) if drug == 'marijuana' else zip(pred_df[drug_key], pred_df['Q75'])
        
    df = impute_MARs(nonet_vars, nonet_df)
    discarded_vars = ['PID','PID2','AL6B','ID13','ID14_4','ID14_5','ID14_6','ID14_7','ND13','ND15_4','ND15_5','ND15_6','ND15_7',
                'DA5','DA6','DA7','DA7a','DA7b','DA7c','DA7d','DA8','DA8a','DA8b','DA8c','DA8d'] + [v for v in list(df.columns) if 'TEXT' in v]

    dep_var_full = []
    use_changes = []
    for a, b in pred_var:
        if not np.isnan(a) and not np.isnan(b):
            # if a < b: d = 0
            # elif a > b: d = 1
            # else: d = 2
            if reg_or_classf:   d = b - a
            else:     d = 1 if b - a > 1 else 0
            dep_var_full.append(d)
            use_changes.append((a, b))
        else:
            dep_var_full.append(np.nan)
            use_changes.append(np.nan)

    df = pd.concat([df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
    y = np.array(dep_var_full)
    y = y[~np.isnan(y)]
    if reg_or_classf:     y = le.transform(y)
    zero_pt = le.transform([0])[0]

    X_df = df[df['pred'].notna()].drop(discarded_vars+['pred'], axis=1)
    ids = df['PID'][X_df.index].to_numpy()
    use_changes = np.array(use_changes, dtype=object)[X_df.index]
    X_df.reset_index(drop=True, inplace=True)


    return X_df, y, dep_var_full, zero_pt, ids, use_changes


def case_var_delete(cohort, drug, thresh=0.9):  # delete case and/or variable if too many missing values

    X_df, y, dep_var_full, zero_pt, ids, use_changes = load_data(cohort, drug)
    X_dropcol_df = X_df.dropna(axis=1, thresh=len(X_df)*thresh)  # drop column (variable) if missing 10% out of all participants
    X_drop_df = X_df.dropna(axis=0, thresh=len(X_dropcol_df.columns)*thresh)  # drop row (partcipant) if missing 10% out of all features
    if len(X_drop_df) < len(X_df):
        y = y[X_drop_df.index]
        ids = ids[X_drop_df.index]
        use_changes = use_changes[X_drop_df.index]

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


    return X_imp, y, Xenc_df, list(X_drop_df.index), dep_var_full, zero_pt, ids, use_changes


datapath = 'data/original/pre-imputed/'
C1W1nonet_df = pd.read_csv(datapath + 'C1W1_nonnetwork_preimputed.csv')
C1pred_df = pd.read_csv(datapath + 'C1_nonnetwork_pred.csv')
C1W1nonet_vars = list(C1W1nonet_df.columns)
C2W1nonet_df = pd.read_csv(datapath + '221114/C2W1_nonnetwork_preimputed.csv')
C2pred_df = pd.read_csv(datapath + '221114/C2_nonnetwork_pred.csv')
C2W1nonet_vars = list(C2W1nonet_df.columns)

cohort, drug = 2, 'marijuana'
X_imp, y, Xenc_df, _, dep_var_full, zero_pt, ids, use_changes = case_var_delete(cohort, drug)

if reg_or_classf:  print(f'{len(np.where(y<zero_pt)[0])} + {len(np.where(y==zero_pt)[0])} + {len(np.where(y>zero_pt)[0])} = {len(y)}')
else:   print(f'{len(np.where(y==0)[0])} + {len(np.where(y>0)[0])} = {len(y)}')

# with open('saved-vars/labelings_non-network.json', 'r') as f:
#     labelings = json.load(f)
# f = open('saved-vars/trained_vars_v3.txt', 'w')
# for v in Xenc_df.columns:
#     if len(v.split('.')) > 1:
#         key = v.split('_')[0]
#         f.write(f"{v}: {labelings[key][0]}: {str(labelings[key][1][str(int(float(v.split('_')[1])))])}\n")
#     else:
#         f.write(f"{v}: {labelings[v][0]}: {str(labelings[v][1])}\n")
# f.close()

# # these lines remove participants with zero difference in drug use
# lov_idx = np.where(y!=zero_pt)[0]
# X_imp = X_imp[lov_idx, :]
# ids = ids[lov_idx]
# use_changes = use_changes[lov_idx]
# y = y[lov_idx]

# %% 
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
                'ID1','OD1','OD6','OD8','OD10','ID17','ID18','ID19','ID20','CJ3']

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


def case_var_delete_v2(cohort, drug, thresh=0.9):  # delete case and/or variable if too many missing values

    X_df, y, dep_var_full = load_data_v2(cohort, drug)
    X_dropcol_df = X_df.dropna(axis=1, thresh=len(X_df)*thresh)  # drop column (variable) if missing 10% out of all participants
    X_drop_df = X_df.dropna(axis=0, thresh=len(X_dropcol_df.columns)*thresh)  # drop row (partcipant) if missing 10% out of all features
    if len(X_drop_df) < len(X_df):  y = y[X_drop_df.index]

    nominal_vars = [v for v in X_drop_df.columns if v in ['DM8','DM10','DM12','DM13','CJ6','CJ7']]
    num_vars = ['SC1','DM1']
    X_ordinal_df = X_drop_df.drop(nominal_vars+num_vars, axis=1)
    X_num_df = X_drop_df[num_vars]
    X_nominal_df = X_drop_df[nominal_vars]
    # Encode
    categories = [list(map(int, nnw_labelings[v][1].keys())) for v in X_ordinal_df.columns]
    encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=np.nan)
    Xenc_ordinal_df = pd.DataFrame(encoder.fit_transform(X_ordinal_df), columns=X_ordinal_df.columns)

    nominal_cols = []
    for v in nominal_vars:
        nominal_cols.append(pd.get_dummies(X_nominal_df[v], prefix=v))
    Xenc_nominal_df = pd.concat(nominal_cols, axis=1)
    Xenc_df = pd.concat([X_num_df.reset_index(drop=True), Xenc_ordinal_df.reset_index(drop=True), Xenc_nominal_df.reset_index(drop=True)], axis=1)
    # Impute
    for v in num_vars:
        Xenc_df[v].fillna(Xenc_df[v].mean(), inplace=True)  # mean substitute for numerical vars
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_imp = imp.fit_transform(Xenc_df)
    X_imp_df = merge_vars(pd.DataFrame(X_imp, columns=Xenc_df.columns))
    X_imp_df.to_csv('d.csv', index=False)
    print(X_imp_df.shape)

    return X_imp_df.to_numpy(), y, X_imp_df, list(X_drop_df.index), dep_var_full

datapath = 'data/original/pre-imputed/'
C1W1nonet_df = pd.read_csv(datapath + 'C1W1_nonnetwork_preimputed.csv')
C1pred_df = pd.read_csv(datapath + 'C1_nonnetwork_pred.csv')
C1W1nonet_vars = list(C1W1nonet_df.columns)
C2W1nonet_df = pd.read_csv(datapath + '221114/C2W1_nonnetwork_preimputed.csv')
C2pred_df = pd.read_csv(datapath + '221114/C2_nonnetwork_pred.csv')
C2W1nonet_vars = list(C2W1nonet_df.columns)

with open('saved-vars/labelings_non-network.json', 'r') as f:
        nnw_labelings = json.load(f)

cohort, drug = 2, 'marijuana'
X_imp, y, Xenc_df, _, dep_var_full = case_var_delete_v2(cohort, drug)


# %% Draw histogram of drug use diff
fig, ax = plt.subplots()
u_diff = [u[1]-u[0] for u in use_changes]
u_init = [u[0] for u in use_changes]
counts, edges, bars = plt.hist([u_diff, u_init], density=False, rwidth=0.95,
                        histtype='barstacked',label=['diff','init'])
for b in bars:
    ax.bar_label(b)
plt.legend()
plt.show()


# %% corr between indep vars
mask = np.triu(np.ones_like(Xenc_df.corr(), dtype=bool))
plt.figure(figsize=(50, 30))
heatmap = sns.heatmap(Xenc_df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, fmt='.2f', cmap='Blues')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show()

# %% corr between indep vars and dep var
full_df = pd.concat([Xenc_df, pd.DataFrame({'drug use diff': y})], axis=1)
plt.figure(figsize=(8, 30))
heatmap = sns.heatmap(full_df.corr()[['drug use diff']].sort_values(by='drug use diff', ascending=False), vmin=-1, vmax=1, annot=True, cmap='Blues')
heatmap.set_title('Features Correlating with drug use diff', fontdict={'fontsize':18}, pad=16)

# %% PCA to visualize data

pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
Xt = pipe.fit_transform(X_imp)
plot = plt.scatter(Xt[:,0], Xt[:,1], c=y)
plt.legend(handles=plot.legend_elements()[0], labels=['0', '1', '2'])
plt.show()


#%%
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

results = []
X = standard_scale(X_imp)
for train_idx, test_idx in LeaveOneOut().split(X):
    X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    reg = ElasticNetCV(cv=10, tol=1e-3, selection='random').fit(X_train, y_train)
    # reg = RidgeCV(cv=10).fit(X_train, y_train)
    results.append(reg.predict(X_test)[0])

score = sum([get_score(pred, gt, zero_pt) for pred, gt in zip(results, y)])/len(y)

# %%
reg_name = 'RF'
scores = []
calibs = np.linspace(-5, 5, 20)
baselines = []
for i, calib in enumerate(calibs):
    calib_zero_pt = zero_pt + calib
    baseline = get_baseline(y, calib_zero_pt)
    baselines.append(baseline)
    score = sum([get_score(pred, gt, calib_zero_pt) for pred, gt in zip(results, y)])/len(y)
    scores.append(score)
    if i % 5 == 0 or i == len(calibs)-1:
        ax_min, ax_max = np.amin(y)-1, np.amax(y)+1
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.scatter(y, results, label=f'cutoff = {round(calib, 2)} wrt zero diff; ACC = {round(score,2)}/{round(baseline,2)}')
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
        plt.xlim([ax_min, ax_max])
        plt.ylim([ax_min, ax_max])
        ax.fill_between([ax_min, calib_zero_pt],ax_min,calib_zero_pt,alpha=0.3, color='#1F98D0')  # blue low
        ax.fill_between([calib_zero_pt, ax_max], ax_min, calib_zero_pt, alpha=0.3, color='#DA383D')  # red low
        ax.fill_between([ax_min, calib_zero_pt], calib_zero_pt, ax_max, alpha=0.3, color='#DA383D')  # red high
        ax.fill_between([calib_zero_pt, ax_max], calib_zero_pt, ax_max, alpha=0.3, color='#1F98D0')  # blue high
        plt.ylabel("Model prediction")
        plt.xlabel("Label (ground truth)")
        plt.legend()
        plt.savefig(f'plots/analysis/threshold/{reg_name}/threshold_{round(calib,2)}_{cohort}_{drug}_{reg_name}.pdf', facecolor='white')

plt.clf()
plt.plot(calibs, scores, label='LOO prediction')
plt.plot(calibs, baselines, label='baseline')
plt.ylabel("ACC")
plt.xlabel("Cutoff point (wrt zero difference)")
plt.legend()
plt.savefig(f'plots/analysis/threshold/{reg_name}/threshold_{cohort}_{drug}_{reg_name}.pdf', facecolor='white')

# plt.plot(y, label='gt')
# plt.plot(results, label='pred (Lasso)')
# plt.hlines([zero_pt], *plt.xlim(), color='r', label='no change')
# plt.legend()
# plt.show()

# %%
f = open('saved-vars/misclf_ppl.txt', 'w')
incorrect = 0
gt, pred = [], []
train_vars = [v.split('_')[0] for v in Xenc_df.columns]
orig_df = C1W1nonet_df if cohort == 1 else C2W1nonet_df
for a,b,c,d in zip(results, y, ids, use_changes):
    # if a > zero_pt:
    #     pred.append(1)
    # else: pred.append(0)
    # if b > zero_pt:
    #     gt.append(1)
    # else: gt.append(0)
    if (a>zero_pt and b <= zero_pt) or (a <= zero_pt and b > zero_pt):
        incorrect+=1
        print(a, b, c, d)
        ind_df = orig_df.loc[orig_df['PID'] == c]
        f.write(c+'\n')
        ind_info = {nnw_labelings[k][0]: nnw_labelings[k][1][str(int(ind_df[k]))] for k in ind_df.columns if k in train_vars and not pd.isna(ind_df.iloc[0][k]) and k not in ['SC1','DM1']}
        f.write(f'Age: {int(ind_df["DM1"])}\n')
        f.write(f'Years live in current community: {int(ind_df["SC1"])}\n')
        for k in ind_df.columns:
            if k in train_vars and not pd.isna(ind_df.iloc[0][k]) and k not in ['SC1','DM1']:
                f.write(f'{nnw_labelings[k][0]}: {nnw_labelings[k][1][str(int(ind_df[k]))]}\n')
        f.write('\n\n')
f.close()

# %%
zero_pt = zero_pt
ax_min, ax_max = np.amin(y)-1, np.amax(y)+1
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(y, results, label=f'ACC = {round(score,2)}')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
plt.xlim([ax_min, ax_max])
plt.ylim([ax_min, ax_max])
ax.fill_between([ax_min, zero_pt],ax_min,zero_pt,alpha=0.3, color='#1F98D0')  # blue low
ax.fill_between([zero_pt, ax_max], ax_min, zero_pt, alpha=0.3, color='#DA383D')  # red low
ax.fill_between([ax_min, zero_pt], zero_pt, ax_max, alpha=0.3, color='#DA383D')  # red high
ax.fill_between([zero_pt, ax_max], zero_pt, ax_max, alpha=0.3, color='#1F98D0')  # blue high
plt.ylabel("Model predictions")
plt.xlabel("Truths")
plt.legend()
plt.show()


# %% clustering
from sklearn.cluster import Birch

brc = Birch(n_clusters=3)
brc.fit(X_imp)
labels = brc.predict(X_imp)


# %% RANSAC
from sklearn.linear_model import RANSACRegressor, ElasticNet

X = X_imp
results = []
est = ElasticNetCV(cv=10, tol=1e-3, selection='random')
# est = ElasticNet(tol=1e-3, selection='random')
for train_idx, test_idx in LeaveOneOut().split(X):
    X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    reg = RANSACRegressor(random_state=0, min_samples=50).fit(X_train, y_train)
    results.append(reg.predict(X_test)[0])

score = sum([get_score(pred, gt, zero_pt) for pred, gt in zip(results, y)])/len(y)


# %% random forest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

X = X_imp
hparams={'n_estimators': [100],
        "max_depth" : np.arange(2, 12),
        }

results = []
reg = RandomForestRegressor()
for train_idx, test_idx in LeaveOneOut().split(X):
    X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    search = GridSearchCV(reg, param_grid=hparams, cv=10, n_jobs=10)
    search.fit(X_train, y_train)
    results.append(search.predict(X_test)[0])

score = sum([get_score(pred, gt, zero_pt) for pred, gt in zip(results, y)])/len(y)


# %% decision tree regressor
from sklearn.tree import DecisionTreeRegressor

X = X_imp

hparams={"splitter":["best","random"],
        "max_depth" : np.arange(2, 12),
        "min_samples_leaf":[1,3,5,7,9],
        # "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        "max_features":["auto","log2","sqrt",None],
        # "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90]
        }
results = []
reg = DecisionTreeRegressor()
for train_idx, test_idx in LeaveOneOut().split(X):
    X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    search = GridSearchCV(reg, param_grid=hparams, cv=10, n_jobs=10)
    search.fit(X_train, y_train)
    results.append(search.predict(X_test)[0])

score = sum([get_score(pred, gt, zero_pt) for pred, gt in zip(results, y)])/len(y)


# %% Ridge coefficients (weights) as a function of the regularization (alpha)
from sklearn.linear_model import Ridge
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
# X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y = np.ones(10)

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=True)
    ridge.fit(X[:,:10], y)
    coefs.append(ridge.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show()


# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
roc_auc_score(y, clf.predict_proba(X)[:, 1])

# roc_auc_score(y, clf.decision_function(X))

# %%
labelings_NS = {
    'NSX2': 'How long person X has been someone that you could confide in',
    'NSX3': 'How often you hang out or spend time with person X',
    'NSX5': 'Age of person X',
    'NSX7': 'Level of education person X has finished to the best of your knowledge',
    'NSX8': 'Current employment status of person X to the best of your knowledge',
    'NSX9': 'Whether person X is a person you are comfortable using drugs around',
    'NSX10': 'How often you have used drugs with person X in the past 30 days',
    'NSX11': 'How often (you think) person X uses drugs in general',
    'NSX12': 'Relationship with person X'
}

cate_mappings_NS = {
    'NSX2A': {1: 'Less than 6 months', 4: '6 months to a year', 5: '1-2 years', 6: '3-5 years', 7: 'More than 5 years'},
    'NSX3A': {1: 'Less than once a months', 7: 'Once a month', 8: 'Once a week', 9: 'Almost daily'},
    'NSX5A': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7A': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8A': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9A': {1: 'No', 2: 'Yes'},
    'NSX10A': {1: 'Never', 4: 'Less than once a month', 5: 'Once a month', 6: 'Once a week', 7: '2-6 times a week', 8: 'One time per day', 9: '2-3 times per day', 10: '4 or more times per day'},
    'NSX11A': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12A': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NSX2B': {1: 'Less than 6 months', 48: '6 months to a year', 49: '1-2 years', 50: '3-5 years', 51: 'More than 5 years'},
    'NSX3B': {1: 'Less than once a months', 7: 'Once a month', 8: 'Once a week', 9: 'Almost daily'},
    'NSX5B': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7B': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8B': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9B': {1: 'No', 2: 'Yes'},
    'NSX10B': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX11B': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12B': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NSX2C': {1: 'Less than 6 months', 4: '6 months to a year', 5: '1-2 years', 6: '3-5 years', 7: 'More than 5 years'},
    'NSX3C': {1: 'Less than once a months', 7: 'Once a month', 8: 'Once a week', 9: 'Almost daily'},
    'NSX5C': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7C': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8C': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9C': {1: 'No', 2: 'Yes'},
    'NSX10C': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX11C': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12C': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NSX2D': {1: 'Less than 6 months', 4: '6 months to a year', 5: '1-2 years', 6: '3-5 years', 7: 'More than 5 years'},
    'NSX3D': {1: 'Less than once a months', 7: 'Once a month', 8: 'Once a week', 9: 'Almost daily'},
    'NSX5D': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7D': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8D': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9D': {1: 'No', 2: 'Yes'},
    'NSX10D': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX11D': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12D': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NSX2E': {1: 'Less than 6 months', 4: '6 months to a year', 5: '1-2 years', 6: '3-5 years', 7: 'More than 5 years'},
    'NSX3E': {1: 'Less than once a months', 7: 'Once a month', 8: 'Once a week', 9: 'Almost daily'},
    'NSX5E': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7E': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8E': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9E': {1: 'No', 2: 'Yes'},
    'NSX10E': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX11E': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12E': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NSX2F': {1: 'Less than 6 months', 8: '6 months to a year', 9: '1-2 years', 10: '3-5 years', 11: 'More than 5 years'},
    'NSX3F': {1: 'Less than once a months', 4: 'Once a month', 5: 'Once a week', 6: 'Almost daily'},
    'NSX5F': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7F': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8F': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9F': {1: 'No', 2: 'Yes'},
    'NSX10F': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX11F': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12F': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NSX2G': {1: 'Less than 6 months', 4: '6 months to a year', 5: '1-2 years', 6: '3-5 years', 7: 'More than 5 years'},
    'NSX3G': {1: 'Less than once a months', 8: 'Once a month', 9: 'Once a week', 10: 'Almost daily'},
    'NSX5G': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7G': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8G': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9G': {1: 'No', 2: 'Yes'},
    'NSX10G': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX11G': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12G': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NSX2H': {1: 'Less than 6 months', 4: '6 months to a year', 5: '1-2 years', 6: '3-5 years', 7: 'More than 5 years'},
    'NSX3H': {1: 'Less than once a months', 7: 'Once a month', 8: 'Once a week', 9: 'Almost daily'},
    'NSX5H': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7H': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8H': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9H': {1: 'No', 4: 'Yes'},
    'NSX10H': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX11H': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12H': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NSX2I': {1: 'Less than 6 months', 4: '6 months to a year', 5: '1-2 years', 6: '3-5 years', 7: 'More than 5 years'},
    'NSX3I': {1: 'Less than once a months', 7: 'Once a month', 8: 'Once a week', 9: 'Almost daily'},
    'NSX5I': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NSX7I': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NSX8I': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NSX9I': {1: 'No', 2: 'Yes'},
    'NSX10I': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX11I': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NSX12I': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'}
}

labelings_ND = {
    'NDX2': 'Age of person X',
    'NDX4': 'Level of education person X has finished to the best of your knowledge',
    'NDX5': 'Current employment status of person X to the best of your knowledge',
    'NDX6': 'Relationship with person X',
    'NDX8': 'How often you used drugs with person X in the past 30 days',
    'NDX9': 'How often (you think) person X uses drugs in general',
    'NDX12': 'How long you have used drugs with person X'
}

cate_mappings_ND = {
    'NDX2J': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4J': {1: 'Less than high school', 4: 'High school degree', 5: 'Some college', 6: '2 year degree', 7: '4 year degree', 8: 'Graduate or professional school', 9: "Don't know"},
    'NDX5J': {1: 'Employed full-time', 4: 'Employed part-time', 5: 'Not employed', 6: 'Others', 7: "Don't know"},
    'NDX6J': {1: 'They are my spouse/romantic partner', 4: 'They are my parent', 5: 'They are my child', 6: 'They are another relative', 7: 'They are my friend', 8: 'They are my something else'},
    'NDX8J': {11: 'Never', 12: 'Less than once a month', 13: 'Once a month', 14: 'Once a week', 15: '2-6 times a week', 16: 'One time per day', 17: '2-3 times per day', 18: '4 or more times per day'},
    'NDX9J': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX12J': {8: 'Less than 6 months', 9: '6 months to a year', 10: '1-2 years', 11: '3-5 years', 12: 'More than 5 years'},
    'NDX2K': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4K': {10: 'Less than high school', 11: 'High school degree', 12: 'Some college', 13: '2 year degree', 14: '4 year degree', 15: 'Graduate or professional school', 16: "Don't know"},
    'NDX5K': {1: 'Employed full-time', 8: 'Employed part-time', 9: 'Not employed', 10: 'Others', 11: "Don't know"},
    'NDX6K': {1: 'They are my spouse/romantic partner', 9: 'They are my parent', 10: 'They are my child', 11: 'They are another relative', 12: 'They are my friend', 14: 'They are my something else'},
    'NDX8K': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 17: '2-3 times per day', 19: '4 or more times per day'},
    'NDX9K': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 17: '2-3 times per day', 19: '4 or more times per day'},
    'NDX12K': {1: 'Less than 6 months', 8: '6 months to a year', 9: '1-2 years', 10: '3-5 years', 11: 'More than 5 years'},
    'NDX2L': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4L': {1: 'Less than high school', 10: 'High school degree', 11: 'Some college', 12: '2 year degree', 13: '4 year degree', 14: 'Graduate or professional school', 15: "Don't know"},
    'NDX5L': {1: 'Employed full-time', 8: 'Employed part-time', 9: 'Not employed', 10: 'Others', 11: "Don't know"},
    'NDX6L': {1: 'They are my spouse/romantic partner', 9: 'They are my parent', 10: 'They are my child', 11: 'They are another relative', 12: 'They are my friend', 13: 'They are my something else'},
    'NDX8L': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 17: '2-3 times per day', 19: '4 or more times per day'},
    'NDX9L': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 17: '2-3 times per day', 19: '4 or more times per day'},
    'NDX12L': {1: 'Less than 6 months', 8: '6 months to a year', 9: '1-2 years', 10: '3-5 years', 11: 'More than 5 years'},
    'NDX2M': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4M': {1: 'Less than high school', 10: 'High school degree', 11: 'Some college', 12: '2 year degree', 13: '4 year degree', 14: 'Graduate or professional school', 15: "Don't know"},
    'NDX5M': {1: 'Employed full-time', 8: 'Employed part-time', 9: 'Not employed', 10: 'Others'},
    'NDX6M': {1: 'They are my spouse/romantic partner', 9: 'They are my parent', 10: 'They are my child', 11: 'They are another relative', 12: 'They are my friend', 13: 'They are my something else'},
    'NDX8M': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 17: '2-3 times per day', 19: '4 or more times per day'},
    'NDX9M': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 17: '2-3 times per day', 19: '4 or more times per day'},
    'NDX12M': {1: 'Less than 6 months', 8: '6 months to a year', 9: '1-2 years', 12: '3-5 years', 15: 'More than 5 years'},
    'NDX2N': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4N': {1: 'Less than high school', 10: 'High school degree', 11: 'Some college', 12: '2 year degree', 13: '4 year degree', 14: 'Graduate or professional school', 15: "Don't know"},
    'NDX5N': {1: 'Employed full-time', 8: 'Employed part-time', 9: 'Not employed', 10: 'Others', 11: "Don't know"},
    'NDX6N': {1: 'They are my spouse/romantic partner', 9: 'They are my parent', 10: 'They are my child', 11: 'They are another relative', 12: 'They are my friend', 13: 'They are my something else'},
    'NDX8N': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX9N': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX12N': {1: 'Less than 6 months', 8: '6 months to a year', 9: '1-2 years', 10: '3-5 years', 11: 'More than 5 years'},
    'NDX2O': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4O': {1: 'Less than high school', 10: 'High school degree', 11: 'Some college', 12: '2 year degree', 13: '4 year degree', 14: 'Graduate or professional school', 15: "Don't know"},
    'NDX5O': {1: 'Employed full-time', 9: 'Employed part-time', 10: 'Not employed', 11: 'Others', 12: "Don't know"},
    'NDX6O': {9: 'They are my spouse/romantic partner', 10: 'They are my parent', 11: 'They are my child', 12: 'They are another relative', 13: 'They are my friend', 14: 'They are my something else'},
    'NDX8O': {11: 'Never', 12: 'Less than once a month', 13: 'Once a month', 14: 'Once a week', 15: '2-6 times a week', 16: 'One time per day', 17: '2-3 times per day', 18: '4 or more times per day'},
    'NDX9O': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX12O': {1: 'Less than 6 months', 8: '6 months to a year', 9: '1-2 years', 10: '3-5 years', 11: 'More than 5 years'},
    'NDX2P': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4P': {1: 'Less than high school', 10: 'High school degree', 11: 'Some college', 12: '2 year degree', 13: '4 year degree', 14: 'Graduate or professional school', 15: "Don't know"},
    'NDX5P': {1: 'Employed full-time', 8: 'Employed part-time', 9: 'Not employed', 10: 'Others', 11: "Don't know"},
    'NDX6P': {1: 'They are my spouse/romantic partner', 9: 'They are my parent', 10: 'They are my child', 11: 'They are another relative', 12: 'They are my friend', 13: 'They are my something else'},
    'NDX8P': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX9P': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX12P': {1: 'Less than 6 months', 8: '6 months to a year', 9: '1-2 years', 10: '3-5 years', 11: 'More than 5 years'},
    'NDX2Q': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4Q': {1: 'Less than high school', 10: 'High school degree', 11: 'Some college', 12: '2 year degree', 13: '4 year degree', 14: 'Graduate or professional school', 15: "Don't know"},
    'NDX5Q': {1: 'Employed full-time', 8: 'Employed part-time', 9: 'Not employed', 10: 'Others', 11: "Don't know"},
    'NDX6Q': {1: 'They are my spouse/romantic partner', 9: 'They are my parent', 10: 'They are my child', 11: 'They are another relative', 12: 'They are my friend', 13: 'They are my something else'},
    'NDX8Q': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX9Q': {11: 'Never', 12: 'Less than once a month', 13: 'Once a month', 14: 'Once a week', 15: '2-6 times a week', 16: 'One time per day', 17: '2-3 times per day', 18: '4 or more times per day'},
    'NDX12Q': {1: 'Less than 6 months', 8: '6 months to a year', 9: '1-2 years', 10: '3-5 years', 11: 'More than 5 years'},
    'NDX2R': {0: 'Children (14 and below)', 1: 'Youth (15 to 24)', 2: 'Adult (25 to 64)', 3: 'Senior (65 and above)'},
    'NDX4R': {1: 'Less than high school', 10: 'High school degree', 11: 'Some college', 12: '2 year degree', 13: '4 year degree', 14: 'Graduate or professional school', 15: "Don't know"},
    'NDX5R': {1: 'Employed full-time', 8: 'Employed part-time', 9: 'Not employed', 10: 'Others', 11: "Don't know"},
    'NDX6R': {1: 'They are my spouse/romantic partner', 9: 'They are my parent', 10: 'They are my child', 11: 'They are another relative', 12: 'They are my friend', 13: 'They are my something else'},
    'NDX8R': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX9R': {1: 'Never', 11: 'Less than once a month', 12: 'Once a month', 13: 'Once a week', 14: '2-6 times a week', 15: 'One time per day', 16: '2-3 times per day', 17: '4 or more times per day'},
    'NDX12R': {1: 'Less than 6 months', 15: '6 months to a year', 16: '1-2 years', 17: '3-5 years', 18: 'More than 5 years'}
}


# %% process C2 network data
datapath = 'data/original/pre-imputed/221114/'
C2W1net_df = pd.read_csv(datapath + 'C2W1_network_preimputed.csv')

C2W1_NS_df = C2W1net_df.filter(regex='(?=NS)(?=^(?!NSD))')  # network-confide df
C2W1_ND_df = pd.concat([C2W1net_df['NSD1'], C2W1net_df.filter(regex='ND')], axis=1)  # network-drug df

NS_features_list, ND_features_list = extract_net_info(C2W1_NS_df, cate_mappings_NS, C2W1_ND_df, cate_mappings_ND)

NS_df = pd.DataFrame(
    [[i_info[v][cate] for v in i_info.keys() if not v.startswith('n_') for cate in i_info[v].keys()] + [i_info['n_nodes_NS'], i_info['n_edges_NS']] for i_info in NS_features_list],
    columns=[f'{v}_{i+1}' for v in NS_features_list[0].keys() if not v.startswith('n_') for i in range(len(NS_features_list[0][v].keys()))]+['n_nodes_NS','n_edges_NS']
)
ND_df = pd.DataFrame(
    [[i_info[v][cate] for v in i_info.keys() if not v.startswith('n_') for cate in i_info[v].keys()] + [i_info['n_nodes_ND'], i_info['n_edges_ND']] for i_info in ND_features_list],
    columns=[f'{v}_{i+1}' for v in ND_features_list[0].keys() if not v.startswith('n_') for i in range(len(ND_features_list[0][v].keys()))]+['n_nodes_ND','n_edges_ND']
)
df = pd.concat([NS_df, ND_df], axis=1)
df.to_csv(f'saved-vars/C2_network_221114-processed.csv', index=False)

# %% process C1+2 network data
datapath = 'data/original/pre-imputed/'
C1W1net_df = pd.read_csv(datapath + 'C1W1_network_preimputed.csv')
C2W1net_df = pd.read_csv(datapath + '221114/C2W1_network_preimputed.csv')

C1W1_NS_df = C1W1net_df.filter(regex='(?=NS)(?=^(?!NSD))')  # network-confide df
C1W1_ND_df = pd.concat([C1W1net_df['NSD1'], C1W1net_df.filter(regex='ND')], axis=1)  # network-drug df
C2W1_NS_df = C2W1net_df.filter(regex='(?=NS)(?=^(?!NSD))')  # network-confide df
C2W1_ND_df = pd.concat([C2W1net_df['NSD1'], C2W1net_df.filter(regex='ND')], axis=1)  # network-drug df

df = pd.DataFrame()

for cohort in range(1,3):

    if cohort == 1:
        NS_features_list, ND_features_list = extract_net_info(C1W1_NS_df, cate_mappings_NS, C1W1_ND_df, cate_mappings_ND)
    else:
        NS_features_list, ND_features_list = extract_net_info(C2W1_NS_df, cate_mappings_NS, C2W1_ND_df, cate_mappings_ND)

    NS_df = pd.DataFrame(
        [[i_info[v][cate] for v in i_info.keys() if not v.startswith('n_') for cate in i_info[v].keys()] + [i_info['n_nodes_NS'], i_info['n_edges_NS']] for i_info in NS_features_list],
        columns=[f'{v}_{i+1}' for v in NS_features_list[0].keys() if not v.startswith('n_') for i in range(len(NS_features_list[0][v].keys()))]+['n_nodes_NS','n_edges_NS']
    )
    ND_df = pd.DataFrame(
        [[i_info[v][cate] for v in i_info.keys() if not v.startswith('n_') for cate in i_info[v].keys()] + [i_info['n_nodes_ND'], i_info['n_edges_ND']] for i_info in ND_features_list],
        columns=[f'{v}_{i+1}' for v in ND_features_list[0].keys() if not v.startswith('n_') for i in range(len(ND_features_list[0][v].keys()))]+['n_nodes_ND','n_edges_ND']
    )

    if df.empty:
        df = pd.concat([NS_df, ND_df], axis=1)
    else:
        df = pd.concat([df, pd.concat([NS_df, ND_df], axis=1)], ignore_index=True)
    df.to_csv(f'saved-vars/C12_network_221114-processed.csv', index=False)
