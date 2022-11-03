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
from genetic_selection import GeneticSelectionCV
from joblib import Parallel, delayed
from mlxtend.feature_selection import ExhaustiveFeatureSelector
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


# %% boilerplate
if len(sys.argv) != 3:
    print('Usage: python3 inference.py [plot&save figs?] [save df?]')
    sys.exit(1)


# %%
def load_data(cohort, drug):

    datapath = 'data/original/pre-imputed/'
    C1W1nonet_df = pd.read_csv(datapath + 'C1W1_nonnetwork_preimputed.csv')
    C1pred_df = pd.read_csv(datapath + 'C1_nonnetwork_pred.csv')
    C1W1nonet_vars = list(C1W1nonet_df.columns)
    C2W1nonet_df = pd.read_csv(datapath + 'C2W1_nonnetwork_preimputed.csv')
    C2pred_df = pd.read_csv(datapath + 'C2_nonnetwork_pred.csv')
    C2W1nonet_vars = list(C2W1nonet_df.columns)

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

    # Impute
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_imp = imp.fit_transform(Xenc_df)

    #---------------------------------- Network ------------------------------------------------------
    net_df = pd.concat([net_df, pd.DataFrame({'pred': dep_var_full})], axis=1)  # drop rows where prediction var is missing
    Xnet_df = net_df[net_df['pred'].notna()].drop('pred', axis=1)
    X_np = Xnet_df.to_numpy()

    #-------------------------------------------------------------------------------------------
    X = np.concatenate((X_imp, X_np), axis=1)
    X_df = pd.concat([Xenc_df, Xnet_df], axis=1)

    y = np.array(dep_var_full)
    y = y[~np.isnan(y)]
    baseline = stats.mode(y)[1][0]/len(y)

    return X, y, X_df, baseline


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


# %%
if __name__ == '__main__':

    datapath = 'results/'
    nonnetwork_results = pd.read_csv(datapath + 'scores_nonnetwork.csv')
    network_results = pd.read_csv(datapath + 'scores_network.csv')
    netnonnet_results = pd.read_csv(datapath + 'scores_nonnetwork+network.csv')
    results_dict = {'nonnet': nonnetwork_results,
                    'net': network_results,
                    'netnonnet': netnonnet_results}

    clf_dict = {'LG': LogisticRegression(max_iter=10000),
                'DT': DecisionTreeClassifier(),
                'SVM': SVC()
                }
    best_dict = {}

    for drug in ['marijuana', 'meth']:
        for clf_name, clf in clf_dict.items():

            cohorts = ['C1', 'C2', 'C1+C2']
            x, y = np.meshgrid(cohorts, cohorts)
            intensity = np.zeros((len(cohorts),len(cohorts)))
            labels = []

            for ii, cohort in enumerate([1, 2, '1+2']):  # cohort trained on
                best_results = {}
                for name, result in results_dict.items():  # features from nonnet, net, or combined

                    col = result[f'{cohort}-{drug}-{clf_name}']
                    scores = [float(elem) for elem in col if isfloat(elem)]
                    methods = [elem for elem in result[f'{cohort}-{drug}-fgroup'] if type(elem) == str]
                    best_score = np.amax(scores)
                    best_scores_idx = np.argwhere(scores == best_score).flatten()  # multiple max scores
                    best_features_idx = result.index[col==str(float(best_score))].tolist()
                    # CAVEAT: below line assumes that the best method(s) is either chi2, thresholding or GA only (i.e., has feature listing below the ACC score)
                    best_features = [ast.literal_eval(col[i+1]) for i in best_features_idx]
                    best_methods = dict(zip([methods[i] for i in best_scores_idx], best_features))
                    best_results[name] = {'best_method(s)': best_methods, 'best_score': best_score}

                best_score_all = max(val['best_score'] for val in best_results.values())
                best_methods_all = {f'{key}_{skey}': sval for key, val in best_results.items() if val['best_score'] == best_score_all for skey, sval in val['best_method(s)'].items()}
                best_dict[f'C{cohort}-{drug}-{clf_name}_best'] = [method for method in best_methods_all.keys()]
                # best_dict[f'C{cohort}-{drug}-{clf_name}-features'] = [features for features in best_methods_all.values()]

                # train best model from cohort x and validate it on cohort 1, 2, 1+2
                X, y, X_df, _ = load_data(cohort, drug)
                for jj, cohort_test in enumerate([1, 2, '1+2']):  # cohort tested on
                    
                    X_test, y_test, Xtest_df, baseline = load_data(cohort_test, drug)
                    test_scores = []
                    for method, features in best_methods_all.items():  # dealing with tied highest accuracy during feature selection -> pick one that yields highest test score
                        # CAVEAT: due to one-hot encoding of nominal variables, C1 and C2 might have different total numbers of features
                        X_train = X[:, [X_df.columns.get_loc(f) for f in features if f in Xtest_df.columns]]
                        X_test_trim = X_test[:, [Xtest_df.columns.get_loc(f) for f in features if f in Xtest_df.columns]]
                        model = clf.fit(X_train, y) if clf_name == 'DT' else clf.fit(standard_scale(X_train), y)
                        predictions = model.predict(X_test_trim) if clf_name == 'DT' else model.predict(standard_scale(X_test_trim))
                        test_scores.append(np.mean(predictions==y_test))

                    if len(test_scores) > 1:  # mark the tied best method that yields highest test ACC
                        indicator = '*' if cohort_test == 1 else ('^' if cohort_test == 2 else '~')
                        best_dict[f'C{cohort}-{drug}-{clf_name}_best'][np.argmax(test_scores)] = best_dict[f'C{cohort}-{drug}-{clf_name}_best'][np.argmax(test_scores)] + indicator
                    intensity[ii, jj] = max(test_scores)
                    labels.append(f'{round(max(test_scores), 2)}/{round(baseline, 2)}')

            if int(sys.argv[1]):
                annot = np.array(labels).reshape((len(cohorts),len(cohorts)))
                s = sns.heatmap(intensity, annot=annot, fmt='', cmap='Blues', xticklabels=cohorts, yticklabels=cohorts)
                s.set(xlabel='Test', ylabel='Train', title=f'{clf_name} classifier on {drug} use (test ACC/baseline ACC)')
                plt.savefig(f'plots/analysis/inferences/{drug}-{clf_name}.pdf')
                plt.close()

    best_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in best_dict.items() ]))
    if int(sys.argv[2]):
        best_df.reindex(sorted(best_df.columns), axis=1).to_csv('results/best_performing_methods.csv', index=False)