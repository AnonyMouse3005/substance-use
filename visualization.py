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

from helper import *


# %%
def load_and_impute(cohort, encode=True, impute=True):

    if cohort == 1:
        nonet_vars, nonet_df = C1W1nonet_vars, C1W1nonet_df
    elif cohort == 2:
        nonet_vars, nonet_df = C2W1nonet_vars, C2W1nonet_df
    elif cohort == '1+2':
        nonet_df = pd.concat([C1W1nonet_df, C2W1nonet_df], ignore_index=True)
        nonet_vars = C1W1nonet_vars  # same set of columns for both cohorts

    df = impute_MARs(nonet_vars, nonet_df)
    discarded_vars = ['PID','PID2','AL6B','ID13','ID14_4','ID14_5','ID14_6','ID14_7','ND13','ND15_4','ND15_5','ND15_6','ND15_7',
                'DA5','DA6','DA7','DA7a','DA7b','DA7c','DA7d','DA8','DA8a','DA8b','DA8c','DA8d'] + [v for v in list(df.columns) if 'TEXT' in v]
    nominal_vars = ['DM8','DM10','DM12','DM13']

    X_df = df.drop(discarded_vars, axis=1)
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

    if impute:
        # Impute
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_imp = imp.fit_transform(Xenc_df)
        return X_imp
    else:
        return Xenc_df if encode else X_df


def plot_missingness(cohort):

    X_df = load_and_impute(cohort, encode=False, impute=False)

    fig = plt.subplots(figsize=(15,50), tight_layout=True)
    ax = sns.heatmap(X_df.isna().transpose(), cbar=False)
    ax.set(xlabel='Participant', ylabel='Non-network Variable')
    if cohort == "1+2":     ax.vlines([35], *ax.get_ylim(), color='r')
    plt.xticks(rotation=90)
    
    plt.savefig(f'plots/analysis/visualization/missingness/{cohort}_missingness.pdf', facecolor='white')


# %% main
if __name__ == '__main__':

    datapath = 'data/original/pre-imputed/'
    C1W1nonet_df = pd.read_csv(datapath + 'C1W1_nonnetwork_preimputed.csv')
    C1W1nonet_vars = list(C1W1nonet_df.columns)
    C2W1nonet_df = pd.read_csv(datapath + 'C2W1_nonnetwork_preimputed.csv')
    C2W1nonet_vars = list(C2W1nonet_df.columns)

    # %% plot missingness of (non-network) data
    cohorts = [1, 2, "1+2"]
    for cohort in cohorts:
        plot_missingness(cohort)


    # %% plot distribution of C1 vs C2 (non-network data)
    with open('saved-vars/labelings_non-network.json', 'r') as f:  # load dict containing lists of cohorts, drugs, and methods to be investigated
        mappings = json.load(f)

    X_c1_df = load_and_impute(cohort=1, encode=False, impute=False)
    X_c2_df = load_and_impute(cohort=2, encode=False, impute=False)
    # mappings = {"SC3": ["How satisfied with living in current community", {"1": "Very dissatisfied", "6": "Somewhat dissatisfied", "7": "Neutral", "8": "Somewhat satisfied", "9": "Very satisfied"}], "SC4": ["Think whether will still be living in current community 2 years into the future", {"1": "No", "2": "Yes"}]}
    for v_name, v_info in mappings.items():

        v_label, v_cate = v_info[0], v_info[1]

        X_c1 = X_c1_df.loc[:,v_name]
        X_c2 = X_c2_df.loc[:,v_name]

        if v_name not in ['SC1', 'DM1']:

            X_c1 = [v_cate[str(int(entry))] for entry in sorted(X_c1) if not np.isnan(entry)]
            X_c2 = [v_cate[str(int(entry))] for entry in sorted(X_c2) if not np.isnan(entry)]

            c1_df = pd.DataFrame({'C1': pd.Categorical(X_c1, categories=list(v_cate.values()))}).value_counts(sort=False)
            c2_df = pd.DataFrame({'C2': pd.Categorical(X_c2, categories=list(v_cate.values()))}).value_counts(sort=False)
            df = pd.concat([c1_df, c2_df], keys=['C1', 'C2'], axis=1)
            ax = df.plot.bar()
            plt.title(v_label, fontdict={'fontsize' : 10})
            plt.tight_layout()
            plt.savefig(f'plots/analysis/visualization/distribution/{v_name}_distribution.pdf', bbox_inches='tight', facecolor='white')

        
# %%
