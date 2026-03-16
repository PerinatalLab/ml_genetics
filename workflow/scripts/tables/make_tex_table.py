import pandas as pd
import pyarrow.feather
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn.objects as so
from scipy import stats
import argparse

import warnings
warnings.filterwarnings("ignore")

import importlib
import sys
import os
hostname = os.uname().nodename
if hostname == 'BlackBeast':
    path = '/home/hedvigs/snake_book/econ'
    site = 'home'
elif hostname == 'hedvig-hp-elitedesk-800-g5-twr':
    path = '/home/hedvigs/PycharmProjects/homewrs/snake_book/econ'
    site = 'work'
elif hostname == 'work-computer':
    path = '/mnt/work/workbench/hedvigs/snake_book/econ'
    site = 'server'
elif hostname == 'wl-241113-007':
    path = '/home/hedvigs/wslGit/snake_book/econ'
    site = 'silverFlex'

sys.path.append(path)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import table_functions as tf
from figures import figure_functions as ff
from data_management import setup_data as sd
from data_management.parsing_set import ParseKwargs




def aggregate_df(df, sub, filtered, combined=True):

    df = tf.get_all_metrics(df)
    df = tf.divide_concur(df)

    #rename columns
    renamed_df = tf.rename_cols(df)

    # get categorical and numerical columns and rownames for categorical
    cat_names, val_names, rownames  = tf.get_rownames(renamed_df)
    cm_cols = [col for col in val_names if '(cM' in col]
    auc_cols = [col for col in val_names if 'AUC' in col]
    nonc_cols = [col for col in val_names if col not in cm_cols]
    nonp_cols = [col for col in val_names if '(' not in col]
    
    if combined:
        renamed_df = renamed_df.drop(columns=nonc_cols)
        renamed_df.columns = renamed_df.columns.str.replace(r'\(cMean\)$', '', regex=True)
        # update col and row names
        cat_names, val_names, rownames  = tf.get_rownames(renamed_df)
    else:
        renamed_df['AUC'] = renamed_df['AUC(test)']

    renamed_df.sort_values(by="Model", inplace=True)
    for key, values in rownames.items():
        if filtered in values:
            filterkey = key
            renamed_df = renamed_df[renamed_df[filterkey]==filtered]


    cols_few_unique = [col for col in renamed_df.columns if renamed_df[col].nunique() < 2]
    filtered = np.unique(renamed_df[cols_few_unique].values)
    filtered = filtered[0] if len(filtered)>0 else ''


    filterkey = filtered
    for key, values in rownames.items():
        if filtered in values:
            filterkey = key


    for key, values in rownames.items():
        if sub in values:
            sub_df = renamed_df[renamed_df[key]==sub]
            categories = [k_name for k_name in rownames.keys() if k_name!=key and k_name!=filterkey]
            categories_str = ''.join(categories)
            file_name = f"{sub}by{categories_str}_{filtered}"
        elif sub == key:
            sub_df = renamed_df
            categories = [sub]
            file_name = f"{sub}_{filtered}"
        elif sub == '':
            sub_df = renamed_df
            categories = [k_name for k_name in rownames.keys() if k_name!=filterkey]
            categories_str = ''.join(categories)
            file_name = f"{sub}by{categories_str}_{filtered}"

    valnames = [
        'AUC',
        'OR', 
        'Sens', 
        'Spec', 
        'BalAcc',
        'MCC'
        ]

    categories.append('Fold')
    funcs = {
        'AUC':['mean', 'median', 'CI'], 
        'OR':['mean', 'CI']}
    full_agg = tf.agg_table(sub_df, categories, valname=valnames, functions=funcs)
    
    return full_agg

#########################################################
if __name__=='__main__':


    parser=argparse.ArgumentParser()

    parser.add_argument("-o", "--out")
    parser.add_argument('-d', '--data')
    parser.add_argument("-p", "--pheno")
    parser.add_argument("-u", "--utils")
    parser.add_argument('-w', '--wild', action=ParseKwargs)
    args=parser.parse_known_args()
    args = args[0] if len(args)>0 else args

    wildcards   = args.wild
    out_file    = args.out
    in_file     = args.data


    SUBSET      = wildcards['iSubset']
    GEN         = wildcards['iGen']
    MODEL_NAME  = wildcards['iModel']


    # read files
    dfn = pd.read_csv(in_file)

    dfn = tf.format_df(dfn)
    # drop any columns with only NaN
    dfn.dropna(axis=1, inplace=True, how='all')

    # calculate all metrics 
    #dfn = tf.get_all_metrics(dfn)
    file_name = str.split(out_file, sep='/')[-1]
    sub, categories, filtered = str.split(file_name,sep='_')
    full_agg = aggregate_df(dfn, sub=sub, filtered=filtered)
    
    tf.save_tex(full_agg, out_file, site=site)
    




