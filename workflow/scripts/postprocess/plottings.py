# %%
import pandas as pd
import pyarrow.feather
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn.objects as so
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

import importlib
import sys
import os
hostname = os.uname().nodename
if hostname == 'work-computer':
    path = '/mnt/work/workbench/hedvigs/snake_book/econ'
    site = 'server'
elif hostname == 'SilverFlex':
    path = '/home/hedvigs/gitrepos/plab_workflow'
    site = 'silverFlex'

sys.path.append(path)

from workflow.scripts.tables import table_functions as tf
from workflow.scripts.figures import figure_functions as ff
from workflow.scripts.data_management import setup_data as sd
# %%
importlib.reload(tf)
### Summary file
sum_file = path + '/results/report/sum_file.csv'

# read files
dfn = pd.read_csv(sum_file, sep='\t')
dfn = tf.format_df(dfn, gen, nsub, sub, mod)

# # drop any columns with only NaN
dfn.dropna(axis=1, inplace=True, how='all')

# %%
importlib.reload(tf)
importlib.reload(ff)

# get all metrics and rename columns
dfn = tf.get_all_metrics(dfn)
renamed_dfn = tf.rename_cols(dfn)
renamed_dfn = tf.rename_models(renamed_dfn)
renamed_dfn['Model'] = renamed_dfn['ModelName']
renamed_dfn.drop(columns=["ModelName"], inplace=True)

# get categorical and numerical columns and rownames for categorical
cat_namesn, val_namesn, rownamesn  = tf.get_rownames(renamed_dfn)
renamed_dfn['AUC'] = renamed_dfn['AUC(test)']

# %%
importlib.reload(tf)

# choose subset/filter
sub = 'top5'
filtered = 'combine'

renamed_df  = renamed_dfn
rownames    = rownamesn
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
    'MCC'
    ]

funcs = {
    'AUC':['mean', 'median', 'CI'], 
    'OR':['mean', 'CI']}
full_agg = tf.agg_table(sub_df, categories, valname=valnames, functions=funcs)#{'AUC':['median', 'CI'],'AUC(cMean)':['median', 'CI'], 'OR':['median', 'CI']})
if dfname == 'cm':
    file_name = file_name + f'_{dfname}'

tf.save_tex(full_agg, file_name, site=site)

