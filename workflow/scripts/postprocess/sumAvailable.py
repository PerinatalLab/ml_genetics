# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyarrow.feather as feather
from scipy.stats import kstest

import sys
import os
hostname = os.uname().nodename
if hostname == 'work-computer':
    path = '/mnt/work/workbench/hedvigs/snake_book/econ'
    site = 'server'
elif hostname == 'SilverFlex':
    path = '/home/hedvigs/gitrepos/plab_workflow/'
    site = 'silverFlex'

sys.path.append(path)
from workflow.scripts.data_management import setup_data as gt

# %%
def summarize_scores(
    subset=None, model=None, gen=None, fold=None, path=None, verbose=0
    ):
    inputs = [subset, gen, fold, model]
    config_names = ["subsets", "genomes", "folds", "models"]
    config_dict = {}
    for item, config_name in zip(inputs, config_names):
        if item is not None:
            config_dict[config_name] = [item]
        else:
            config_dict[config_name] = gt.read_config(config_name, path=path)
    y_pred = []
    missing = pd.DataFrame()
    k = 0

    for subset in config_dict["subsets"]:
        for model in config_dict["models"]:
            for gen in config_dict["genomes"]:
                for fold in config_dict["folds"]:
                    score_file = (
                        path
                        + f"results/scores/PTD/{subset}/{model}_{gen}_{fold}.csv"
                        )
                    score_file_adj = (
                        path
                        + f"results/scores/PTD/{subset}/{model}_{gen}_{fold}_adj.csv"
                        )
                    try:
                        try:
                            scores = pd.read_csv(score_file, sep="\t")
                        except FileNotFoundError:
                            scores = pd.read_csv(score_file_adj, sep='\t')
                            
                        scores["fold"] = fold
                        scores["gen"] = gen
                        scores["model"] = model
                        scores["subset"] = subset
                        y_pred.append(scores)
                    except FileNotFoundError:
                        
                        k += 1
                        missing.loc[k, "subset"] = subset
                        missing.loc[k, "model"] = model
                        missing.loc[k, "gen"] = gen
                        missing.loc[k, "fold"] = fold
    if k == 0:
        print(f" All files found!, {k} missing")
    elif verbose == 1:
        print(f"Files not found:", k)
    elif verbose == 2:
        print(f"Files not found:", k)
        subsetnames, subsetcounts = np.unique(missing["subset"], return_counts=True)
        for sname, scount in zip(subsetnames, subsetcounts):
            print(f"  {sname}: {scount} missing")
            missubset = missing[missing["subset"] == sname]
            modelnames, modelcounts = np.unique(missubset["model"], return_counts=True)
            for mname, mcount in zip(modelnames, modelcounts):
                print(f"    {mname}: {mcount} missing")
    elif verbose == 3:
        print("Total files not found:", k)
        for colname in missing.columns:
            names, counts = np.unique(missing[colname], return_counts=True)
            print(f"{colname}: ")
            for name, count in zip(names, counts):
                print(f"  {name}: {count} missing")
    elif verbose == 4:
        print(f"Files not found:", k)
        subsetnames, subsetcounts = np.unique(missing["subset"], return_counts=True)
        for sname, scount in zip(subsetnames, subsetcounts):
            print(f"  {sname}: {scount} missing")
            missubset = missing[missing["subset"] == sname]
            modelnames, modelcounts = np.unique(missubset["model"], return_counts=True)
            for mname, mcount in zip(modelnames, modelcounts):
                print(f"    {mname}: {mcount} missing")
                missmodel = missubset[missubset["model"] == mname]
                gennames, gencounts = np.unique(missmodel["gen"], return_counts=True)
                for gname, gcount in zip(gennames, gencounts):
                    print(f"        {gname}: {gcount} missing")

    full_set = pd.concat(y_pred)
    return full_set

# %%
subset = None 
model = None
gen = None 

sum_df = summarize_scores(subset=subset, model=model, gen=gen, fold=None,path=path, verbose=4)
auc_df = sum_df.drop(columns=["number"])
sum_file= path + 'results/report/sum_file.csv'
auc_df.to_csv(sum_file, sep="\t", index=False)


