import os
import pandas as pd
import numpy as np
import yaml
import pyarrow.feather as feather
import sys
sys.path.append('/mnt/work/hedvig/grepos/plab_workflow/workflow/scripts/')

hostname = os.uname().nodename
if hostname == 'BlackBeast':
    path = '/home/hedvigs/snake_book/econ'
    site = 'home'
    sys.path.append(path)
    
elif hostname == 'hedvig-hp-elitedesk-800-g5-twr':
    path = '/home/hedvigs/PycharmProjects/homewrs/snake_book/econ'
    site = 'work'
    sys.path.append(path)
    
elif hostname == 'work-computer':
    path = '/mnt/work/workbench/hedvigs/snake_book/econ'
    site = 'server'
    sys.path.append(path)
    
elif hostname == 'SilverFlex':
    path = '/home/hedvigs/gitrepos/plab_workflow/workflow/scripts/'
    site = 'silverFlex'
    sys.path.append(path)

#sys.path.append('/mnt/work/workbench/hedvigs/snake_book/econ')
#from src.data_management.read_files import fold_data


def inv_norm_ga(y_dat_nga, ga_max=307, ga_min=182):
    """
    Normalize the values in the "GA" column of the input DataFrame.

    Parameters:
    -----------
    y_dat : pd.DataFrame
        Input DataFrame containing a column named "GA" to be normalized.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with an additional column named "N_GA" containing the normalized values.

    Notes:
    ------
    The normalization is performed by subtracting the minimum value in the "GA" column from each value,
    and then dividing the result by the maximum value in the "GA" column. The normalized values are then
    subtracted from 1 to obtain the final normalized values.

    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'GA': [10, 20, 30, 40]})
    >>> norm_ga(data)
       GA  N_GA
    0  10   0.0
    1  20   0.5
    2  30   0.75
    3  40   1.0
    """
    ga_norm = 1 - y_dat_nga
    ga_sub = ga_norm * ga_max
    y_dat_ga = ga_sub+ga_min
    
    return y_dat_ga


def rename_snps(snp_list, n="tops"):
    if n != "tops":
        k = int(n.removeprefix(("top")))
        snp_list = snp_list[:k]
    suffixes = ["_f", "_m"]
    snp_f = [snp + suffixes[0] for snp in snp_list]
    snp_m = [snp + suffixes[1] for snp in snp_list]
    renamed_snps = snp_f + snp_m
    return renamed_snps


def read_config(access_name, path='/mnt/work/hedvig/grepos/plab_workflow/config'):
    
    #    with open('/mnt/work/workbench/hedvigs/snake_book/econ/config.yaml', 'r') as yamlfile:
    with open(path + '/config.yaml', 'r') as yamlfile:

        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return data[access_name]

def read_smk_config(access_name, path='/mnt/work/hedvig/grepos/plab_workflow/workflow/rules/common.smk'):
    data = {}
    exec(open(path).read(), {"__builtins__": {}}, data)
    return data[access_name]


def fold_summary(y_data=None, k=5):
    if not isinstance(y_data, pd.DataFrame):
        y_path= 'results/data/y_data.feather' #'/mnt/work/workbench/hedvigs/snake_book/econ/out/data/y_data.feather'
        y_data = feather.read_feather(y_path)

    fold_summary = pd.DataFrame()
    for fold in range(k):
        fold_data = y_data[y_data["Test_batch"]==fold]
        max_ga = max(fold_data["GA"])
        min_ga = min(fold_data["GA"])
        n_ptd = np.sum(fold_data["PTD"])
        n_tot = len(fold_data)
        av_ga = np.mean(fold_data["GA"])
        med_ga = np.median(fold_data["GA"])
        fold_summary.loc[fold, "max_ga"] = max_ga
        fold_summary.loc[fold, "min_ga"] = min_ga
        fold_summary.loc[fold, "n_ptd"] = n_ptd
        fold_summary.loc[fold, "n_tot"] = n_tot
        fold_summary.loc[fold, "max_ga"] = max_ga
        fold_summary.loc[fold, "av_ga"] = av_ga
        fold_summary.loc[fold,"med_ga"] = med_ga

    return fold_summary

if __name__ == '__main__':
    print(hostname)
    from data_management.read_files import fold_data
    y_path= 'results/data/y_data.feather' #'/mnt/work/workbench/hedvigs/snake_book/econ/out/data/y_data.feather'
    y_data = feather.read_feather(y_path)
    tenfold_y = fold_data(y_data, k=10)
    fs = fold_summary(y_data=tenfold_y, k=10)
    print(fs)
