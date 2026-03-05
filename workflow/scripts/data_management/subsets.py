import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from sklearn.decomposition import PCA
import pyarrow.feather as feather


import sys
sys.path.append('/mnt/work/workbench/hedvigs/snake_book/econ')
from src.data_management.resampling import sampling
from src.data_management.batches import get_batch
from src.data_management.setup_data import read_config, rename_snps


def select_features(x_data: pd.DataFrame, y_data: pd.DataFrame, gen: str = "m", components: int = 100, fold: int = 0, trait: str = "PTD") -> pd.DataFrame:
    """
    Selects the top k best features from the input features in x_data using the f_classif scoring function.

    Parameters
    ----------
    x_data : pd.DataFrame
        The input features DataFrame.
    y_data : pd.DataFrame
        The target variable DataFrame.
    gen : str, optional
        The specific genotype data to be used. Defaults to "m".
    components : int, optional
        The number of top features to select. Defaults to 10.
    fold : int, optional
        The fold of data to be used for training and testing. Defaults to 0.
    trait : str, optional
        The target variable to be used for feature selection. Can be "PTD" (for PTD) or "GA" (for N_GA). Defaults to "PTD".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the selected top features.

    Examples
    --------
    >>> x_data = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6], 'Feature3': [7, 8, 9]})
    >>> y_data = pd.DataFrame({'PTD': [10, 11, 12], 'N_GA': [13, 14, 15]})
    >>> select_features(x_data, y_data, gen="m", components=2, fold=0, trait="PTD")
       Feature1  Feature2
    0         1         4
    1         2         5
    2         3         6
    """
    x_data = mac_filter(x_data)
    x_batch, y_batch = get_batch(x_data, y_data, fold=fold, gen=gen)
    x_train, x_test = x_batch
    # Check Variance for features
    var_thresh = VarianceThreshold()
    var_thresh.set_output(transform='pandas')
    x_train = var_thresh.fit_transform(x_train)
    x_test = var_thresh.transform(x_test)  
    x_data = var_thresh.transform(x_data)
    y_train, y_test = y_batch
    scorer = f_classif
    if trait != 'PTD':
        scorer = f_regression
    
    selector = SelectKBest(scorer, k=components)
    selector.fit(x_train, y_train[trait])
    selector.set_output(transform="pandas")

    x_data_trans = selector.transform(x_data)
    
    return x_data_trans


def reduce_features(x_data: pd.DataFrame, y_data: pd.DataFrame, gen: str, components: int = 100, fold: int = 0, trait: str = "PTD") -> pd.DataFrame:
    """
    Reduces the dimensionality of the input features in x_data using Principal Component Analysis (PCA).

    Parameters:
    - x_data (pd.DataFrame): The input features DataFrame.
    - y_data (pd.DataFrame): The target variable DataFrame.
    - gen (str): The specific genotype data to be used.
    - components (int, optional): The number of components to keep after dimensionality reduction. Defaults to 100.
    - fold (int, optional): The fold of data to be used for training and testing. Defaults to 0.
    - trait (str, optional): The target variable to be used for PCA. Can be "PTD" (for PTD) or "GA" (for N_GA). Defaults to "PTD".

    Returns:
    - pd.DataFrame: The DataFrame with reduced dimensionality after applying PCA.

    Example:
    >>> x_data = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6], 'Feature3': [7, 8, 9]})
    >>> y_data = pd.DataFrame({'PTD': [10, 11, 12], 'N_GA': [13, 14, 15]})
    >>> reduce_features(x_data, y_data, gen='m', components=2, fold=0, trait="PTD")
       PCA_Component_1  PCA_Component_2
    0         5.372281        -0.372281
    1         0.000000         0.000000
    2        -5.372281         0.372281
    """
    x_batch, y_batch = get_batch(x_data, y_data, fold=fold, gen=gen)
    x_train, x_test = x_batch
    y_train, y_test = y_batch
    var_thresh = VarianceThreshold()
    var_thresh.set_output(transform='pandas')
    x_train = var_thresh.fit_transform(x_train)
    x_test = var_thresh.transform(x_test)  
    x_data = var_thresh.transform(x_data)

    pca = PCA(n_components=components)
    pca.fit(x_train, y_train[trait])
    pca.set_output(transform="pandas")

    x_data_red = pca.transform(x_data)

    return x_data_red

def mac_filter(x_gen, mac_min=1):
    """
    :param mac_min: min frequency
    :param x_gen: genotypes
    :param markers: markers
    :return: markers, genotypes, minor allele count
    """
    """ filter for minor allele frequencies"""
    ac1 = np.sum(x_gen, axis=0)
    ac0 = x_gen.shape[0] - ac1
    macs = np.minimum(ac1, ac0)
    markers = x_gen.columns
    markers_used = markers[macs >= mac_min]
    x_data = x_gen[markers_used]

    return x_data  # , markers_used, macs


def load_and_preprocess_data(study_id):
    path = read_config('root_path')
    trait, subset, model_name, gen, fold = study_id.rsplit('_')
    # if trait=='NGA':
    #     trait='N_GA'
    fold=int(fold)
    x_data = feather.read_feather(path + f'out/data/x_{gen}.feather')
    y_file = path + 'out/data/y_data.feather' 
    k = 100
    if subset == "selected":
        x_data = select_features(x_data, y_file, gen, k, fold, trait)
    elif subset == "reduced":
        x_data = reduce_features(x_data, y_file, gen, k, fold, trait)
    elif "top" in subset:
        snp_list = read_config('top_path')
        if gen == 'combine':
            snp_list    = rename_snps(snp_list, subset)
        x_data = x_data[snp_list]

    data = preprocess_data(
        get_batch(x_data, y_file, fold, gen), trait
        )
    return data

def preprocess_data(data, trait):
    [[x_train, x_test], [y_train, y_test]] = data
    y_train = y_train[trait]
    y_test = y_test[trait]
    k=5


    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    if trait == "PTD":
        x_train, y_train = sampling(x_train, y_train, "under")
    else:
        x_train = x_train.drop_duplicates()
        sen = x_train.index
        y_train = y_train.loc[sen]

    return [[x_train, y_train], [x_val, y_val], [x_test, y_test]]

def fold_val(x_train, y_train):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for k, (_, tes) in enumerate(skf.split(y_train, y_train["PTD"])):
        val_ind = y_data.index[tes].values
        y_data.loc[test_ind, "Test_batch"] = k

    return y_data

def load_data(study_id):
    path = read_config('root_path')
    trait, subset, model_name, gen, fold = study_id.rsplit('_')
    fold=int(fold)
    x_data = feather.read_feather(path + f'out/data/x_{gen}.feather')
    y_file = path + 'out/data/y_data.feather' 
    k = 100
    if subset == "selected":
        x_data = select_features(x_data, y_file, gen, k, fold, trait)
    elif subset == "reduced":
        x_data = reduce_features(x_data, y_file, gen, k, fold, trait)
    elif "top" in subset:
        snp_list = read_config('top_path')
        if gen == 'combine':
            snp_list    = rename_snps(snp_list, subset)
        else:
            if subset != "tops":
                i = int(subset.removeprefix(("top")))
                snp_list = snp_list[:i]
        x_data = x_data[snp_list]

    data = get_batch(x_data, y_file, fold, gen)
    return data
