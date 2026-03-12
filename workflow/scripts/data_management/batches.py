# import os
#import pandas as pd
#import numpy as np
import pyarrow.feather as feather

# from src.data_management.resampling import sampling
from resampling import sampling

# import warnings


def get_batch(x_data, y_data, fold=0, gen="m"):
    """
    Get a batch of data for training and testing.

    Parameters
    ----------
    x_data : str or pandas.DataFrame
        The input feature data or the file path to the input feature data.
    y_data : str or pandas.DataFrame
        The target variable data or the file path to the target variable data.
    fold : int, optional
        The fold number for cross-validation. Default is 0.
    gen : str, optional
        The generation of the data. Default is "m".

    Returns
    -------
    x_data : list
        A list containing the training and testing feature data.
    y_data : list
        A list containing the training and testing target variable data.

    Notes
    -----
    This function retrieves a batch of feature data and target variable data for training and testing. It accepts either the actual data as pandas DataFrames or the file paths to the data. If the data is provided as file paths, it reads the data using the feather format.

    The function then performs the following steps:
    1. Calls the `get_ybatch` function to split the target variable data into training and testing sets based on the specified fold.
    2. Resets the index of the training and testing target variable data.
    3. Retrieves the values of the specified generation from the training and testing target variable data.
    4. Uses the values of the specified generation to index the feature data and retrieve the corresponding training and testing feature data.
    5. Sets the index of the training and testing target variable data to the specified generation.

    The function returns the training and testing feature data as well as the training and testing target variable data as lists.

    Examples
    --------
    >>> x_data = "path/to/x_data.feather"
    >>> y_data = "path/to/y_data.feather"
    >>> fold = 0
    >>> gen = "m"
    >>> x_train, x_test, y_train, y_test = get_batch(x_data, y_data, fold, gen)
    >>> print(x_train)
    DataFrame containing the training feature data
    >>> print(x_test)
    DataFrame containing the testing feature data
    >>> print(y_train)
    DataFrame containing the training target variable data
    >>> print(y_test)
    DataFrame containing the testing target variable data
    """
    ind_names = {"m": "Full_sentrix", "f": "Full_sentrix_f", "combine": "Preg_id"}
    ind_name = ind_names[gen]

    if isinstance(y_data, str):
        y_data = feather.read_feather(y_data)
    if isinstance(x_data, str):
        x_data = feather.read_feather(x_data)

    y_m, y_t = get_ybatch(y_data, fold)

    y_train = y_m.reset_index()
    y_test = y_t.reset_index()

    x_train = x_data.loc[y_train[ind_name]]
    x_test = x_data.loc[y_test[ind_name]]
    y_train = y_train.set_index(ind_name)
    y_test = y_test.set_index(ind_name)

    return [x_train, x_test], [y_train, y_test]


def get_ybatch(y_data, fold=0):
    """
    Get the training and test batches from the data based on the specified fold.

    Parameters:
    -----------
    y_data : pd.DataFrame
        DataFrame containing the data.
    fold : int, optional
        Fold number to use as the test batch (default is 0).

    Returns:
    --------
    tuple
        A tuple containing the training and test batches as pd.DataFrame objects.

    Notes:
    ------
    This function selects the training and test batches from y_data based on the specified fold.
    The "Test_batch" column in y_data is used to determine the train/test split.

    Example:
    --------
    >>> data = pd.DataFrame({"PTD": [1, 2, 3, 4, 5], "Test_batch": [0, 0, 1, 2, 2]})
    >>> get_ybatch(data, fold=1)
    (   PTD  Test_batch
    0      1           0
    1      2           0
    3      4           2
    4      5           2,    PTD  Test_batch
    2      3           1)
    """

    test_ind = y_data["Test_batch"] == fold
    train_ind = ~test_ind
    y_test = y_data.loc[test_ind]
    y_train = y_data.loc[train_ind]
    return y_train, y_test


def get_xbatch(x_data, y_data, fold=0, gen="m"):
    """
    Generate train and test batches of x_data based on y_data and fold.

    Parameters
    ----------
    x_data : numpy.ndarray or pandas.DataFrame
        The input dataset.

    y_data : numpy.ndarray or pandas.DataFrame
        The target labels corresponding to the x_data.

    fold : int, optional
        The fold index to use for generating the batches. Default is 0.

    gen : str, optional
        The generation type to use for selecting the index name from ind_names.
        Valid options are 'm' (for "Full_sentrix"), 'f' (for "Full_sentrix_f"), and 'combine' (for "Preg_id").
        Default is 'm'.

    Returns
    -------
    x_train : numpy.ndarray or pandas.DataFrame
        The training batch of x_data.

    x_test : numpy.ndarray or pandas.DataFrame
        The testing batch of x_data.
    """
    [x_train, x_test], _ = get_batch(x_data, y_data, fold=fold, gen=gen)

    return x_train, x_test


def get_sampled_batch(x_file, y_file, fold=0, gen="m", method="under"):
    """
    Get a sampled batch of data for training and testing.

    Parameters
    ----------
    x_file : str
        The file path to the input feature data.
    y_file : str
        The file path to the target variable data.
    fold : int, optional
        The fold number for cross-validation. Default is 0.
    gen : str, optional
        The generation of the data. Default is "m".
    method : str, optional
        The sampling method for balancing the data. Default is "under".

    Returns
    -------
    x_data : list
        A list containing the training and testing feature data.
    y_data : list
        A list containing the training and testing target variable data.

    Notes
    -----
    This function retrieves a batch of feature data and target variable data using the `get_batch` function. It then extracts the "PTD" column from the target variable data. The feature data and target variable data are then sampled using the specified sampling method. The sampled feature data and target variable data are returned as lists.

    Examples
    --------
    >>> x_file = "path/to/x_data.csv"
    >>> y_file = "path/to/y_data.csv"
    >>> fold = 0
    >>> gen = "m"
    >>> method = "under"
    >>> x_data, y_data = get_sampled_batch(x_file, y_file, fold, gen, method)
    >>> print(x_data)
    [x_train, x_test]
    >>> print(y_data)
    [y_train, y_test]
    """
    [x_main, x_test], [y_main, y_test] = get_batch(x_file, y_file, fold=fold, gen=gen)

    y_main = y_main["PTD"]
    y_test = y_test["PTD"]

    x_train, y_train = sampling(x_main, y_main, method=method)

    return [x_train, x_test], [y_train, y_test]
