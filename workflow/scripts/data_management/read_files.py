#import warnings
import pandas as pd
import numpy as np
import pyarrow.feather as feather
#from pandas_plink import read_plink
from sklearn.model_selection import StratifiedKFold, train_test_split


def get_x_data(x_file="/mnt/work/workbench/hedvigs/hedvproj/data/out_top_M"):
    """
    Generate a genotype data DataFrame from a PLINK file.

    Parameters
    ----------
    x_file : str
        The path to the PLINK file (with .bed, .bim, and .fam extensions).

    Returns
    -------
    x_data : pandas.DataFrame
        The genotype data DataFrame with markers as columns and "Full_sentrix" as the index.

    Raises
    ------
    FileNotFoundError
        If the PLINK file is not found.

    Notes
    -----
    - The PLINK file is expected to have .bed, .bim, and .fam extensions.
    - The genotype data is computed by dividing the bed matrix by 2 and converting it to a NumPy array of dtype=np.float64.
    - The "snp" column from the bim file is used as markers for the columns of the genotype data DataFrame.
    - The "iid" column from the fam file is used as the "Full_sentrix" index of the genotype data DataFrame.

    """
    (bim, fam, bed) = read_plink(x_file)
    x_gen = np.asarray(bed.compute() / 2, dtype=np.float64).T
    markers = bim["snp"].to_numpy()

    x_data = pd.DataFrame(
        {
            "Full_sentrix": fam["iid"],
            **{marker: x_gen[:, i] for i, marker in enumerate(markers)}
        }
    )
    x_data.set_index("Full_sentrix", inplace=True)

    return x_data


def get_y_data(y_file_m, y_file_mt, y_file_f, y_file_ft):
    """
    Read and process the y data from the specified files.

    Parameters
    ----------
    y_file_m : str
        File path to the input file containing y data for mothers (ga).
    y_file_mt : str
        File path to the input file containing y data for mothers (ptd).
    y_file_f : str
        File path to the input file containing y data for kids (ga).
    y_file_ft : str
        File path to the input file containing y data for kids (ptd).

    Returns
    -------
    pd.DataFrame
        The processed y data.

    Examples
    --------
    >>> y_data = get_y_data(y_file_m, y_file_mt, y_file_f, y_file_ft)
    >>> print(y_data.shape)

    Notes
    -----
    - The function assumes that the `match_pheno`, `match_preg`, and `norm_ga` functions are defined and implemented elsewhere.
    - The function assumes that the input files exist and are in the correct format.
    """
    return norm_ga(match_preg(match_pheno(y_file_mt, y_file_m), match_pheno(y_file_ft, y_file_f)))

def match_pheno(y_file_ptd, y_file_ga):
    """
    Merge phenotype data from two files based on a common "Preg_id" column.

    Parameters:
    -----------
    y_file_ptd : str
        File path of the phenotype data file for ptd_data.
    y_file_ga : str
        File path of the phenotype data file for ga_data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing merged phenotype data from y_file_ptd and y_file_ga.

    Notes:
    ------
    This function reads two phenotype data files in tab-separated format and performs the following steps:
    1. Selects specific columns from the y_file_ptd and y_file_ga DataFrames.
    2. Renames the selected columns of y_file_ptd DataFrame to "Full_sentrix", "PTD", and "Preg_id".
    3. Subtracts 1 from the "PTD" column values in the ptd_data DataFrame.
    4. Sorts the ptd_data DataFrame based on the "Full_sentrix" column.
    5. Renames the selected columns of y_file_ga DataFrame to "Full_sentrix", "GA", and "Preg_id".
    6. Merges the ptd_data and ga_data DataFrames based on the "Preg_id" column using a left join.
    7. If all values in the "Full_sentrix" column of the merged DataFrame are equal to the values in
       the "Full_sentrix_ptd" column, the "Full_sentrix_ptd" column is dropped from the merged DataFrame.

    Example:
    --------
    >>> y_file_ptd = "ptd_data.csv"
    >>> y_file_ga = "ga_data.csv"
    >>> match_pheno(y_file_ptd, y_file_ga)
           Full_sentrix  PTD  Preg_id  GA
    0             1234      0        1   7
    1             5678      1        2   4
    2             9012      2        3   5
    ...
    """

    y_phe_ga = pd.read_csv(y_file_ga, sep="\t")
    y_phe_ptd = pd.read_csv(y_file_ptd, sep="\t")

    ptd_data = y_phe_ptd[["SENTRIX_ID", "PTD", "PREG_ID_1724"]].copy()
    ptd_data.columns = ["Full_sentrix", "PTD", "Preg_id"]
    ptd_data["PTD"] -= 1
    ptd_data.sort_values("Full_sentrix", inplace=True)

    ga_data = y_phe_ga[["SENTRIX_ID", "SVLEN_DG", "PREG_ID_1724"]].copy()
    ga_data.columns = ["Full_sentrix", "GA", "Preg_id"]

    y_data = ptd_data.set_index("Preg_id").merge(
        ga_data.set_index("Preg_id"), on="Preg_id", how="left", suffixes=(None, "_ptd"))

    if np.all(y_data["Full_sentrix"] == y_data["Full_sentrix_ptd"]):
        y_data.drop("Full_sentrix_ptd", axis=1, inplace=True)
    
    return y_data


def match_preg(y_data_m: pd.DataFrame, y_data_f: pd.DataFrame):
  # TODO: check whether y_data is mod 5 before removing a control sample 
    """ Match maternal and fetal data based on pregnancy id, remove unmatched samples, and clean the data.

    Parameters:
    -----------
    y_data_m : pd.DataFrame
        DataFrame containing maternal data.
    y_data_f : pd.DataFrame
        DataFrame containing fetal data.

    Returns:
    --------
    pd.DataFrame
        DataFrame with matched and cleaned data.

    Notes:
    ------
    This function joins the y_data_m and y_data_f DataFrames based on the "Preg_id" column.
    It then performs the following steps:
    1. Removes rows with missing values.
    2. Removes the "GA_f" column if all values in the "GA" column are equal to the corresponding values in the "GA_f" column.
    3. Removes the "PTD_f" column if all values in the "PTD" column are equal to the corresponding values in the "PTD_f" column.
    4. Drops the row with index 52103.
    """

    y_data = y_data_m.join(y_data_f, on="Preg_id", rsuffix="_f")

    # Clean the data
    y_data.dropna(inplace=True)

    if np.all(y_data["GA"] == y_data["GA_f"]):
        y_data.pop("GA_f")

    if np.all(y_data["PTD"] == y_data["PTD_f"]):
        y_data.pop("PTD_f")

    y_data.drop(index=52103, inplace=True)

    return y_data


def norm_ga(y_dat):
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

    ga_min = y_dat["GA"].min()
    ga_max = y_dat["GA"].max()

    ga_sub = y_dat["GA"] - ga_min
    ga_norm = ga_sub / ga_max
    n_ga = 1 - ga_norm

    y_dat["NGA"] = n_ga
    return y_dat


def fold_data(y_data, k=5):
    """
    Divide the data into batches using stratified k-fold cross-validation.

    Parameters:
    -----------
    y_data : pd.DataFrame
        DataFrame containing the data to be divided into batches.
    k : int, optional
        Number of folds (default is 5).

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional "Test_batch" column indicating the batch assignment.

    Notes:
    ------
    This function uses stratified k-fold cross-validation to divide the data into k batches.
    The "PTD" column in y_data is used as the target variable for stratification.
    The "Test_batch" column in y_data is updated to indicate the batch assignment.

    Example:
    --------
    >>> data = pd.DataFrame({"PTD": [1, 2, 3, 4, 5]})
    >>> fold_data(data, k=3)
       PTD  Test_batch
    0      1           0
    1      2           0
    2      3           1
    3      4           2
    4      5           2
    """

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    y_data["Test_batch"] = -1

    for k, (_, tes) in enumerate(skf.split(y_data, y_data["PTD"])):
        test_ind = y_data.index[tes].values
        y_data.loc[test_ind, "Test_batch"] = k

    return y_data


def divide_gen(x_data, y_data, gen="m", col_name="Full_sentrix", col_name_f="Full_sentrix_f"):
    """
    Divide the x data based on the specified genotype file.

    Parameters
    ----------
    x_data : pd.DataFrame
        The input x data.
    y_data : pd.DataFrame
        The input y data.
    gen : str, optional
        The gender to divide the data by, by default "m".
        If "m", the "Full_sentrix" column will be used.
        If not "m", the "Full_sentrix_f" column will be used.
    col_name : str, optional
        The column name for the "Full_sentrix" column, by default "Full_sentrix".
    col_name_f : str, optional
        The column name for the "Full_sentrix_f" column, by default "Full_sentrix_f".

    Returns
    -------
    pd.DataFrame
        The divided x data based on the specified gender.

    Examples
    --------
    >>> x_data_gen = divide_gen(x_data, y_data, gen="f", col_name="Full_sentrix", col_name_f="Full_sentrix_f")
    >>> print(x_data_gen.shape)

    Notes
    -----
    - The function assumes that the input dataframes have the specified column names.
    - The function modifies the input x_data dataframe in-place by dropping columns with NaN values.
    """
    if gen != "m":
        sen = (isec for isec in y_data["Full_sentrix_f"])
    else:
        sen = (isec for isec in y_data["Full_sentrix"])

    x_data_gen = x_data.loc[sen]
    x_data_gen.dropna(axis=1, inplace=True)

    return x_data_gen


def combine_gen(x_data: pd.DataFrame, y_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combines two DataFrames, x_data and y_data, based on specific columns and returns the combined DataFrame.

    Parameters:
    - x_data (pd.DataFrame): The first DataFrame to be merged.
    - y_data (pd.DataFrame): The second DataFrame to be merged.

    Returns:
    - pd.DataFrame: The combined DataFrame with the specified columns merged.

    Example:
    >>> x_data = pd.DataFrame({'Full_sentrix': [1, 2, 3], 'Full_sentrix_f': [4, 5, 6], 'A': [7, 8, 9]})
    >>> y_data = pd.DataFrame({'Full_sentrix': [1, 2, 3], 'B': [10, 11, 12]})
    >>> combine_gen(x_data, y_data)
       Full_sentrix  Full_sentrix_f   B  A
    0             1               4  10  7
    1             2               5  11  8
    2             3               6  12  9
    """
    comb = y_data.merge(
        x_data,
        left_on="Full_sentrix",
        right_on="Full_sentrix",
        how="left",
    ).merge(
        x_data,
        left_on="Full_sentrix_f",
        right_on="Full_sentrix",
        how="left",
        suffixes=("_m", "_f"),
    ).drop(
        columns=["Full_sentrix", "Full_sentrix_f", "Test_batch", "NGA", "PTD", "GA"]
    )

    return comb.set_index(y_data.index)


def get_pgs(y_data, pgs_file='/mnt/work/workbench/hedvigs/snake_book/econ/src/data/pgs_pol.csv'):
    """
    Load PGS data from a CSV file and add it to the input y_data DataFrame.

    Parameters
    ----------
    y_data : pandas.DataFrame
        The input DataFrame containing the target labels.

    Returns
    -------
    y_data : pandas.DataFrame
        The input DataFrame with the "PGS" column added.

    Raises
    ------
    FileNotFoundError
        If the CSV file "pgs_pol.csv" is not found.

    ValueError
        If the CSV file does not contain the expected columns.

    Notes
    -----
    - The CSV file is expected to have a column named "moms_GAraw_PGS" containing the PGS data.
    - The CSV file is expected to have a column named "Full_sentrix" containing the identifier for each row in the DataFrame.

    """

    pgs = pd.read_csv(pgs_file)
    y_data = y_data.merge(pgs, how='left', left_on='Full_sentrix', right_on='IID')
    y_data = y_data.drop(columns=['IID'])
    y_data.columns = y_data.columns.str.replace(r"moms_GAraw_", "")

    return y_data
