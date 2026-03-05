from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np


def sampling(x_data: np.ndarray, y_data: pd.DataFrame, method: str = "under", rs: int = 42, rp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Sampling function to perform oversampling or undersampling on the data.

    Parameters
    ----------
    x_data : np.ndarray
        The input features.
    y_data : pd.DataFrame
        The target variable.
    method : str, optional
        The sampling method to use ("under" for undersampling, "over" for oversampling, None for no sampling), by default "under".
    rs : int, optional
        The random seed, by default 42.
    rp : bool, optional
        Whether to use replacement in undersampling, by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The sampled features and target variable.
    """
    if isinstance(y_data, pd.DataFrame):
        y_data = y_data["Trait"]

    if method == "under":
        rus = RandomUnderSampler(replacement=rp, random_state=rs)
        x_res, y_res = rus.fit_resample(x_data, y_data)

    elif method == "over":
        ros = RandomOverSampler(random_state=rs)
        x_res, y_res = ros.fit_resample(x_data, y_data)

    else:
        x_res = x_data
        y_res = y_data

    return x_res, y_res
