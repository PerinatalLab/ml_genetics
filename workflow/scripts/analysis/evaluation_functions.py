import numpy as np
import pandas as pd

#from sklearn.model_selection import permutation_test_score, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import (
    roc_auc_score,
    r2_score,
    explained_variance_score,
    mean_absolute_error,
    fbeta_score,
    f1_score,
    mean_absolute_percentage_error,
    class_likelihood_ratios,
    make_scorer,
)

from sklearn.base import clone

import numpy as np
from sklearn.utils import resample

def bootstrap_test_score(model, X, y, n_permutations=1000, scoring=roc_auc_score, random_state=None):
    rng = np.random.RandomState(random_state)
    [x_train, x_test] = X
    [y_train, y_test] = y

    scorer = make_scorer(scoring, needs_proba=True)
    n_bootstraps = n_permutations
    # Original score without shuffling
    model.fit(x_train, y_train)
    original_score = scoring(y_test, model.predict_proba(x_test)[:,1])
    
    # Initialize list to store bootstrap scores
    bootstrap_scores = []
    
    # Perform bootstrap resampling
    for i in range(n_bootstraps):
        model_clone=clone(model)
        # Resample X and y with replacement
        X_resampled, y_resampled = resample(x_train, y_train, stratify=y_train, random_state=rng)
        
        # Fit model on resampled data and calculate score
        model_clone.fit(X_resampled, y_resampled)
        score = scoring(y_test, model_clone.predict_proba(x_test)[:,1])
        
        bootstrap_scores.append(score)
    
    # Convert bootstrap scores to a NumPy array for easy manipulation
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate p-value as proportion of bootstrap scores >= original score
    p_value = np.sum(bootstrap_scores >= original_score) / n_bootstraps
    
    return original_score, bootstrap_scores, p_value



def custom_permutation_test_score(model, X, y, scoring=roc_auc_score, cv=5, n_permutations=1000, random_state=None):
    """
    Custom permutation test to compare the original model score with permuted labels.

    Parameters:
    - model: The machine learning model to be evaluated.
    - X: Feature matrix.
    - y: Target vector.
    - scoring: Scoring function to evaluate the model (default is roc_auc_score).
    - cv: Number of cross-validation folds (default is 5).
    - n_permutations: Number of permutations for the test (default is 1000).
    - random_state: Seed for reproducibility (default is None).

    Returns:
    - original_score: The score of the model on the original data.
    - permuted_scores: List of scores from each permutation.
    - p_value: P-value calculated as the proportion of permuted scores better than or equal to the original score.
    """
    # Set the random state for reproducibility
    rng = np.random.RandomState(random_state)
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Compute the original score
    original_score = []
    for train_idx, val_idx in cv.split(X, y):
        model_clone = clone(model)
        model_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_val_pred = model_clone.predict_proba(X.iloc[val_idx])[:, 1]  # Assuming binary classification
        original_score.append(scoring(y.iloc[val_idx], y_val_pred))
    original_score = np.mean(original_score)
    
    # Compute the scores for permuted labels
    permuted_scores = []
    for _ in range(n_permutations):
        # Shuffle the labels
        y_permuted = rng.permutation(y)
        
        permuted_score = []
        for train_idx, val_idx in cv.split(X, y_permuted):
            model_clone = clone(model)
            model_clone.fit(X.iloc[train_idx], y_permuted[train_idx])
            y_val_pred = model_clone.predict_proba(X.iloc[val_idx])[:, 1]
            permuted_score.append(scoring(y_permuted[val_idx], y_val_pred))
        permuted_scores.append(np.mean(permuted_score))
    
    # Calculate the p-value
    p_value = (np.sum(permuted_scores >= original_score) + 1.0) / (n_permutations + 1)
    #p_value = np.mean(np.array(permuted_scores) >= original_score)
    
    return original_score, permuted_scores, p_value
    
    
def my_permutation_test_score(model, X, y, scoring=roc_auc_score, n_permutations=1000, random_state=None):
    """
    Custom permutation test to compare the original model score with permuted labels.

    Parameters:
    - model: The machine learning model to be evaluated.
    - X: Feature matrix.
    - y: Target vector.
    - scoring: Scoring function to evaluate the model (default is roc_auc_score).
    - cv: Number of cross-validation folds (default is 5).
    - n_permutations: Number of permutations for the test (default is 1000).
    - random_state: Seed for reproducibility (default is None).

    Returns:
    - original_score: The score of the model on the original data.
    - permuted_scores: List of scores from each permutation.
    - p_value: P-value calculated as the proportion of permuted scores better than or equal to the original score.
    """
    # Set the random state for reproducibility
    rng = np.random.RandomState(random_state)
    [x_train, x_test] = X
    [y_train, y_test] = y
   # Initialize cross-validation
#    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Compute the original score
    model_clone = clone(model)
    model_clone.fit(x_train, y_train)
    y_val_pred = model_clone.predict_proba(x_test)[:, 1]  # Assuming binary classification
    original_score = (scoring(y_test, y_val_pred))
    
    # Compute the scores for permuted labels
    permuted_scores = []
    for _ in range(n_permutations):
        # Shuffle the labels
        y_permuted = rng.permutation(y_train)
        
        
        permuted_score = []
        model_clone = clone(model)
        model_clone.fit(x_train, y_permuted)
        y_val_pred = model_clone.predict_proba(x_test)[:, 1]
        permuted_scores.append(scoring(y_test, y_val_pred))
    
    # Calculate the p-value
    p_value = (np.sum(permuted_scores >= original_score) + 1.0) / (n_permutations + 1)
    #p_value = np.mean(np.array(permuted_scores) >= original_score)
    
    return original_score, permuted_scores, p_value
    
    
    
def ps_permutation_test_score(model, X, y, scorer=roc_auc_score, n_permutations=1000, random_state=None):
    """
    Custom permutation test to compare the original model score with permuted labels.

    Parameters:
    - model: The machine learning model to be evaluated.
    - X: Feature matrix.
    - y: Target vector.
    - scoring: Scoring function to evaluate the model (default is roc_auc_score).
    - cv: Number of cross-validation folds (default is 5).
    - n_permutations: Number of permutations for the test (default is 1000).
    - random_state: Seed for reproducibility (default is None).

    Returns:
    - original_score: The score of the model on the original data.
    - permuted_scores: List of scores from each permutation.
    - p_value: P-value calculated as the proportion of permuted scores better than or equal to the original score.
    """
    # Set the random state for reproducibility
    # rng = np.random.RandomState(random_state)
    [x_train, x_test] = X
    [y_train, y_test] = y

    #scorer = make_scorer(scoring, needs_proba=True)

        # Combine training and test data
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Create a test_fold array where -1 indicates training data and 0 indicates test data
    test_fold = np.zeros(len(X), dtype=int)
    test_fold[:len(x_train)] = -1  # Training data
    test_fold[len(x_train):] = 0   # Test data

    # Create the PredefinedSplit object
    ps = PredefinedSplit(test_fold)
    
    score, permutation_scores, pvalue = permutation_test_score(
    model, X, y, cv=ps, n_permutations=n_permutations, scoring=scorer, random_state=random_state
    )
    
    return score, permutation_scores, pvalue
 
