import numpy as np
import pandas as pd
import glob


configfile: "config/config.yaml"


# -----------------------Dictionaries----------------------
MODELS = [
    "bnb",
    "knn",
    "lda",
    "lrc",
    "nn",
    "qda",
    "rfc",
    "svc",
]  # all models to be tuned
GENOME = ["m", "f", "combine"]  # maternal, fetal, combined genotype data
FOLDS = [0, 1, 2, 3, 4]
SUBSETS = ["top5", "top23", "selected", "all"]  # feature subsets to be tuned
NMODELS = [1, 2, 3, 4, 5, 6, 7, 8]  # number of models to be included in summary tables


wildcard_constraints:
    iModel="|".join(MODELS),
    iGen="|".join(GENOME),
    iFold="|".join(map(str, FOLDS)),
    iSubset="|".join(SUBSETS),
    nModels="|".join(map(str, NMODELS)),


# -------------------------Variables---------------------- #
logAll = " 2>&1 | tee -a {log}"
