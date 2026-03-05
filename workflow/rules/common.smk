import numpy as np
import pandas as pd
import glob

configfile: "config/config.yaml"

#-----------------------Dictionaries----------------------
MODELS = ["bnb", "knn", "lda", "lrc","nn", "qda", "rfc", "svc"] # all models to be tuned
GENOME = ["m", "f", "combine"] # maternal, fetal, combined genotype data
FOLDS = [0, 1, 2, 3, 4]
SUBSETS = ["top5", "top23", "selected", "all"] # feature subsets to be tuned
