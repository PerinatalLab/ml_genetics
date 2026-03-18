import pandas as pd
import numpy as np
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from data_management.parsing_set import ParseKwargs


######################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--out")
    parser.add_argument("-d", "--data")
    parser.add_argument("-p", "--pheno")
    parser.add_argument("-u", "--utils")
    parser.add_argument("-w", "--wild", action=ParseKwargs)
    args = parser.parse_known_args()
    args = args[0] if len(args) > 0 else args

    wildcards = args.wild
    out_file = args.out
    score_file = args.data


    subset = wildcards["iSubset"]
    gen = wildcards["iGen"]
    model = wildcards["iModel"]
    fold = int(wildcards["iFold"])


    if os.path.exists(out_file):
        score_df = pd.read_csv(out_file, sep='\t')
        new_score = pd.read_csv(score_file, sep='\t')
        new_score["Gen"] = gen
        new_score["Model"] = model
        new_score["Fold"] = fold
        new_score["Subset"] = subset
        combined_score = pd.concat([score_df, new_score], ignore_index=True)
        combined_score.to_csv(out_file, sep='\t', float_format='%.5f', index=False)
    else:
        score_df = pd.read_csv(score_file, sep='\t')
        score_df["Gen"] = gen
        score_df["Model"] = model
        score_df["Fold"] = fold
        score_df["Subset"] = subset
        score_df.to_csv(out_file, sep='\t', float_format='%.5f', index=False)