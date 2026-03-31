import os

import pandas as pd
import numpy as np
import json

from snakemake.scripts import snakemake

params_file = snakemake.input[0]
outputfile = snakemake.output[0]

TARGET = snakemake.wildcards.iTarget
SUBSET = snakemake.wildcards.iSubset
GEN = snakemake.wildcards.iGen
MODEL = snakemake.wildcards.iModel
FOLD = snakemake.wildcards.iFold

if os.path.isfile(outputfile):
    param_df = pd.read_csv(outputfile, index_col=0)
else:
    param_df = pd.DataFrame(columns=['target', 'subset', 'gen', 'model', 'fold', 'param', 'value'])
    
    
try:
    # Load parameters from JSON
    with open(params_file, "r") as f:
        json_par = json.load(f)
    print(f"Loaded {len(json_par)} parameter sets from {params_file}")
    for key, value in json_par.items():
        param_df.loc[key] = value

except FileNotFoundError:
    print(f"File {params_file} not found.")