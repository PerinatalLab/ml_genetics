import os

import pandas as pd
import numpy as np
import json

#from snakemake.scripts import snakemake
input_files = snakemake.input


outputfile = snakemake.output[0]

TARGET = snakemake.wildcards.iTarget
SUBSET = snakemake.wildcards.iSubset
#GEN = snakemake.wildcards.iGen
MODEL = snakemake.wildcards.iModel
#FOLD = snakemake.wildcards.iFold

if os.path.isfile(outputfile):
    param_df = pd.read_csv(outputfile, index_col=0)
else:
    param_df = dict.fromkeys(['target', 'subset', 'gen', 'model', 'fold', 'param', 'value'])
    
for param_file in input_files:
    FOLD = param_file.split('/')[-1].split('.')[0].split('_')[-1]
    GEN = param_file.split('/')[-1].split('.')[0].split('_')[-2]
    # Load parameters from JSON
    with open(param_file, "r") as f:
        json_par = json.load(f)
    print(f"Loaded {len(json_par)} parameter sets from {param_file}")
    for key, value in json_par.items():
        param_df = param_df.append({
            'target': TARGET,
            'subset': SUBSET,
            'gen': GEN,
            'model': MODEL,
            'fold': FOLD,
            'param': key,
            'value': value
        }, ignore_index=True)   


param_df.to_csv(outputfile)