import pandas as pd
import numpy as np
import json

from snakemake.scripts import snakemake

inputfile = snakemake.input[0]
outputfile = snakemake.output[0]

modelname = snakemake.wildcards[0]
