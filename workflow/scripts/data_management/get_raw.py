import time
import argparse
import sys
import pandas as pd
import numpy as np
import pyarrow.feather as feather

sys.path.append("/mnt/work/hedvigs/grepos/plab_workflow/workflow")
print(sys.path)
from scripts.data_management.setup_data import get_y_data, fold_data, get_x_data


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mat")
parser.add_argument("-f", "--fet")

parser.add_argument("-d", "--data")
parser.add_argument("-o", "--out", nargs="*")
parser.add_argument("-p", "--pheno")

args = parser.parse_intermixed_args()

out_x = args.out[1]
out_y = args.out[0]

y_in = args.pheno
x_in = args.data
x_file = x_in + "/all-samples"

y_in_m = y_in + "/ga_moms.pheno"
y_in_mt = y_in + "/ptd_moms.pheno"

y_in_f = y_in + "/ga_kids.pheno"
y_in_ft = y_in + "/ptd_kids.pheno"

x_data = get_x_data(x_file)


if args.mat != None:
    out_m = args.mat
    out_f = args.fet
    y_data_full_m = get_y_data(
        y_file_m=y_in_m, y_file_mt=y_in_mt, y_file_f=y_in_m, y_file_ft=y_in_mt
    )

    y_data_full_f = get_y_data(
        y_file_m=y_in_f, y_file_mt=y_in_ft, y_file_f=y_in_f, y_file_ft=y_in_ft
    )

    feather.write_feather(y_data_full_m, out_m)
    feather.write_feather(y_data_full_f, out_f)
    feather.write_feather(x_data, out_x)
else:
    y_data = get_y_data(
        y_file_m=y_in_m, y_file_mt=y_in_mt, y_file_f=y_in_f, y_file_ft=y_in_ft
    )
    print("start")
    start_time = time.time()
    nan_ind = x_data[x_data.isna().any(axis=1)].index
    y_data = y_data[~y_data.Full_sentrix.str.contains("|".join(nan_ind))]
    y_data = y_data[~y_data.Full_sentrix_f.str.contains("|".join(nan_ind))]

    x_data = x_data.dropna()

    print("took", time.time() - start_time)

    y_data = fold_data(y_data, k=5)

    feather.write_feather(y_data, out_y)
    feather.write_feather(x_data, out_x)
