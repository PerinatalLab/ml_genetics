import argparse
import pyarrow.feather as feather

import read_files as gt
from parsing_set import ParseKwargs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data")
    parser.add_argument("-o", "--out")
    parser.add_argument("-p", "--pheno")
    parser.add_argument("-l", "--list_top", nargs="*")
    parser.add_argument("-f", "--full")
    parser.add_argument("-w", "--wild", action=ParseKwargs)

    args = parser.parse_known_args()  ##CHANGED
    args = args[0] if len(args) > 0 else args

    wildcards = args.wild
    out_file = args.out

    x_data = feather.read_feather(args.data)
    y_dat = feather.read_feather(args.pheno)
    gen = wildcards["iGen"]

    if gen == "combine":
        x_data = gt.combine_gen(x_data, y_dat)
    else:
        x_data = gt.divide_gen(x_data, y_dat, gen)

    if args.full:
        y_dat = feather.read_feather(args.full)
        x_data = gt.divide_gen(x_data, y_dat, gen)

    feather.write_feather(x_data, out_file)
# print(np.argwhere(np.isnan(x_data)))
# ind_nan = np.argwhere(np.isnan(x_data))
# x_data.axes[0][2048]
