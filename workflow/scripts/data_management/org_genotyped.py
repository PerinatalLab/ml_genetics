import argparse
import sys
import pyarrow.feather as feather

sys.path.append("/mnt/work/hedvigs/grepos/plab_workflow/workflow/scripts/")

from data_management import setup_data as gt
from data_management.parsing_set import ParseKwargs

"""   
    x_in = "/mnt/work/workbench/hedvigs/hedvproj/hed_ML/results/geno/plink"
    subset      = 'top5'
    gen         = 'combined'
    x_file      = x_in + '/all-samples'
    tops        = ["rs2963463", "rs12037376", "rs5991030", "rs4359773", "rs34555419", 
                    "rs13161560", "rs6472846", "rs7650602", "rs7023208", "rs28654158", "rs3129768", 
                    "rs6831441", "rs6090040", "rs1650530", "rs62270785", "rs5930554", "rs9823520",
                    "rs2659685", "rs113828443", "rs72898946", "rs2387280", "rs73815678","rs875581"]
    target      = 'PTD'  
    out_file = f'/mnt/work/workbench/hedvigs/snake_book/econ/out/data/{subset}/x_data.feather'

    y_dat = feather.read_feather('/mnt/work/workbench/hedvigs/snake_book/econ/out/data/y_data.feather')
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data")
    parser.add_argument("-o", "--out")
    parser.add_argument("-p", "--pheno")
    parser.add_argument("-l", "--list_top", nargs="*")
    parser.add_argument("-f", "--full")
    parser.add_argument("-w", "--wild", action=ParseKwargs)

    args = parser.parse_intermixed_args()

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
