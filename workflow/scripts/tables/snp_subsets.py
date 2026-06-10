import os
import pandas as pd
import numpy as np
import yaml
import pyarrow.feather as feather
import sys
sys.path.append('/mnt/work/hedvig/grepos/plab_workflow/workflow')

hostname = os.uname().nodename
if hostname == 'BlackBeast':
    path = '/home/hedvigs/snake_book/econ'
    site = 'home'
    sys.path.append(path)
    
elif hostname == 'hedvig-hp-elitedesk-800-g5-twr':
    path = '/home/hedvigs/PycharmProjects/plab_workflow/workflow'
    site = 'work'
    sys.path.append(path)
    
elif hostname == 'work-computer':
    path = '/mnt/work/hedvig/grepos/plab_workflow/workflow'
    site = 'server'
    sys.path.append(path)
    
elif hostname == 'SilverFlex':
    path = '/home/hedvigs/gitrepos/plab_workflow/workflow'
    site = 'silverFlex'
    sys.path.append(path)
    
from scripts.data_management.subsets import select_features
from scripts.data_management.setup_data import read_config

def get_selected_snps(x_data, y_data, gen):
    trait = 'PTD'
    components = 100
    rows = {
        fold: select_features(x_data, y_data, gen=gen, components=components, fold=fold, trait=trait).columns.tolist()
        for fold in read_config('folds')
    }

    selected_snps = pd.DataFrame.from_dict(rows)
    return selected_snps


if __name__ == "__main__":
    path = read_config('root_path')
    gen = sys.argv[1] if len(sys.argv) > 1 else 'm'    
    x_data = feather.read_feather(path + f'results/data/x_{gen}.feather')
    y_file = path + 'results/data/y_data.feather' 
    selected_snps = get_selected_snps(x_data, y_file, gen)
    snp_file = path + f'results/report/selected_snps_{gen}.csv'
    selected_snps.to_csv(snp_file, sep='\t', index=False)