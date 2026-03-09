import pandas as pd
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
sys.path.append('/mnt/work/workbench/hedvigs/snake_book/econ')
from src.data_management.setup_data import read_config
from src.data_management.parsing_set import ParseKwargs




def comb_run(pred_dir, subset, gen, fold):
    models = read_config('all_models')
    id_name = {'m': 'Full_sentrix', 'combine': 'Preg_id', 'f':'Full_sentrix'}
    c_pred = pd.DataFrame()
    for model in models:
        pred_file = pred_dir + f'pred_PTD_{subset}_{model}_{gen}_{fold}.csv'
        try:
            pred_df = pd.read_csv(pred_file, sep='\t').set_index(id_name[gen])
            y_target = pred_df.Target
            pred_df = pred_df.drop(columns=['Target'])
            pred_df['AvPred'] = np.mean(pred_df, axis=1)
            if len(c_pred.columns)<1:
                c_pred['Target'] = y_target
                c_pred.set_index(pred_df.index, inplace=True)
                c_pred[model] = pred_df.AvPred
            elif y_target.equals(c_pred.Target):
                c_pred[model] = pred_df.AvPred
            else:
                print('Target not matched')
                continue
        except FileNotFoundError:
            print(f'Missing: {gen},{model},Fold{fold}')
            continue
    return c_pred
                        
 
#########################################################
if __name__=='__main__':
   # """

    parser=argparse.ArgumentParser()

    parser.add_argument("-o", "--out")
    parser.add_argument('-d', '--data')
    parser.add_argument("-p", "--pheno")
    parser.add_argument("-u", "--utils")
    parser.add_argument('-w', '--wild', action=ParseKwargs)
    args=parser.parse_intermixed_args()


    wildcards   = args.wild
    out_file    = args.out
    score_file  = args.utils


    SUBSET      = wildcards['iSubset']
    GEN         = wildcards['iGen']
   # MODEL_NAME  = wildcards['iModel']
    FOLD        = int(wildcards['iFold'])
    
    
    save_file = out_file

    path = read_config('root_path')
    pred_dir = path + 'out/analysis/predictions/'
    
#    save_file = pred_dir + f'avpred_{subset}_{gen}_{fold}.csv'

    c_pred = comb_run(pred_dir, SUBSET, GEN, FOLD)
    c_pred.to_csv(save_file, sep='\t', float_format='%.5f')






