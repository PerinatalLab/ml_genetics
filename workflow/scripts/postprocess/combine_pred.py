import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, confusion_matrix
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

#sys.path.append('/mnt/work/workbench/hedvigs/snake_book/econ')
from data_management.setup_data import read_config
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
    score_file = args.utils


    subset = wildcards["iSubset"]
    gen = wildcards["iGen"]
    model = wildcards["iModel"]
    fold = int(wildcards["iFold"])

path = read_config('root_path')
models = read_config('all_models')
gens = read_config('genotypes')

# read files

pred_dir = path + 'out/analysis/predictions/'
score_dir = path + f'out/analysis/scores/PTD/classic/{subset}/'
save_file = path + f'out/tables/comb_score_{subset}.csv'

#save_file = f'/mnt/work/hedvig/datafiles/c_score_{subset}.csv'

id_name = {'m': 'Full_sentrix', 'combine': 'Preg_id', 'f':'Full_sentrix'}
scores=[]

for gen in gens:
    print(f'{gen}')
    for model in models:
        for fold in range(5):
            pred_file = pred_dir + f'pred_PTD_{subset}_{model}_{gen}_{fold}.csv'
            score_file = score_dir + f'{model}_{gen}_{fold}_pruned.csv'
            try:
                pred_df = pd.read_csv(pred_file, sep='\t').set_index(id_name[gen])
                old_score = pd.read_csv(score_file, sep='\t').set_index('number')
                #mean_old = old_score.median(axis=0)
                mean_old = old_score.mean(axis=0)
                y_target = pred_df['Target'].values
                pred_df = pred_df.drop(columns=['Target'])
                avpred = np.mean(pred_df, axis=1)
                medpred = np.median(pred_df, axis=1)
                pred_df['AvPred'] = avpred #np.mean(pred_df, axis=1)
                pred_df['MedPred'] = medpred # np.median(pred_df, axis=1)
                avy_pred = pred_df.AvPred.values
                my_pred = pred_df.MedPred.values
                auc_score = roc_auc_score(y_target, avy_pred)
                mauc_score = roc_auc_score(y_target, my_pred)
                tn, fp, fn, tp = confusion_matrix(y_target, avy_pred.round()).ravel()
                score_dict = {
                    "Gen": gen,
                    "Model": model,
                    "Fold": fold,
                    "AUC(meanPred)": auc_score,
                    "AUC(medianPred)": mauc_score,
                    "AUC(scoreMean)": mean_old.auc_prob,
                    "tn": tn, 
#                    "otn": int(mean_old.tn.round()),
                    "fp": fp, 
#                    "ofp": int(mean_old.fp.round()),
                    "fn": fn,
#                    "ofn": int(mean_old.fn.round()), 
                    "tp": tp,
#                    "otp": int(mean_old.tp.round())
                    }
                                
                scores.append(score_dict)
                
            except FileNotFoundError:
                print(f'Missing: {gen},{model},Fold{fold}')
                continue



score_df = pd.DataFrame.from_records(scores)
score_df.to_csv(save_file, sep='\t', float_format='%.5f', index=False)
