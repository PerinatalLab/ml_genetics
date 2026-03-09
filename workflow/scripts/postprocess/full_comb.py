import pandas as pd
import sys
import numpy as np
import json
import argparse

from sklearn.metrics import roc_auc_score, confusion_matrix
sys.path.append('/mnt/work/workbench/hedvigs/snake_book/econ')
from src.data_management.setup_data import read_config
from itertools import combinations


def calculate_scores(pred_df, y_target, models, num_to_drop, subset, gen, fold, in_scores):
    used_combinations = set()  # To keep track of unique combinations
    # Determine the number of models to drop
#    num_to_drop = int(model.removeprefix(('n')))
#    num_to_keep = len(models)-num_to_drop
    if num_to_drop >= len(models):
        print('Need to keep at least one')
        return
        
    # Generate all unique combinations of `num_to_drop` models
    for combo in combinations(models, num_to_drop):
        # Skip if this combination (or its permutations) was already processed
        if frozenset(combo) in used_combinations:
            continue

        # Add the combination to the used set
        used_combinations.add(frozenset(combo))
        
        # Drop the selected models from pred_df
        try:
            pred_new = pred_df.drop(columns=list(combo))
        except KeyError:
            pred_new = pred_df
            for mod in list(combo):
                try:
                    pred_new = pred_new.drop(columns=[mod])
                except KeyError:
                    print(f'cant remove {mod}')
                    continue
                    
        pred_new['AvPred'] = np.mean(pred_new, axis=1)
        kept_models = [mod for mod in models if mod not in list(combo)]
        model_name = f'c_{"_".join(kept_models)}'

        # Calculate predictions and metrics
        y_pred = pred_new.AvPred.values
        auc_score = roc_auc_score(y_target, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_target, y_pred.round()).ravel()

        # Store the result in scores
        score_dict = {
            "Subset": subset,
            "Gen": gen,
            "Model": model_name,
            "Fold": fold,
            "AUC": auc_score,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        in_scores.append(score_dict)
    return in_scores

#############################################################################
if __name__ == '__main__':
    
    
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


    TARGET      = wildcards['iTarget']
    SUBSET      = wildcards['iSubset']
    GEN         = wildcards['iGen']
    MODEL_NAME  = wildcards['iModel']
    FOLD        = int(wildcards['iFold'])

# alternative input method for testing
    model = sys.argv[1]
    
    path = read_config('root_path')
    models = read_config('all_models')
    gens = read_config('genotypes')
    if len(sys.argv)>2:
        print(sys.argv)
        subsets = [sys.argv[2]]
    else:
        subsets = read_config('subsets')
    
    
    # read files
    
    pred_dir = path + 'out/analysis/predictions/'    
###

    id_name = {'m': 'Full_sentrix', 'combine': 'Preg_id', 'f':'Full_sentrix'}
    
    
    scores=[]
    for subset in subsets:
        for gen in gens:
            for fold in range(5):
                pred_file = pred_dir + f'avpred_{subset}_{gen}_{fold}.csv'
                try:
                    pred_df = pd.read_csv(pred_file, sep='\t').set_index(id_name[gen])
                    y_target = pred_df['Target'].values
                    pred_df = pred_df.drop(columns=['Target'])
                    if model == 'all':
                        if len(pred_df.columns)>7:
                            pred_df['AvPred'] = np.mean(pred_df, axis=1)
                            y_pred = pred_df.AvPred.values
                            auc_score = roc_auc_score(y_target, y_pred)
                            tn, fp, fn, tp = confusion_matrix(y_target, y_pred.round()).ravel()
                            score_dict = {
                            "Subset": subset,
                            "Gen": gen,
                            "Model": model,
                            "Fold": fold,
                            "AUC": auc_score,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "tp": tp
                            }
                            scores.append(score_dict)
                        else:
                            print(f'Not enough models in {subset}, {gen}, Fold{fold}')
                            continue                    
                    else:
                        num_to_drop = int(model.removeprefix(('n')))
                        num_to_keep = len(models)-num_to_drop

                        if len(pred_df.columns)<num_to_keep:
                            print(f'Not enough models available in {subset}, {gen}, Fold{fold}')
                            continue
                        elif len(pred_df.columns) == num_to_keep:
                            available_models = pred_df.columns
                            num_to_drop = 0
                            scores = calculate_scores(
                                pred_df, 
                                y_target, 
                                available_models, 
                                num_to_drop, 
                                subset, 
                                gen, 
                                fold, 
                                scores
                                )
                        elif len(pred_df.columns) < len(models):
                            available_models = pred_df.columns
                            num_to_drop = len(available_models) - num_to_keep
                            scores = calculate_scores(
                                pred_df, 
                                y_target, 
                                available_models, 
                                num_to_drop, 
                                subset, 
                                gen, 
                                fold, 
                                scores
                                )
                        else:              
                            scores = calculate_scores(
                                pred_df, 
                                y_target, 
                                models, 
                                num_to_drop, 
                                subset, 
                                gen, 
                                fold, 
                                scores
                                )
                except FileNotFoundError:
                    print(f'Missing: {subset},{gen},Fold{fold}')
                    continue

                        
                                       
    
    
  #  save_file = path + f'out/tables/comb_score_{model}.csv'
    
    score_df = pd.DataFrame.from_records(scores)
    score_df.to_csv(out_file, sep='\t', float_format='%.5f', index=False)
    
    
