
# Rules: parameters
#-----------------------Dictionaries----------------------
ALL_MODELS = ["bnb", "knn", "lda", "lrc","nn", "qda", "rfc", "svc"] # all models to be tuned
GEN = ["m", "f", "combine"] # maternal, fetal, combined genotype data
FOLDS = [0, 1, 2, 3, 4]
SUBSETS = ["top5", "top23", "selected", "all"] # feature subsets to be tuned

## make_params: Helper rule to expand parameters
rule make_params:
    input:
        expand(config["out_params"] + "{iTarget}/{iType}/{iSubset}/{iModel}_{iGen}_{iFold}_pruned.json",
                                        iTarget = 'PTD',
                                        iType = 'classic',
                                        iSubset= 'all', 
                                        iModel= ALL_MODELS,
                                        iGen= GEN, 
                                        iFold = FOLDS),
        expand(config["out_analysis"] + "scores/{iTarget}/classic/{iSubset}/{iModel}_{iGen}_{iFold}_pruned.csv",
                                        iTarget = 'PTD',
                                        iType = 'classic',
                                        iSubset= 'all', 
                                        iModel= ALL_MODELS,
                                        iGen= GEN, 
                                        iFold = FOLDS),

rule param_pred:
    input:
        script      = config["src_analysis"] + "tune_prediction.py",
    output:
        best_params = config["out_params"] + "{iTarget}/classic/{iSubset}/{iModel}_{iGen}_{iFold}_pruned.json",
        score_dir   = config["out_analysis"] + "scores/{iTarget}/classic/{iSubset}/{iModel}_{iGen}_{iFold}_pruned.csv"
    log:
        config["log"] + "data_cleaning/analysis/{iTarget}/classic/{iSubset}_{iGen}_{iModel}_{iFold}_pruned.txt",
    conda:
        "workflow/envs/analysis.yml",
    shell:
        "python {input.script} \
        --out {output.best_params} \
        --utils {output.score_dir} \
        --wild {wildcards} \
            > {log} {logAll}"