
# Rules: parameters
#

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
    shell:
        "python {input.script} \
        --out {output.best_params} \
        --utils {output.score_dir} \
        --wild {wildcards} \
            > {log} {logAll}"