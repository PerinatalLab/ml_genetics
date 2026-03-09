
# Rules: summary
#

## summarize results: Helper rule to expand parameters
rule summarize_results:
    input:
        expand(config["out_summary"] + "combined_{nModels}.csv",
                                        nModels = NMODELS
                                        ),
        config["out_summary"] + "complete_summary.csv",
    output:
        config["checks"] + "summary_done.txt",
    shell:        "touch {output[0]}"


rule append_subsets:
    input:
        script = config["scripts_postprocess"] + "combine_subsets.py",
        data = expand(config["out_summary"] + "comb_score_{iSubset}.csv",
                                        iSubset = SUBSETS
                                        ),
    output:
        combined_subsets = config["out_summary"] + "complete_summary.csv"
    shell:
        "python {input.script} --out {output.combined_subsets} \
                --data {input.data}"


## append folds and gens to one df per subset
rule append_fold_gen:
    input:
        script = config["scripts_postprocess"] + "combined_pred.py",
        data = expand(config["out_pred"] + "avpred_{{iSubset}}_{iGen}_{iFold}.csv",
                                        iGen = GENOME,
                                        iFold = FOLDS
                                        ),
    output:
        pred = config["out_summary"] + "comb_score_{iSubset}.csv",
    shell:        
        "python {input.script} --out {output.pred} \
                --wild {wildcards}"

rule combine_models:
    input:
        script = config["scripts_postprocess"] + "full_comb.py",
        data = expand(config["out_pred"] + "avpred_{iSubset}_{iGen}_{iFold}.csv",
                                        iSubset = SUBSETS,
                                        iGen = GENOME,
                                        iFold = FOLDS),
    output:
        combined_models = config["out_summary"] + "combined_{nModels}.csv"
    shell:
        "python {input.script} --out {output.combined_models} \
                --data {input.data}"


## append all models to one df per subset, gen and fold
rule append_models:
    input:
        script = config["scripts_postprocess"] + "combine_runs.py",
    
    output:
        average_pred = config["out_pred"] + "avpred_{iSubset}_{iGen}_{iFold}.csv"
    shell:
        "python {input.script} --out {output.average_pred} \
                --wild {wildcards}"
                                        