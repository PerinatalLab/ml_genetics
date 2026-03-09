
# Rules: summary
#

## summarize results: Helper rule to expand parameters
rule summarize_results:
    input:
        expand(config["out_summary"] + "comb_score_{nModels}.csv",
                                        nModels = NMODELS
                                        ),
    output:
        config["checks"] + "summary_done.txt",
    shell:        "touch {output[0]}"

rule combine_subsets:
    input:
        script = config["scripts_postprocess"] + "full_comb.py",
        data = expand(config["out_tables"] + "comb_score_{iSubset}_{nModels}.csv",
                                        iSubset = SUBSETS,
                                        nModels = NMODELS),
    output:
        combined_subsets = config["out_summary"] + "comb_score_{nModels}.csv"
    shell:
        "python {input.script} --out {output.combined_subsets} \
                --data {input.data}"

## append folds and gens to one df per subset
rule combine_fold_gen:
    input:
        script = config["scripts_postprocess"] + "combined_pred.py",
    output:
        pred = config["out_summary"] + "comb_score_{iSubset}.csv",
    shell:        
        "python {input.script} --out {output.pred} \
                --wild {wildcards}"


## append all models to one df per subset, gen and fold
rule combine_models:
    input:
        script = config["scripts_postprocess"] + "combine_runs.py",
    
    output:
        average_pred = config["out_pred"] + "avpred_{iSubset}_{iGen}_{iFold}.csv"
    shell:
        "python {input.script} --out {output.average_pred} \
                --wild {wildcards}"
                                        