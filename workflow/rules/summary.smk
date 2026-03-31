# Rules: summary


## summarize results: Helper rule to expand parameters
rule make_summary:
    input:
        expand(config["out_summary"] + "combined_{nModels}.csv", nModels=NMODELS),
        config["out_summary"] + "complete_summary.csv",
        config["out_summary"] + "complete_scores.csv",
    output:
        config["checks"] + "summary_done.txt",
    shell:
        "touch {output[0]}"


## append different subsets to one df
rule append_subsets:
    input:
        script=config["scripts_postprocess"] + "combine_subsets.py",
        data=expand(config["out_summary"] + "comb_score_{iSubset}.csv", iSubset=SUBSETS),
    output:
        combined_subsets=config["out_summary"] + "complete_summary.csv",
    log:
        config["log"] + "summary/complete_summary.txt",
    conda:
        "../envs/summary.yml"
    shell:
        "python {input.script} --out {output.combined_subsets} \
                --data {input.data}"


## append folds and gens to one df per subset
rule append_fold_gen:
    input:
        script=config["scripts_postprocess"] + "combine_pred.py",
        data=expand(
            config["out_pred"] + "avpred_{{iSubset}}_{iGen}_{iFold}.csv",
            iGen=GENOME,
            iFold=FOLDS,
        ),
    output:
        pred=config["out_summary"] + "comb_score_{iSubset}.csv",
    log:
        config["log"] + "summary/comb_score_{iSubset}.txt",
    conda:
        "../envs/summary.yml"
    shell:
        "python {input.script} --out {output.pred} \
                --wild {wildcards}"


## combine different number of models.
rule combine_models:
    input:
        script=config["scripts_postprocess"] + "full_comb.py",
        data=expand(
            config["out_pred"] + "avpred_{iSubset}_{iGen}_{iFold}.csv",
            iSubset=SUBSETS,
            iGen=GENOME,
            iFold=FOLDS,
        ),
    output:
        combined_models=config["out_summary"] + "combined_{nModels}.csv",
    log:
        config["log"] + "summary/combined_{nModels}.txt",
    conda:
        "../envs/summary.yml"
    shell:
        "python {input.script} --out {output.combined_models} \
                --data {input.data}"


## combine_subsets_scores: combine scores from different subsets to one df.
rule combine_subsets_scores:
    input:
        expand(
            config["out_summary"] + "combined_{iSubset}.csv",
            iSubset=SUBSETS,
        ),
    output:
        config["out_summary"] + "complete_scores.csv",
    log:
        config["log"] + "summary/complete_scores.txt",
    run:
        import pandas as pd

        combined_df = pd.concat([pd.read_csv(f) for f in input]).to_csv(
            output[0],
            sep="\t",
            index=False,
            float_format="%.5f",
        )


## combine scores from different folds, genomes and models to one df per subset.
rule combine_scores:
    input:
        script=config["scripts_postprocess"] + "combine_scores.py",
        data=expand(
            config["out_scores"] + "PTD/{{iSubset}}/{iModel}_{iGen}_{iFold}.csv",
            iModel=MODELS,
            iGen=GENOME,
            iFold=FOLDS,
        ),
    output:
        combined_models=config["out_summary"] + "combined_{iSubset}.csv",
    log:
        config["log"] + "summary/combined_{iSubset}.txt",
    conda:
        "../envs/summary.yml"
    shell:
        "python {input.script} --out {output.combined_models} \
                --data {input.data}"


## average predictions over all runs and save.
rule average_predictions:
    input:
        script=config["scripts_postprocess"] + "combine_runs.py",
        data=expand(
            config["out_pred"]
            + "{iTarget}/{{iSubset}}/{iModel}_{{iGen}}_{{iFold}}.csv",
            iTarget="PTD",
            iModel=MODELS,
        ),
    output:
        average_pred=config["out_pred"] + "avpred_{iSubset}_{iGen}_{iFold}.csv",
    log:
        config["log"] + "summary/avpred_{iSubset}_{iGen}_{iFold}.txt",
    conda:
        "../envs/summary.yml"
    shell:
        "python {input.script} \
        --out {output.average_pred} \
        --data {input.data} \
        --wild {wildcards}"

## average predictions over all runs and save.
rule sum_model_params:
    input:
        data=expand(
            config["out_summary"] + "params/{{iTarget}}/{{iModel}}_{iSubset}.csv",
            iSubset=SUBSETS,
        ),
    output:
        hyperpars=config["out_summary"] + "params/{iTarget}/{iModel}.csv",
    log:
        config["log"] + "summary/{iTarget}/hyperpar_{iModel}.txt",
    conda:
        "../envs/summary.yml"
    script:
        config["scripts_postprocess"] + "sum_hyperparameters.py"

## average predictions over all runs and save.
rule sum_subset_model_params:
    input:
        data=expand(
            config["out_params"]
            + "{{iTarget}}/{{iSubset}}/{{iModel}}_{iGen}_{iFold}.json",
            iGen=GENOME,
            iFold=FOLDS,
        ),
    output:
        hyperpars=config["out_summary"] + "params/{iTarget}/{iModel}_{iSubset}.csv",
    log:
        config["log"] + "summary/{iTarget}/hyperpar_{iModel}_{iSubset}.txt",
    conda:
        "../envs/summary.yml"
    script:
        config["scripts_postprocess"] + "sum_hyperparameters.py"
