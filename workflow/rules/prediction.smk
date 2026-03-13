# Rules: parameters


## make_params: Helper rule to expand parameters
rule make_prediction:
    input:
        expand(
            config["checks"] + "chunks/{iTarget}/{iSubset}_done.txt",
            iTarget="PTD",
            iSubset=SUBSETS,
        ),
    output:
        config["checks"] + "prediction_done.txt",
    shell:
        "touch {output[0]}"


## make_params: Helper rule to expand parameters
rule models_done:
    input:
        expand(
            config["checks"] + "chunks/{{iTarget}}/{{iSubset}}/{iModel}_done.txt",
            iModel=MODELS,
        ),
    output:
        config["checks"] + "chunks/{iTarget}/{iSubset}_done.txt",
    shell:
        "touch {output[0]}"


## make_params: Helper rule to expand parameters
rule gens_done:
    input:
        expand(
            config["checks"]
            + "chunks/{{iTarget}}/{{iSubset}}/{{iModel}}/{iGen}_done.txt",
            iGEN=GENOME,
        ),
    output:
        config["checks"] + "chunks/{iTarget}/{iSubset}/{iModel}_done.txt",
    shell:
        "touch {output[0]}"


## folds_done: Helper rule to expand parameters
rule folds_done:
    input:
        expand(
            config["out_params"]
            + "{{iTarget}}/{{iSubset}}/{{iModel}}_{{iGen}}_{iFold}.json",
            iFold=FOLDS,
        ),
        expand(
            config["out_pred"]
            + "{{iTarget}}/{{iSubset}}/{{iModel}}_{{iGen}}_{iFold}.csv",
            iFold=FOLDS,
        ),
    output:
        config["checks"] + "chunks/{iTarget}/{iSubset}/{iModel}/{iGen}_done.txt",
    shell:
        "touch {output[0]}"


### param_pred: get best parameters and predictions for each fold, subset, model and genome
rule param_pred:
    input:
        script=config["scripts_analysis"] + "tune_prediction.py",
        check=config["checks"] + "datamgt.txt",
    output:
        best_params=config["out_params"]
        + "{iTarget}/{iSubset}/{iModel}_{iGen}_{iFold}.json",
        score_dir=config["out_pred"] + "{iTarget}/{iSubset}/{iModel}_{iGen}_{iFold}.csv",
    log:
        config["log"] + "prediction/{iTarget}/{iSubset}_{iGen}_{iModel}_{iFold}.txt",
    conda:
        "../envs/analysis.yml"
    shell:
        "python {input.script} \
            --out {output.best_params} \
            --utils {output.score_dir} \
            --wild {wildcards} \
                > {log} {logAll}"
