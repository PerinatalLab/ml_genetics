# Rules: data-management


#----------------------Dictionaries----------------------
GENOME = ["m", "f", "combine"] # maternal, fetal, combined genotype data

## create genotype and phenotype data
rule get_data:
    input:
        expand(config["out_data"] + "x_{iGen}.feather",
                iGen=GENOME),
        y_data = config["out_data"] + "y_data.feather" 


## get_geno extract genotype data
rule reduce_features:
    input:
        script = config["src_data_mgt"] + "org_genotyped.py",
        x_data = config["out_data"] + "x_data.feather",
        y_data = config["out_data"] + "y_data.feather",
    output:
        x_data = config["out_data"] + "x_{iGen}.feather",
    log:
        config["log"] + "data_cleaning/x_{iGen}_data.txt"
    conda:
        "workflow/envs/datamngt.yml",
    shell:
        "python {input.script} \
            --out {output.x_data} \
            --data {input.x_data} \
            --pheno {input.y_data} \
            --wild {wildcards} \
                > {log} {logAll}"

## get_pheno extract phenotype data
rule get_raw:
    input:
        script=config["src_data_mgt"] + "get_raw.py",
        y_data=config["src_data_pheno"],
        x_data=config["data_path"],
    output:
        y_data=config["out_data"] + "y_data.feather",
        x_data=config["out_data"] + "x_data.feather",
    log:
        config["log"] + "data_cleaning/get_raw.txt",
    conda:
        "workflow/envs/datamngt.yml",
    shell:
        "python {input.script} \
        --out {output.y_data} {output.x_data} \
        --pheno {input.y_data} \
        --data {input.x_data} \
        > {log} {logAll}"


# ## get_full_geno extract genotype data for full set of maternal genotype data
# rule reduce_full_geno:
#     input:
#         script = config["src_data_mgt"] + "reduce_feat.py",
#         x_data = config["data_path"],
#         y_data = config["out_data"] + "y_full_{iGen}.feather"
#     output:
#         x_data = temp(config["out_data"] + "{iTarget}/x_{iSubset}_full_{iGen}.feather"),
#     params:
#         full = True,
#         list_top = config["top_path"]
#     log:
#         config["log"] + "data_cleaning/{iTarget}/{iSubset}_full_{iGen}_data.txt"
#     shell:
#         "python {input.script} \
#             --out {output.x_data} \
#             --data {input.x_data} \
#             --pheno {input.y_data} \
#             --full {params.full} \
#             --list_top {params.list_top} \
#             --wild {wildcards} \
#                 > {log} {logAll}"

# # get_full_pheno extract phenotype data for all maternal/fetal samples
# rule get_full_pheno:
#     input:
#         script = config["src_data_mgt"] + "get_raw.py",
#         y_data = config["src_data_pheno"],
#         x_data = config["data_path"],
#     output:
#         y_data_m = temp(config["out_data"] + "y_full_m.feather"),
#         y_data_f = temp(config["out_data"] + "y_full_f.feather")
#     log:
#         config["log"] + "data_cleaning/y_full_data.txt"
#     shell:
#         "python {input.script} \
#             --mat {output.y_data_m} \
#             --fet {output.y_data_f} \
#             --pheno {input.y_data} \
#             --data {input.x_data} \
#                 > {log} {logAll}"
