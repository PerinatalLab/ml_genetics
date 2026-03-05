
# Rules: parameters
#

## make_params: Helper rule to expand parameters
rule make_params:
    input:
        # expand(config["out_params"] + "{iTarget}/net/{iSubset}/{iGen}_nn_{iFold}.json",
        #                                 iSubset=DATA_SUBSET, 
        #                                 iGen=GEN, 
        #                                 iTarget='PTD',
        #                                 iFold = FOLDS),
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
        # expand(config["out_params"] + "{iTarget}/{iType}/{iSubset}/{iModel}_{iGen}_{iFold}.json",
        #                                 iTarget = 'PTD',
        #                                 iType = 'classic',
        #                                 iSubset= 'selected_features', 
        #                                 iGen=GEN, 
        #                                 iModel= 'knn',
        #                                 iFold = FOLDS)
                                        
## net parameters: Tune parameters for network model
rule net_parameters:
    input:
        script  = config["src_model_specs"] + "net_parameters_copy.py",
        y_data  = config["out_data"] + "y_data.feather",
        data    = config["out_data"] + "x_{iGen}.feather", 
    output:
        best_params = config["out_params"] + "{iTarget}/net/{iSubset}/nn_{iGen}_{iFold}.json"
    params:
        plog        = config["log"] + "parameters/net/"
    log:
        config["log"] + "parameters/{iTarget}/net/{iSubset}_nn_{iGen}_{iFold}.txt"
    shell:
        "python {input.script} --out {output.best_params} \
                --pheno {input.y_data} \
                --data {input.data} \
                --wild {wildcards} \
                    > {log} {logAll}" 
# class parameters: Tune parameters for classic models
#rule class_parameters:
#    input:
#        script = config["src_model_specs"] + "classic_net.py",
#        script = config["src_model_specs"] + "param_tune_prune.py",
#        y_data = config["out_data"] + "y_data.feather",
#        data = config["out_data"] + "x_{iGen}.feather",
#    output:
#        best_params = config["out_params"] + "PTD/classic/{iSubset}/{iModel}_{iGen}_{iFold}_pruned.json"
#    params:
#        plog        = config["log"] + "parameters/classic/"
#    log:
#        config["log"] + "parameters/PTD/classic/{iSubset}_{iModel}_{iGen}_{iFold}_pruned.txt"
#    shell:
#        "python {input.script} --out {output.best_params} \
#                --pheno {input.y_data} \
#                --data {input.data} \
#                --wild {wildcards} \
#                    > {log} {logAll}"

# rule all_parameters:
#    input:
#        script = config["src_model_specs"] + "ga_param_tune.py",
#    output:
#        best_params = config["out_params"] + "{iTarget}/classic/{iSubset}/{iModel}_{iGen}_{iFold}_pruned.json"
#    params:
#        plog        = config["log"] + "parameters/classic/"
#    log:
#        config["log"] + "parameters/{iTarget}/classic/{iSubset}_{iModel}_{iGen}_{iFold}_pruned.txt"
#    shell:
#        "python {input.script} --out {output.best_params} \
#                --wild {wildcards} \
#                    > {log} {logAll}"

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











## class parameters: Tune parameters for classic models
# rule class_parameters:
    # input:
        # script = config["src_model_specs"] + "classic_parameters_copy.py",
        # y_data = config["out_data"] + "y_data.feather",
        # data = config["out_data"] + "x_{iGen}.feather",
    # output:
        # best_params = config["out_params"] + "{iTarget}/classic/{iSubset}/{iModel}_{iGen}_{iFold}.json"
    # params:
        # plog        = config["log"] + "parameters/classic/"
    # log:
        # config["log"] + "parameters/{iTarget}/classic/{iSubset}_{iModel}_{iGen}_{iFold}.txt"
    # shell:
        # "python {input.script} --out {output.best_params} \
                # --pheno {input.y_data} \
                # --data {input.data} \
                # --wild {wildcards} \
                    # > {log} {logAll}"



# rule class_parame:
#     input:
#         script = config["src_model_specs"] + "classic_parameters copy.py",
#         y_data = config["out_data"] + "y_data.feather",
#         data = config["out_data"] + "x_{iGen}.feather", 
#     output:
#         best_params = config["out_params"] + "{iTarget}/classic/{iSubset}/lrc_{iGen}_{iFold}.json"
#     params:
#         plog        = config["log"] + "parameters/classic/"
#     log:
#         config["log"] + "parameters/{iTarget}/classic/{iSubset}_lrc_{iGen}_{iFold}.txt"
#     shell:
#         "python {input.script} --out {output.best_params} \
#                 --pheno {input.y_data} \
#                 --data {input.data} \
#                 --wild {wildcards} \
#                     > {log} {logAll}"

# ## net parameters: Tune parameters for network model
# rule net_parameters:
#     input:
#         script = config["src_model_specs"] + "net_params_ga.py",
#         y_data = config["out_data"] + "y_data.feather",
#         data = config["out_data"] + "{iTarget}/x_{iSubset}_{iGen}.feather", 
#     output:
#         best_params = config["out_params"] + "{iTarget}/net/{iSubset}/{iGen}/{iModel}.json"
#     params:
#         models      = '{iModel}',
#         gen         = '{iGen}',
#         fold        = T_FOLD,
#         sampling    = SAMPLING_METHOD,     
#         param_log = config["out_params"] + "{iTarget}/net/{iSubset}/{iGen}/{iModel}.log.json" 
#     log:
#         config["log"] + "parameters/{iTarget}/net/{iModel}.{iSubset}.{iGen}.txt"
#     conda:
#         "environments/atom_env.yml"
#     shell:
#         "python {input.script} --out {output.best_params} \
#                 --pheno {input.y_data} \
#                 --data {input.data} \
#                 --mod {params.models} \
#                 --gen {params.gen} \
#                 --fold {params.fold} \
#                 --utils {params.param_log} \
#                 --wild {wildcards} \
#                     > {log} {logAll}" 
# rule net_parameters2:
#     input:
#         script  = config["src_model_specs"] + "get_params_gpt.py",
#         y_data  = config["out_data"] + "y_data.feather",
#         data    = config["out_data"] + "x_{iGen}.feather", 
#     output:
#         best_params = config["out_params"] + "2/{iTarget}/net/{iSubset}/{iModel}_{iGen}_{iFold}.json"
#     params:
#         top_snps    = config["top_path"]
#     log:
#         config["log"] + "parameters/{iTarget}/net/{iSubset}_{iModel}_{iGen}_{iFold}_2.txt"
#     shell:
#         "python {input.script} --out {output.best_params} \
#                 --pheno {input.y_data} \
#                 --data {input.data} \
#                 --wild {wildcards} \
#                 --snps {params.top_snps} \
#                     > {log} {logAll}" 
