# configfile: "config.yaml"
# Rules: tables
#


## make_tables: Helper rule to expand wildcard and generate all tables
rule make_tables:
    input:
        expand(config["out_tables"] + "report/{iGen}.byModelSNPs.tex",
                                        iGen=GENOME),
        expand(config["out_tables"] + "report/{iSubset}.byModel_{iGen}.tex",
                                        iSubset = SUBSETS,
                                        iGen = GENOME),
        expand(config["out_tables"] + "report/{iModel}.byGenSNPs.tex",
                                        iModel =MODELS),
    output:
        config["checks"] + "tables_done.txt"
    shell:
        "touch {output[0]}"


        

# one gen: all models, all subsets
rule gen_tables:
    input:
        script = config["scripts_tables"] + "make_tex_table.py",
        data = config["out_summary"] + "complete_summary.csv"
    output:
        table = config["out_tables"] + "report/{iGen}.byModelSNPs.tex",        
    log:
        config["log"] + "tables/{iGen}_modelSNPs.txt"
    shell:
        "python {input.script}\
            --out {output.table} \
            --data {input.data} \
            --wild {wildcards} \
            > {log} {logAll}"

# One subset, one gen
rule subset_tables:
    input:
        script = config["scripts_tables"] + "make_tex_table.py",
        data   = config["out_summary"] + "complete_summary.csv",
    output:
        table = config["out_tables"] + "report/{iSubset}.byModel_{iGen}.tex",
    log:
        config["log"] + "tables/{iSubset}_{iGen}_model.txt"
    shell:
        "python {input.script}\
                --out {output.table} \
                --data {input.data} \
                --wild {wildcards} \
                    > {log} {logAll}"

# one model
rule model_tables:
    input:
        script = config["scripts_tables"] + "make_tex_table.py",
        data = config["out_summary"] + "complete_summary.csv"
    output:
        table = config["out_tables"] + "report/{iModel}.byGenSNPs.tex"        
    log:
        config["log"] + "tables/{iModel}_GenSNPs.txt"
    shell:
        "python {input.script}\
            --out {output.table} \
            --data {input.data} \
            --wild {wildcards} \
            > {log} {logAll}"