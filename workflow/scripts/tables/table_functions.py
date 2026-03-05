import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn.objects as so
import re
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# visuals

def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')


def style_negative(v, props=''):
    return props if v < 0 else None
    
    
def plus_minus_format(val):
    # format for std presentation
    return f"± {val:.2f}"


def ci_format(val):
    # format for ci presentation
    ci_low, ci_high = val
    return f"({ci_low:.2f}-{ci_high:.2f})"


# Aggregate results

def agg_table(df,categories=['subset'], valname=['AUC'], functions=["median", "mean", "max", "min", 'std'], color_funcs=["mean", "median"], cmaps=None,
 **kwargs):

   #group and aggregate for functions    
   if isinstance(functions, dict):
    nkeys = len(functions.keys())
    # Custom logic to interpret 'CI' as confidence_interval function
    for key, funcs in functions.items():
        functions[key] = [(col if col != 'CI' else ('95% CI', confidence_interval)) for col in funcs]
    if nkeys != len(valname):
        vals = [val for val in valname if val not in functions.keys()]
        for val in vals:
            functions[val] = ['mean']
        
   agg_df = df.groupby(categories)[valname].agg(functions)
 #  ci_df = df.groupby(categories)[valname].apply(lambda x: confidence_interval(x['AUC']))
 #  agg_df = pd.concat([agg_df, ci_df], axis=1)

   # extract all columns to be colored
   colored_columns = [col for col in agg_df.columns if col[1] in color_funcs]
   std_columns = [col for col in agg_df.columns if col[1] == 'std']
   max_columns = [col for col in agg_df.columns if col[1] == 'max']
   ci_columns = [col for col in agg_df.columns if col[1] == 'CI']
   
   styled_df = agg_df.style.format(precision=3).format(subset=std_columns, formatter=plus_minus_format)\
       .format(subset=ci_columns, formatter=ci_format)\
       .apply(highlight_max, props='font-weight: bold;', subset=max_columns)

    # set up color maps for different metrics
   if cmaps is None:
      cmaps = {'AUC': 'Greens',
               'pVal': 'Reds',
               'OR': 'Greens',
               'NLR': 'Reds',
               'FP': 'Reds',
               'FN': 'Reds'}


   for col in colored_columns:
       # only look at the first part of the colname to decide cmap
      colname = re.sub(r'\(.*?\)', '', col[0]).strip()
      cmap_to_use = cmaps.get(colname, 'Greens')
      styled_df = styled_df.background_gradient(cmap=cmap_to_use, axis=0, subset=[col])
   
   return styled_df


def rename_models(df, colname='Model'):
    modelname = {'bnb': 'B Naive Bayes', 'lrc': 'Log. Regression', 'rfc': 'Random Forest', 'dtc': 'Decision Tree', 'svc': 'Support Vector', 'lda': 'LDA', 'qda': 'QDA', 'PGS': 'Polygenic Score', 'knn': 'k-Nearest Neighbor', 'nn': 'Neural Network'}
    
    for i, mod in enumerate(df[colname]):
        df.loc[i, 'ModelName'] = modelname[mod]
    
    return df



def save_tex(df, filename, site='home'):
    # save as latex table

    if site != 'home':
        file_path= f'/home/hedvigs/PycharmProjects/homewrs/snake_book/econ/out/tables/{filename}.tex'
    else:
        file_path=f'/home/hedvigs/snake_book/econ/out/tables/{filename}.tex'
    # Get the number of columns in the DataFrame
    num_columns = len(df.columns)
    num_index = len(df.index.names)
    num_funcs = len(df.columns.levels[0])
    num_rows = len(df.index)
    print(num_columns)
    print(num_funcs)
    longtable=None
    pos_float = 'centering'
    if num_rows >= 26:
        longtable='longtable'
        pos_float=None

    # Create the column format string: one "l" followed by "r" repeated for each column in df
    column_format = "l"* num_index + "|" + "r" * num_columns

    df.to_latex(file_path, 
                convert_css=True, 
                column_format=column_format, 
                position="H", 
                position_float=pos_float,
                hrules=True,
                clines="skip-last;index",
                label=f"tab:{filename}",
                caption=f"{filename.split('_')[0]}",
                multirow_align="t",
                multicol_align="c|",
                environment=longtable)
    print('Table saved')
        # Read the LaTeX file to modify it
    if longtable==None:
        with open(file_path, 'r') as f:
            latex_content = f.read()

        # Insert the custom LaTeX commands
        latex_content = latex_content.replace(r"\centering", r"""\centering
    \begin{footnotesize}""")
        latex_content = latex_content.replace(r"\end{tabular}", r"""\end{tabular}
    \end{footnotesize}""")
        
        # Split content to manipulate specific parts
        lines = latex_content.split("\\")

        # Construct new header line with proper formatting for LaTeX
        header_line_index = 9 + (num_columns-num_funcs) -1 # Assuming header is on the 3rd line
        header_line = lines[header_line_index]
        print(header_line)
        # Extract the current column names for proper placement
        current_columns = [col.strip() for col in header_line.split(' & ') if col.strip()]
        print(current_columns)
        col_headers = " & ".join(current_columns[:])  # Skip the initial empty and index column parts
        print(col_headers)
        new_header_line = r" " + r" & ".join(df.index.names) + r" & " + col_headers #+ r" \\ "  # New header line with LaTeX format
        print(new_header_line)

        # Update LaTeX content with new header line
        lines[header_line_index] = new_header_line
        lines[header_line_index + 2] = r" " # r" & ".join(df.index.names) + " & " + " & " * (num_columns - 1) + r" \\"  # Modify to use `\\` for LaTeX
        print(lines[10:14])
        # Join lines back together
        modified_latex_content = "\\".join(lines)
        modified_latex_content = modified_latex_content.replace(r"""\\ \\""", r""" \\ """)

        modified_latex_content = modified_latex_content.replace(r"95%", r"95\%")
        
        # Write the modified LaTeX content back to the file
        with open(file_path, 'w') as f:
            f.write(modified_latex_content)
    else:
        with open(file_path, 'r') as f:
            latex_content = f.read()
            
        # Split content to manipulate specific parts
        lines = latex_content.split("\\")

        # Construct new header line with proper formatting for LaTeX
        header_line_index = 8 + num_funcs  # Assuming header is on the 3rd line
        header_line = lines[header_line_index]
        print('header', header_line)
        # Extract the current column names for proper placement
        current_columns = [col.strip() for col in header_line.split(' & ') if col.strip()]
        print(current_columns)
        col_headers = " & ".join(current_columns[:])  # Skip the initial empty and index column parts
        print(col_headers)
        new_header_line = r" " + r" & ".join(df.index.names) + r" & " + col_headers #+ r" \\ "  # New header line with LaTeX format
        print(new_header_line)

        # Update LaTeX first header line
        lines[header_line_index] = new_header_line
        lines[header_line_index + 2] = r" " 
        print(lines[10:14])
        # Join lines back together
        modified_latex_content = "\\".join(lines)
        modified_latex_content = modified_latex_content.replace(r"""\\ \\""", r""" \\ """)

#        print(lines[header_line_index + num_funcs + 12])
        second_header_line_index = header_line_index + num_funcs + 12
        # Update LaTeX second header line
        lines[second_header_line_index] = new_header_line
        lines[second_header_line_index + 2] = r" " 
        print(lines[22:26])
        # Join lines back together
        modified_latex_content = "\\".join(lines)
        modified_latex_content = modified_latex_content.replace(r"""\\ \\""", r""" \\ """)
        #modified_latex_content = modified_latex_content.replace(r"cline{1-2}", r"cline{1-2} \pagebreak")

        modified_latex_content = modified_latex_content.replace(r"95%", r"95\%")
        
        # Write the modified LaTeX content back to the file
        with open(file_path, 'w') as f:
            f.write(modified_latex_content)
    
    print('Table saved and customized.')
    

def get_subset(y_data, colname='Model', rowname='bnb'):
    sub_data= y_data[y_data[colname]== rowname]
    
    return sub_data

    
def get_rownames(df):
    cat_names = df.select_dtypes(include=['O']).columns.tolist()
    val_names = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove fold from value names
    val_names = [col for col in val_names if col.lower() != 'fold']

    rownames={}
    for name in cat_names:
        rown = np.unique(df[name], equal_nan=True)
        rownames[name] = rown
    return cat_names, val_names, rownames


def confidence_interval(group, confidence_level=0.95):
    # Calculate confidence interval for each group
    mean_val = group.mean()
    std_val = group.std()
    n = len(group)
    if n > 1:  # To avoid division by zero or undefined CI for small groups
        se_val = std_val / np.sqrt(n)
        degrees_freedom = n - 1
        ci_lower, ci_upper = stats.t.interval(confidence_level, degrees_freedom, mean_val, se_val)
        ci_interval = (float(ci_lower.round(2)), float(ci_upper.round(2)))
        return ci_interval
    else:
        return (np.nan, np.nan)#pd.Series([np.nan, np.nan], index=['ci_lower', 'ci_upper'])


def calculate_relative_risk(plr, nlr):
    """
    Calculate the relative risk (RR) based on the positive likelihood ratio (PLR) and negative likelihood ratio (NLR).

    Parameters:
    plr (float): Positive Likelihood Ratio
    nlr (float): Negative Likelihood Ratio

    Returns:
    float: Relative Risk (RR)
    """
    # Avoid division by zero for NLR
    #nlr = np.where(nlr == 0, 0.001, nlr)
    # Sensitivity and Specificity derived from PLR and NLR
    sensitivity = 1 / (1 + nlr)
    specificity = 1 - (1 / (1 + plr))
    
    # Relative Risk calculation
    rr = sensitivity / (1 - specificity)
    
    return rr, sensitivity, specificity

def calc_mcc(tn, fp, fn, tp):
    numerator = (tp*tn) - (fp*fn)
    denominator = np.sqrt((tp+fp)* (tp+fn) *(tn+fp)* (tn+fn))
    with np.errstate(divide='ignore', invalid='ignore'):
        mcc = np.where(denominator != 0, numerator/denominator, 0)
    return mcc

def calc_f1_score(fp, fn, tp):
    precision = calc_precision(fp, tp) #tp/(tp+fp) if (tp+fp) != 0 else 0
    recall = calc_sensitivity(fn, tp) #tp/ (tp+fn) if (tp+fn) != 0 else 0
    return 2 * (precision*recall) / (precision + recall) if (precision + recall) != 0 else 0

def calc_balanced_acc(tn, fp, fn, tp):
    sensitivity = calc_sensitivity(fn,tp)
    specificity = calc_specificity(tn, fp)
    return (sensitivity + specificity) / 2

def calc_or(tn, fp, fn, tp):
    # Gives 0 if or is 0 or inf 
    with np.errstate(divide='ignore', invalid='ignore'):
        oddsr = np.where( fn*fp != 0, (tp *tn) / (fp * fn), 0)
    return oddsr
    # if fn == 0 or fp == 0:
    #     return float('inf') if tp>0 and tn>0 else 0
    # return (tp * tn) / (fp * fn)

def calc_rr(tn, fp, fn, tp):
    risk_in_exposed = calc_sensitivity(fn, tp)
    risk_in_unexposed = calc_fpr(tn,fp) #fp / (fp + tn) if (fp + tn) != 0 else 0
    with np.errstate(divide='ignore', invalid='ignore'): # handle div by zero
        rr = np.where(risk_in_unexposed != 0, risk_in_exposed/risk_in_unexposed, float('inf'))
    return rr #pd.Series(rr) if isinstance(tn, pd.Series) else rr

def calc_lr(tn, fp, fn, tp):
    sensitivity = calc_sensitivity(fn, tp)
    specificity = calc_specificity(tn, fp)
    plr = sensitivity / (1 - specificity) if (1-specificity) != 0 else float('inf')
    nlr = (1 - sensitivity) / specificity if specificity != 0 else float('inf')
    return plr, nlr

def calc_precision(fp, tp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def calc_fpr(tn, fp):
    with np.errstate(divide='ignore', invalid='ignore'):  # To handle division by zero gracefully
        fpr = np.where((tn + fp) != 0, fp / (tn + fp), 0)
    return fpr #pd.Series(fpr) if isinstance(tn, pd.Series) else fpr

    
def calc_sensitivity(fn, tp):
    with np.errstate(divide='ignore', invalid='ignore'):  # To handle division by zero gracefully
        sensitivity = np.where((tp + fn) != 0, tp / (tp + fn), 0)
    return sensitivity #pd.Series(sensitivity) if isinstance(tp, pd.Series) else sensitivity
    # sens = tp / (tp + fn) if (tp + fn) != 0 else 0
 
def calc_specificity(tn, fp):
    with np.errstate(divide='ignore', invalid='ignore'):  # To handle division by zero gracefully
        spec = np.where((tn + fp) != 0, tn / (tn + fp), 0)
    return spec #pd.Series(spec) if isinstance(tn, pd.Series) else spec
#    return tn / (tn + fp) if (tn + fp) != 0 else 0

def get_all_metrics(df, **kwargs):
    # Get OR, RR, Sensitivity, Specificity, MCC and Balanced accuracy
    
    tn = kwargs.get('tn') if 'tn' in kwargs else df.tn 
    fp = kwargs.get('fp') if 'fp' in kwargs else df.fp
    fn = kwargs.get('fn') if 'fn' in kwargs else df.fn
    tp = kwargs.get('tp') if 'tp' in kwargs else df.tp

    df['OR']            = calc_or(tn, fp, fn, tp)
    df['RR']            = calc_rr(tn, fp, fn, tp)
    df['Sensitivity']   = calc_sensitivity(fn, tp)
    df['Specificity']   = calc_specificity(tn, fp)
    df['MCC']           = calc_mcc(tn, fp, fn, tp)
    df['BalancedAcc']   = calc_balanced_acc(tn,fp,fn,tp)

    return df

def rename_cols(df):
    cname_dict = {
        'test_perm': 'AUC(perm)', 
        'test_pval': 'pVal(test)',
        'test_score': 'AUC(test)', 
        'train_score': 'AUC(train)', 
        'auc_prob': 'AUC(prob)',
        'auc_pred': 'AUC(binary)', 
        'r2': 'r2', 
        'tn': 'TN', 
        'fp': 'FP', 
        'fn': 'FN', 
        'tp': 'TP', 
        'fold': 'Fold', 
        'gen': 'MaternalFetal',
        'Gen': 'MaternalFetal', 
        'model': 'Model',
        'subset': 'SNPs',
        'Subset': 'SNPs', 
        'OR': 'OR', 
        'RR': 'RR', 
        'Sensitivity': 'Sens', 
        'Specificity': 'Spec', 
        'MCC': 'MCC',
        'BalancedAcc': 'BalAcc', 
        'cAucMean': 'AUC(cMean)', 
        'cAucMedian': 'AUC(cMedian)', 
        'ctn': 'TN(cMean)', 
        'cfp': 'FP(cMean)', 
        'cfn': 'FN(cMean)', 
        'ctp': 'TP(cMean)',
        'OR_c': 'OR(cMean)', 
        'RR_c': 'RR(cMean)', 
        'Sensitivity_c': 'Sens(cMean)', 
        'Specificity_c': 'Spec(cMean)', 
        'MCC_c': 'MCC(cMean)',
        'BalancedAcc_c': 'BalAcc(cMean)'}
    dfcc= df.rename(columns=cname_dict)
    return dfcc



def name_models(df):
    df.reset_index(drop=True, inplace=True)
    modelname = {'bnb': 'B Naive Bayes', 'lrc': 'Log. Regression', 'rfc': 'Random Forest', 'svc': 'Support Vector', 'lda': 'LDA', 'qda': 'QDA', 'PGS': 'Polygenic Score', 'knn': 'k-Nearest Neighbor', 'nn': 'Neural Network'}
    for i, mod in enumerate(df['Model']):
        df.loc[i, 'ModelName'] = modelname[mod]
    return df

def format_df(df, gen=None, nsub=None, sub=None, mod=None):
    df.drop(columns=['Unnamed: 0'], inplace=True)
    if gen != None:
        df = df[df.gen==gen]
    if sub != None:
        df = df[df.subset==sub]
    elif nsub != None:
        df = df[df.subset!=nsub]
    if mod != None:        
        df = df[df.model == mod]
    return df

def rename_subsets(df):
    df.loc[df["subset"] == "tops", "subset"] = "Top23"
    df.loc[df["subset"] == "top5", "subset"] = "Top5"
    df.loc[df["subset"] == "all", "subset"] = "All"
    df.loc[df["subset"] == "selected", "subset"] = "Selected"
    return df


def divide_concur(dfc):
    # divide com results and individual results for combined df
    c_cols = [col for col in dfc.columns if col.startswith('c')]
    dfc1 = dfc[c_cols]
    dfc2 = dfc.drop(columns=c_cols)

    #get additional metrics
    dfc1 = get_all_metrics(dfc1, tn=dfc.ctn, fn=dfc.cfn,tp=dfc.ctp,fp=dfc.cfp)
    dfc2 = get_all_metrics(dfc2)

    # combine com data again
    dfcc = dfc2.merge(dfc1, suffixes=(None, '_c'), how='left', left_index=True, right_index=True)
    return dfcc



