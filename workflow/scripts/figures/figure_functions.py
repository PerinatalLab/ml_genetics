import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as met
import numpy as np
import seaborn as sns
import seaborn.objects as so
import warnings
warnings.filterwarnings("ignore")



def plotmap(dataset, xname, yname):
    # plot distribution of results
    #EAEAEA
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_context("talk")

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(dataset, row=yname, hue=yname, aspect=16, height=.6, palette=pal)

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot, 
        xname,
        bw_adjust=.5, 
        clip_on=False,
        fill=True, 
        alpha=1, 
        linewidth=1.5)
    g.map(
        sns.kdeplot, 
        xname, 
        clip_on=False, 
        color="w", 
        lw=2, 
        bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, xname)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
    g.figure.suptitle(yname)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    
    


def plot_auc(pred_data, y_data, fig_gen, model, rng=5) -> None:
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    for fold in range(rng - 1):
        fold += 1
        y_train_og, y_test = gt.get_ybatch(y_data, fold)
        y_pred = extract_model_fold(pred_data, mod, fold)
        viz = met.RocCurveDisplay.from_predictions(
            y_test["Trait"],
            y_pred,
            name=f"{mod} fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
                )
        # interpolate tpr to avoid arrays of different lengths
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = met.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.plot([0, 1], [0, 1], "k--")
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"{mod}: Mean ROC curve with variability)",
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        plt.show()

def plot_box(df, x_col, y_col, h_col, c_col):
    n_colors = df[h_col].nunique()
    pal = sns.cubehelix_palette(n_colors=n_colors, rot=-.3, light=.7)
    yMin = df[y_col].min()
    yMax = df[y_col].max()
    plt.figure()
    g = sns.catplot(data=df, x=x_col, y=y_col,kind='box', aspect=2, height=3, hue=h_col, col=c_col, palette=pal, legend='auto', col_wrap=1)
    g.refline(y=df[y_col].median())
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.ylim(yMin, yMax)
    plt.show()
