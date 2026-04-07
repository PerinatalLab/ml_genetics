import warnings
import json
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.linear_model import LogisticRegression as LRC
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.neural_network import MLPRegressor as RNN
from sklearn.covariance import (
    OAS,
    EmpiricalCovariance,
    ShrunkCovariance,
    LedoitWolf,
)
from sklearn.metrics import (
    roc_auc_score,
    r2_score,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    make_scorer,
    confusion_matrix,
)
from sklearn.exceptions import ConvergenceWarning

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from data_management.subsets import load_data
from data_management.resampling import sampling
from data_management.parsing_set import ParseKwargs
from data_management.setup_data import read_config
from analysis.evaluation_functions import ps_permutation_test_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="Setting penalty=None will ignore the C and l1_ratio parameters",
    category=UserWarning,
)


def get_data(study_id):
    [[x_train, x_test], [y_train, y_test]] = load_data(study_id)
    y_train = y_train[TARGET]
    y_test = y_test[TARGET]
    return [[x_train, x_test], [y_train, y_test]]


def get_model(params):
    kwargs = {}

    if TARGET != "PTD":
        model_options = {"nn": RNN, "rfc": RFR}
    else:
        model_options = {
            "bnb": BNB,
            "dtc": DTC,
            "lda": LDA,
            "lrc": LRC,
            "qda": QDA,
            "rfc": RFC,
            "svc": SVC,
            "knn": KNN,
            "nn": MLP,
        }
    covariance_options = {
        "OAS": OAS,
        "EC": EmpiricalCovariance,
        "SC": ShrunkCovariance,
        "LW": LedoitWolf,
        "none": None,
    }

    model_name = MODEL_NAME

    if model_name == "bnb":
        kwargs["fit_prior"] = params["_fit_prior"]
        kwargs["binarize"] = params["_binarize"]
        kwargs["alpha"] = params["_alpha"]

    elif model_name == "dtc":
        kwargs["criterion"] = params["_criterion"]
        kwargs["splitter"] = params["_splitter"]
        kwargs["max_features"] = params["_max_features"]
        kwargs["max_leaf_nodes"] = params["_max_leaf_nodes"]
        kwargs["ccp_alpha"] = params["_ccp_alpha"]

    elif model_name == "lda":
        kwargs["solver"] = params["_solver"]
        kwargs["tol"] = params["_tol"]
        selected_cov = params["_covariance_estimator"]
        if selected_cov != "none":
            kwargs["covariance_estimator"] = covariance_options.get(selected_cov)()
        else:
            kwargs["covariance_estimator"] = None

    elif model_name == "lrc":
    #    kwargs["penalty"] = params["_penalty"]
        kwargs["tol"] = params["_tol"]
        kwargs["C"] = params["_C"]
        kwargs["fit_intercept"] = params["_fit_intercept"]
        kwargs["class_weight"] = params["_class_weight"]
        kwargs["max_iter"] = params["_max_iter"]
        kwargs["solver"] = params["_solver"]
    #    kwargs["multi_class"] = params["_multi_class"]
        kwargs["l1_ratio"] = params["_l1_ratio"]
        kwargs["warm_start"] = params["_warm_start"]

    elif model_name == "qda":
        kwargs["solver"] = params["_solver"]
        kwargs["reg_param"] = params["_reg_param"]
        kwargs["store_covariance"] = params["_store_covariance"]
        kwargs["tol"] = params["_tol"]
        if kwargs["solver"] == "eigen":
            kwargs["shrinkage"] = params["_shrinkage"]

    elif model_name == "rfc":
        kwargs["n_estimators"] = params["_n_estimators"]
        kwargs["criterion"] = params["_criterion"]
        kwargs["min_samples_split"] = params["_min_samples_split"]
        kwargs["min_samples_leaf"] = params["_min_samples_leaf"]
        kwargs["max_features"] = params["_max_features"]
        kwargs["min_impurity_decrease"] = params["_min_impurity_decrease"]
        kwargs["ccp_alpha"] = params["_ccp_alpha"]
        kwargs["warm_start"] = params["_warm_start"]
        kwargs["bootstrap"] = params["_bootstrap"]
        kwargs["oob_score"] = params["_oob_score"]
        kwargs["max_samples"] = params["_max_samples"]
        kwargs["class_weight"] = params["_class_weight"]

    elif model_name == "svc":
        kwargs["C"] = params["_C"]
        kwargs["kernel"] = params["_kernel"]
        kwargs["degree"] = params["_degree"]
        kwargs["gamma"] = params["_gamma"]
        kwargs["coef0"] = params["_coef0"]
        kwargs["shrinking"] = params["_shrinking"]
        kwargs["probability"] = True
        kwargs["tol"] = params["_tol"]
        kwargs["class_weight"] = params["_class_weight"]

    elif model_name == "knn":
        kwargs["algorithm"] = params["_algorithm"]
        kwargs["leaf_size"] = params["_leaf_size"]
        kwargs["metric"] = params["_metric"]
        kwargs["n_neighbors"] = params["_n_neighbors"]
        kwargs["p"] = params["_p"]

    elif model_name == "nn":
        kwargs["hidden_layer_sizes"] = params["_hidden_layer_sizes"]
        kwargs["activation"] = params["_activation"]
        kwargs["solver"] = params["_solver"]
        kwargs["alpha"] = params["_alpha"]
        kwargs["batch_size"] = params["_batch_size"]
        kwargs["warm_start"] = params["_warm_start"]
        kwargs["max_iter"] = params["_max_iter"]
        if kwargs["solver"] == "adam":
            kwargs["learning_rate_init"] = params["_learning_rate_init"]
            kwargs["shuffle"] = params["_shuffle"]
            kwargs["early_stopping"] = params["_early_stopping"]
            if kwargs["early_stopping"]:
                kwargs["validation_fraction"] = params["_validation_fraction"]
            kwargs["beta_1"] = params["_beta_1"]
            kwargs["beta_2"] = params["_beta_2"]
            kwargs["epsilon"] = params["_epsilon"]
            kwargs["n_iter_no_change"] = params["_n_iter_no_change"]
        elif kwargs["solver"] == "sgd":
            kwargs["learning_rate_init"] = params["_learning_rate_init"]
            kwargs["learning_rate"] = params["_learning_rate"]
            if kwargs["learning_rate"] == "invscaling":
                kwargs["power_t"] = params["_power_t"]
            kwargs["shuffle"] = params["_shuffle"]
            kwargs["momentum"] = params["_momentum"]
            if kwargs["momentum"] > 0:
                kwargs["nesterovs_momentum"] = params["_nesterovs_momentum"]
            kwargs["early_stopping"] = params["_early_stopping"]
            if kwargs["early_stopping"]:
                kwargs["validation_fraction"] = params["_validation_fraction"]
            kwargs["n_iter_no_change"] = params["_n_iter_no_change"]
        elif kwargs["solver"] == "lbfgs":
            kwargs["max_fun"] = params["_max_fun"]

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model_options[model_name](**kwargs)


def predict_fold(params, data, scorer):
    [[x_train, x_test], [y_train, y_test]] = data

    if TARGET == "PTD":
        x_train, y_train = sampling(x_train, y_train, "under", rs=None)
    else:
        x_train = x_train.drop_duplicates()
        sen = x_train.index
        y_train = y_train.loc[sen]

    model = get_model(params)

    if isinstance(model, RFC) and getattr(model, "warm_start", False):
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        model.class_weight = dict(zip(classes, class_weights))

    n_perm = 500
    test_score, test_perm, test_pval = ps_permutation_test_score(
        model,
        [x_train, x_test],
        [y_train, y_test],
        scorer=scorer,
        n_permutations=n_perm,
        random_state=None,
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if TARGET == "PTD" else y_pred

    score_dict = {
        "number": params["number"],
        "test_perm": np.mean(test_perm),
        "test_pval": test_pval,
        "test_score": test_score,
    }

    if TARGET == "PTD":
        score_dict.update(
            {
                "train_score": roc_auc_score(
                    y_train, model.predict_proba(x_train)[:, 1], average="weighted"
                ),
                "auc_prob": roc_auc_score(y_test, y_prob, average="weighted"),
                "auc_pred": roc_auc_score(y_test, y_pred, average="weighted"),
                "r2": r2_score(y_test, y_prob),
            }
        )
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        score_dict.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    else:
        score_dict.update(
            {
                "r2": r2_score(y_test, y_pred),
                "ex_var": explained_variance_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "mape": mean_absolute_percentage_error(y_test, y_pred),
            }
        )

    pred_name = f"pred_{STUDYNAME}_{params['number']}"
    pred_df = pd.DataFrame({pred_name: y_prob, "Target": y_test})
    pred_df.set_index(x_test.index, inplace=True)

    return score_dict, pred_df


if __name__ == "__main__":
    print("Starting model evaluation...")
    parser = argparse.ArgumentParser(
        description="Model evaluation script. Loads tuned parameters from JSON, runs repeated test-set evaluation, and saves scores and predictions."
    )
    parser.add_argument("-o", "--out", help="Input JSON file with tuned parameters")
    parser.add_argument("-u", "--utils", help="Output path for scores TSV file")
    parser.add_argument("-d", "--data", help="Data path (passed through to load_data)")
    parser.add_argument("-p", "--pheno", help="Phenotype file")
    parser.add_argument("-w", "--wild", action=ParseKwargs, help="Wildcard key=value pairs")
    args = parser.parse_known_args()
    args = args[0] if len(args) > 0 else args
    
    wildcards = args.wild
    params_file = args.data
    score_file = args.utils
    pred_file = args.out

    TARGET = wildcards["iTarget"]
    SUBSET = wildcards["iSubset"]
    GEN = wildcards["iGen"]
    MODEL_NAME = wildcards["iModel"]
    FOLD = int(wildcards["iFold"])

    ## CHANGE: Adjust N and MAX_PARAM_SETS
    N = 20  # Number of repeated evaluations per parameter set
    MAX_PARAM_SETS = 100  # Maximum number of parameter sets to evaluate

    STUDYNAME = f"{TARGET}_{SUBSET}_{MODEL_NAME}_{GEN}_{FOLD}"

    # Load parameters from JSON
    with open(params_file, "r") as f:
        json_par = json.load(f)
    print(f"Loaded {len(json_par)} parameter sets from {params_file}")

    scorer = (
        make_scorer(roc_auc_score, response_method="predict_proba")
        if TARGET == "PTD"
        else "r2"
    )
    data = get_data(STUDYNAME)

    scores = []
    pred_df = None

    for key in json_par.keys():
        if len(scores) >= MAX_PARAM_SETS:
            break
        parameters = json_par[key]
        print(f"Evaluating parameter set {key}: {parameters}")

        for n in range(N):
            score_dict, y_pred = predict_fold(parameters, data, scorer)
            scores.append(score_dict)

            pred_name = f"pred_{STUDYNAME}_{parameters['number']}"
            if pred_df is None:
                pred_df = y_pred.copy()
            else:
                if any(y_pred[pred_name].equals(pred_df[col]) for col in pred_df.columns):
                    print(f"  Prediction {n} is identical to a previous run, skipping.")
                    continue
                if y_pred["Target"].equals(pred_df["Target"]):
                    y_pred = y_pred.drop(columns=["Target"])
                pred_df = pred_df.join(y_pred, rsuffix=str(n))

            print(f"  Run {n + 1}/{N} done")

    # Save scores
    score_df = pd.DataFrame.from_records(scores)
    score_df.to_csv(score_file, sep="\t", float_format="%.5f", index=False, mode="a")
    print(f"Scores saved to {score_file}")

    # Save predictions
    #    path = read_config("root_path")
    #    pred_dir = path + "results/analysis/predictions/"
    #    pred_file = pred_dir + f"pred_{STUDYNAME}.csv"
    pred_df.to_csv(pred_file, sep="\t", float_format="%.5f")
    print(f"Predictions saved to {pred_file}")
