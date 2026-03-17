import warnings
import logging
import json
import argparse

from time import time
import numpy as np
import pandas as pd
import optuna

from optuna.samplers import TPESampler
import pyarrow.feather as feather
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
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
    MinCovDet,
    LedoitWolf,
)
from sklearn.metrics import roc_auc_score, r2_score, make_scorer
from sklearn.exceptions import ConvergenceWarning

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from data_management.subsets import load_data
from data_management.resampling import sampling
from data_management.parsing_set import ParseKwargs
from data_management.logging_data import log_single_results
from data_management.setup_data import read_config

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


def create_model(trial):
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
        kwargs["fit_prior"] = trial.suggest_categorical("fit_prior", [True, False])
        kwargs["binarize"] = trial.suggest_float("binarize", 1e-7, 1, log=True)
        kwargs["alpha"] = trial.suggest_float("alpha", 1e-7, 1, log=True)

    elif model_name == "dtc":
        kwargs["criterion"] = trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        )
        kwargs["splitter"] = trial.suggest_categorical("splitter", ["best", "random"])
        kwargs["max_features"] = trial.suggest_categorical(
            "max_features", [None, "sqrt", "log2"]
        )
        kwargs["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 2, 10)
        kwargs["ccp_alpha"] = trial.suggest_float("ccp_alpha", 1e-7, 1, log=True)

    elif model_name == "lda":
        kwargs["tol"] = trial.suggest_float("tol_svd", 2e-7, 1, log=True)
        if SUBSET == "all":
            kwargs["solver"] = "svd"
        else:
            kwargs["solver"] = trial.suggest_categorical(
                "solver_lda", ["lsqr", "eigen", "svd"]
            )
        if kwargs["solver"] == "svd":
            selected_cov = "none"
        elif kwargs["solver"] == "eigen":
            selected_cov = trial.suggest_categorical(
                "covariance_estimator_e", ["OAS", "SC", "LW"]
            )
        elif kwargs["solver"] == "lsqr":
            selected_cov = trial.suggest_categorical(
                "covariance_estimator_lsqr", list(covariance_options.keys())
            )
        if selected_cov != "none":
            kwargs["covariance_estimator"] = covariance_options.get(selected_cov)()
        else:
            kwargs["covariance_estimator"] = None

    elif model_name == "lrc":
        kwargs["penalty"] = trial.suggest_categorical(
            "penalty", ["l1", "l2", "elasticnet", None]
        )
        kwargs["warm_start"] = trial.suggest_categorical("warm_start", [False, True])
        if kwargs["penalty"] is not None:
            kwargs["multi_class"] = trial.suggest_categorical(
                "multi_class_l1", ["auto", "ovr"]
            )
            kwargs["C"] = trial.suggest_float("C", 1e-3, 3, log=True)
            if kwargs["penalty"] != "elasticnet":
                kwargs["l1_ratio"] = None
                if kwargs["penalty"] == "l1":
                    kwargs["solver"] = trial.suggest_categorical(
                        "solver_l1", ["liblinear", "saga"]
                    )
                elif kwargs["penalty"] == "l2":
                    kwargs["solver"] = trial.suggest_categorical(
                        "solver_l2",
                        ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga", "liblinear"],
                    )
                else:
                    raise ValueError("Invalid penalty name")
            else:
                kwargs["solver"] = trial.suggest_categorical("solver_en", ["saga"])
                kwargs["l1_ratio"] = trial.suggest_float("l1_ratio_en", 1e-4, 0.99, log=True)
        else:
            kwargs["solver"] = trial.suggest_categorical(
                "solver_none", ["lbfgs", "newton-cg", "saga"]
            )
            kwargs["multi_class"] = trial.suggest_categorical(
                "multi_class_none", ["auto", "ovr", "multinomial"]
            )
        kwargs["tol"] = trial.suggest_float("tol", 1e-6, 1, log=True)
        kwargs["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        kwargs["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", None])
        if kwargs.get("solver") == "sag":
            kwargs["max_iter"] = trial.suggest_int("sag_max_iter", int(1e5), int(1e7))
        else:
            kwargs["max_iter"] = trial.suggest_int("low_max_iter", int(1e3), int(5e3))

    elif model_name == "qda":
        if SUBSET == "top5":
            kwargs["reg_param"] = trial.suggest_float("t5_reg_param", 0.01, 1, log=True)
            kwargs["tol"] = trial.suggest_float("t5_tol", 1e-7, 1, log=True)
        else:
            kwargs["reg_param"] = trial.suggest_float("reg_param", 0.1, 1, log=True)
            kwargs["tol"] = trial.suggest_float("tol", 1e-5, 1, log=True)

    elif model_name == "rfc":
        kwargs["criterion"] = trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        )
        kwargs["min_samples_split"] = trial.suggest_float(
            "min_samples_split", 0.01, 0.99, log=True
        )
        kwargs["min_samples_leaf"] = trial.suggest_float(
            "min_samples_leaf", 0.01, 0.99, log=True
        )
        kwargs["max_features"] = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        )
        kwargs["min_impurity_decrease"] = trial.suggest_float(
            "min_impurity_decrease", 0.001, 1, log=True
        )
        kwargs["ccp_alpha"] = trial.suggest_float("ccp_alpha", 0.001, 1, log=True)
        kwargs["warm_start"] = trial.suggest_categorical("warm_start", [False, True])
        kwargs["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
        if kwargs["bootstrap"]:
            kwargs["oob_score"] = trial.suggest_categorical("oob_score", [True, False])
            kwargs["max_samples"] = trial.suggest_float("max_samples", 0.01, 0.99, log=True)
            kwargs["class_weight"] = trial.suggest_categorical(
                "class_weight_b", ["balanced_subsample", None]
            )
        else:
            kwargs["class_weight"] = trial.suggest_categorical(
                "class_weight", ["balanced", None]
            )
        kwargs["n_estimators"] = trial.suggest_int("n_estimators", 10, int(1e3))

    elif model_name == "svc":
        kwargs["C"] = trial.suggest_float("C", 1e-5, 3, log=True)
        kwargs["kernel"] = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        kwargs["degree"] = trial.suggest_int("degree", 2, 10)
        kwargs["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        kwargs["coef0"] = trial.suggest_float("coef0", 1e-5, 1)
        kwargs["shrinking"] = trial.suggest_categorical("shrinking", [True, False])
        kwargs["probability"] = True
        kwargs["tol"] = trial.suggest_float("tol", 1e-7, 1, log=True)
        kwargs["class_weight"] = trial.suggest_categorical(
            "class_weight_svc", [None, "balanced"]
        )

    elif model_name == "knn":
        kwargs["algorithm"] = trial.suggest_categorical(
            "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        )
        if kwargs["algorithm"] == "ball_tree":
            kwargs["weights"] = trial.suggest_categorical(
                "weights_kd", ["uniform", "distance", None]
            )
            kwargs["metric"] = trial.suggest_categorical(
                "metric_ball",
                [
                    "euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1",
                    "chebyshev", "infinity", "hamming", "canberra", "braycurtis", "jaccard",
                    "dice", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath",
                ],
            )
        elif kwargs["algorithm"] == "kd_tree":
            kwargs["weights"] = trial.suggest_categorical(
                "weights_kd", ["uniform", "distance", None]
            )
            kwargs["metric"] = trial.suggest_categorical(
                "metric_kd",
                ["euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1",
                 "chebyshev", "infinity"],
            )
        elif kwargs["algorithm"] == "brute":
            kwargs["weights"] = trial.suggest_categorical(
                "weights_brute", ["distance", None]
            )
            kwargs["metric"] = trial.suggest_categorical(
                "metric_brute",
                ["cosine", "euclidean", "manhattan", "nan_euclidean", "minkowski"],
            )
        elif kwargs["algorithm"] == "auto":
            kwargs["weights"] = trial.suggest_categorical(
                "weights_brute", ["distance", None]
            )
            kwargs["metric"] = trial.suggest_categorical(
                "metric_auto", ["euclidean", "manhattan", "minkowski"]
            )
        else:
            raise ValueError(f"Invalid algorithm name: {kwargs['algorithm']}")
        kwargs["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 20)
        kwargs["leaf_size"] = trial.suggest_int("leaf_size", 10, 60)
        if kwargs.get("metric") == "minkowski":
            kwargs["p"] = trial.suggest_float("p", 1, 2)

    elif model_name == "nn":
        hidden_layer_sizes = []
        n_layers = trial.suggest_int("n_layers", 1, 12)
        for i in range(n_layers):
            num_hidden = trial.suggest_int(f"n_units_l{i}", 4, 128, log=True)
            hidden_layer_sizes.append(num_hidden)
        kwargs["hidden_layer_sizes"] = np.array(hidden_layer_sizes)
        kwargs["activation"] = trial.suggest_categorical(
            "activation", ["logistic", "tanh", "relu"]
        )
        kwargs["solver"] = trial.suggest_categorical("solver", ["sgd", "adam"])
        kwargs["alpha"] = trial.suggest_float("alpha", 0.00001, 0.5, log=True)
        kwargs["batch_size"] = trial.suggest_int("batch_size", 10, 150)
        if kwargs["solver"] == "adam":
            kwargs["warm_start"] = trial.suggest_categorical("warm_start", [False, True])
            if not kwargs.get("warm_start", False):
                kwargs["early_stopping"] = trial.suggest_categorical(
                    "early_stopping", [True, False]
                )
                if kwargs["early_stopping"]:
                    kwargs["validation_fraction"] = trial.suggest_float(
                        "validation_fraction", 0.1, 0.4
                    )
            kwargs["learning_rate_init"] = trial.suggest_float(
                "lr_init_adam", 0.00001, 1, log=True
            )
            kwargs["shuffle"] = trial.suggest_categorical("shuffle", [True, False])
            kwargs["beta_1"] = trial.suggest_float("beta_1", 1e-5, 0.999, log=True)
            kwargs["beta_2"] = trial.suggest_float("beta_2", 1e-5, 0.999, log=True)
            kwargs["epsilon"] = trial.suggest_float("epsilon", 1e-8, 1e-3)
            kwargs["max_iter"] = trial.suggest_int("adam_max_iter", 200, 1000)
            kwargs["n_iter_no_change"] = trial.suggest_int("adam_iter_no_change", 2, 10)
        elif kwargs["solver"] == "sgd":
            kwargs["warm_start"] = True
            kwargs["learning_rate_init"] = trial.suggest_float(
                "lr_init_sgd", 0.000001, 0.1, log=True
            )
            kwargs["learning_rate"] = trial.suggest_categorical(
                "learning_rate", ["constant", "invscaling", "adaptive"]
            )
            if kwargs["learning_rate"] == "invscaling":
                kwargs["power_t"] = trial.suggest_float("power_t", 0.1, 0.3)
            kwargs["shuffle"] = trial.suggest_categorical("shuffle", [True, False])
            kwargs["momentum"] = trial.suggest_float("momentum", 1e-2, 0.8, log=True)
            if kwargs["momentum"] > 0:
                kwargs["nesterovs_momentum"] = trial.suggest_categorical(
                    "nesterovs_momentum", [True, False]
                )
            kwargs["max_iter"] = trial.suggest_int("sgd_max_iter", 500, 1500)
            kwargs["n_iter_no_change"] = trial.suggest_int("sgd_iter_no_change", 5, 15)
        else:
            kwargs["max_fun"] = trial.suggest_int("max_fun", 1000, 2000)

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model = model_options[model_name](**kwargs)
    usr_params = model.get_params()
    for key, val in usr_params.items():
        if key == "covariance_estimator":
            val = selected_cov if model_name == "lda" else val
        trial.set_user_attr(key, val)

    return model


def objective(trial):
    [[x_outer, x_test], [y_outer, y_test]] = get_data(trial.study.study_name)
    classes = np.unique(y_outer)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_outer
    )
    class_weight_dict = dict(zip(classes, class_weights))
    n_estimators_per_step = 10
    results = []

    model = create_model(trial)

    for step in range(n_train_iter):
        x_train, x_val, y_train, y_val = train_test_split(
            x_outer, y_outer, test_size=0.2, stratify=y_outer
        )
        if TARGET == "PTD":
            x_train, y_train = sampling(x_train, y_train, "under", rs=None)
        else:
            x_train = x_train.drop_duplicates()
            sen = x_train.index
            y_train = y_train.loc[sen]

        if hasattr(model, "partial_fit") and getattr(model, "warm_start", False):
            if TARGET != "PTD":
                model.partial_fit(x_train, y_train)
            else:
                model.partial_fit(x_train, y_train, classes=classes)
        elif isinstance(model, RFC) and getattr(model, "warm_start", False):
            model.class_weight = class_weight_dict
            model.n_estimators = model.n_estimators + step * n_estimators_per_step
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train)

        if TARGET != "PTD":
            y_val_pred = model.predict(x_val)
            intermediate_val = r2_score(y_val, y_val_pred)
        else:
            y_val_pred = model.predict_proba(x_val)[:, 1]
            intermediate_val = roc_auc_score(y_val, y_val_pred)

        results.append(intermediate_val)
        trial.report(intermediate_val, step)

        if trial.should_prune():
            trial.set_user_attr("pruned", True)
            return sum(results) / len(results)

    return sum(results) / len(results)


def run_tuning():
    study = optuna.create_study(
        study_name=STUDYNAME,
        directions=[DIRECTION],
        sampler=TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.WilcoxonPruner(p_threshold=PTHRESH, n_startup_steps=15),
    )

    start = time()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(
        objective,
        n_trials=TRIALS,
        gc_after_trial=True,
        show_progress_bar=False,
        catch=[ValueError],
    )
    print("Optimizing took %.2f seconds for %d trials." % ((time() - start), TRIALS))

    study.set_user_attr("DIRECTION", DIRECTION)
    study.set_user_attr("TRIALS", TRIALS)
    study.set_user_attr("TYPE", "classic")

    best_par = log_single_results(study)
    return best_par


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning script. Runs Optuna optimisation and saves best parameters to JSON."
    )
    parser.add_argument("-o", "--out", required=True, help="Output path for JSON parameter file")
    parser.add_argument("-d", "--data", help="Data path (passed through to load_data)")
    parser.add_argument("-p", "--pheno", help="Phenotype file")
    parser.add_argument("-w", "--wild", action=ParseKwargs, help="Wildcard key=value pairs")
    args = parser.parse_known_args()
    args = args[0] if len(args) > 0 else args

    wildcards = args.wild
    out_file = args.out

    TARGET = wildcards["iTarget"]
    SUBSET = wildcards["iSubset"]
    GEN = wildcards["iGen"]
    MODEL_NAME = wildcards["iModel"]
    FOLD = int(wildcards["iFold"])
    
    ### CHANGE: Adjust TRIALS
    TRIALS = 500
    n_train_iter = 30 if MODEL_NAME == "nn" else 20
    PTHRESH = 0.01 if MODEL_NAME == "nn" else 0.02

    directions_c = {"PTD": "maximize", "GA": "maximize", "NGA": "maximize"}
    DIRECTION = directions_c[TARGET]
    STUDYNAME = f"{TARGET}_{SUBSET}_{MODEL_NAME}_{GEN}_{FOLD}"

    best_par = run_tuning()
    json_par = best_par.to_json(orient="index")
    json_par = json.loads(json_par)
    print(f"Best parameters: {json_par}")

    with open(out_file, "w") as f:
        json.dump(json_par, f, indent=4)

    print(f"Parameters saved to {out_file}")