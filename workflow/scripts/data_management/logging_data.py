import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

import optuna
from optuna.trial import TrialState


import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

sys.path.append("/mnt/work/hedvig/grepos/plab_workflow/")


def make_logger(log_file, name, **kwargs):
    """_summary_

    :param log_file: _description_
    :type log_file: _type_
    :param name: _description_
    :type name: _type_
    :return: _description_
    :rtype: _type_
    """
    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    log_dir = "/mnt/work/hedvig/grepos/plab_workflow/logs/prediction/parameters/logfiles/"
    if log_dir and not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir + log_file
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    f1 = logging.FileHandler(log_file, mode="a+")
    f1.setLevel(logging.INFO)
    my_formatter = logging.Formatter(
        "[%(levelname)1.1s %(asctime)s];%(message)s", "%y-%m-%d %H:%M"
    )
    f1.setFormatter(my_formatter)
    logger.addHandler(f1)

    return logger


def log_study_results(study):
    study_name = study.study_name
    target, subset, model_name, gen, fold = study_name.rsplit("_")
    direction = study.user_attrs["DIRECTION"]
    n_trials = study.user_attrs["TRIALS"]
    model_type = study.user_attrs["TYPE"]
    log_file = f"{study_name}.log"

    logger = make_logger(log_file, name="optuna_study")

    logger.info("Study finished")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned_trials_usr = [
        trial for trial in study.trials if trial.user_attrs.get("pruned", False)
    ]

    logger.info(f"Study statistics: ")
    for key, value in study.user_attrs.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of pruned trials(usr): {len(pruned_trials_usr)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")
    logger.info(f"  Number of trials on the Pareto front: {len(study.best_trials)}")

    logger.info("Best trials:")
    for trial in study.best_trials:
        logger.info(f"  Trial {trial.number}:  {trial.values}")
    n_direction = direction == "maximize"
    best_val_trial = sorted(
        study.best_trials, key=lambda t: t.values[0], reverse=n_direction
    )[0]
    logger.info(f"  Best Val: {best_val_trial.number}")
    best_train_trial = sorted(
        study.best_trials, key=lambda t: t.values[1], reverse=n_direction
    )[0]
    logger.info(f"  Best Train: {best_train_trial.number}")

    trial = sorted(
        study.best_trials,
        key=lambda t: sorted(t.values[:], reverse=(n_direction == False)),
        reverse=n_direction,
    )[0]
    logger.info(f"Statistics from trial {trial.number}: ")
    logger.info(f"  Val value: {trial.values[0]}")
    logger.info(f"  Train value: {trial.values[1]}")

    logger.info(f"  User Params: ")
    for key, value in trial.user_attrs.items():
        logger.info(
            f"    {key}: {round(value,5) if isinstance(value, float) else value}"
        )

    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info(
            f"    {key}: {round(value,5) if isinstance(value, float) else value}"
        )

    first_values = [trial.values[0] for trial in study.trials]
    if np.var(first_values) != 0:
        logger.info(f"  Param importance: ")
        par_imp = optuna.importance.get_param_importances(
            study, target=lambda t: t.values[0]
        )
        for key, value in par_imp.items():
            logger.info(
                f"    {key}: {round(value,5) if isinstance(value, float) else value}"
            )
    else:
        logger.info(f"All suck")
        return pd.DataFrame()

    pd.set_option("display.max_columns", None)
    df = study.trials_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == n_trials

    df = df.drop(columns=["datetime_start", "datetime_complete", "state"])
    df.columns = df.columns.str.replace(r"params_", "")
    df.columns = df.columns.str.replace(r"user_attrs", "")
    df.columns = df.columns.str.replace(r"values_0", "Val_score")
    df.columns = df.columns.str.replace(r"values_1", "Train_score")
    base_d = pd.date_range("2023-10-25 00:00:00.000000", periods=n_trials)
    df["duration"] = (base_d + df["duration"]).dt.strftime("%M:%S")

    logger.info(f"\n {df}")

    best_numbers = (trial.number for trial in study.best_trials)
    df_best = df.loc[best_numbers]

    print("done")
    logger.removeHandler(logger.handlers[0])
    return df_best


def log_single_results(study):
    study_name = study.study_name
    target, subset, model_name, gen, fold = study_name.rsplit("_")
    direction = study.user_attrs["DIRECTION"]
    n_trials = study.user_attrs["TRIALS"]
    model_type = study.user_attrs["TYPE"]
    log_file = f"{study_name}.log"

    logger = make_logger(log_file, name="optuna_study")

    logger.info("Study finished")

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned_trials = [
        trial for trial in study.trials if trial.user_attrs.get("pruned", False)
    ]

    logger.info(f"Study statistics: ")
    for key, value in study.user_attrs.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    # logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(
        f"  Number of complete trials: {len(complete_trials)- len(pruned_trials)}"
    )
    logger.info(f"  Number of trials on the Pareto front: {len(study.best_trials)}")

    logger.info("Best trials:")
    for trial in study.best_trials:
        logger.info(f"  Trial {trial.number}:  {trial.values}")
    n_direction = direction == "maximize"
    best_val_trial = sorted(
        study.best_trials, key=lambda t: t.values[0], reverse=n_direction
    )[0]
    logger.info(f"  Best Val: {best_val_trial.number}")
    # best_train_trial = sorted(
    #     study.best_trials, key=lambda t: t.values[1], reverse=n_direction
    # )[0]
    # logger.info(f"  Best Train: {best_train_trial.number}")

    trial = sorted(
        study.best_trials,
        key=lambda t: sorted(t.values[:], reverse=(n_direction == False)),
        reverse=n_direction,
    )[0]
    logger.info(f"Statistics from trial {trial.number}: ")
    logger.info(f"  Val value: {trial.values[0]}")
    # logger.info(f"  Train value: {trial.values[1]}")

    logger.info(f"  User Params: ")
    for key, value in trial.user_attrs.items():
        logger.info(
            f"    {key}: {round(value,5) if isinstance(value, float) else value}"
        )

    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info(
            f"    {key}: {round(value,5) if isinstance(value, float) else value}"
        )

    first_values = [
        trial.values[0]
        for trial in study.trials
        if trial is not None and trial.values is not None
    ]
    if target != "PTD":
        logger.info(f"  Param importance: ")
        par_imp = optuna.importance.get_param_importances(
            study, target=lambda t: t.values[0]
        )
        for key, value in par_imp.items():
            logger.info(
                f"    {key}: {round(value,5) if isinstance(value, float) else value}"
            )
    else:
        logger.info(f" Trial variance: {np.var(first_values)} ")
        if np.var(first_values) > 1e-4:
            # logger.info(f" Trial variance: {np.var(first_values)} ")
            logger.info(f"  Param importance: ")
            par_imp = optuna.importance.get_param_importances(
                study, target=lambda t: t.values[0]
            )
            for key, value in par_imp.items():
                logger.info(
                    f"    {key}: {round(value,5) if isinstance(value, float) else value}"
                )
        elif np.max(first_values) < 0.51:
            logger.info(f"All suck")
            return pd.DataFrame()
        else:
            logger.info(f"Equally bad")
    pd.set_option("display.max_columns", None)
    df = study.trials_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == n_trials

    df = df.drop(columns=["datetime_start", "datetime_complete", "state"])
    df.columns = df.columns.str.replace(r"params_", "")
    df.columns = df.columns.str.replace(r"user_attrs", "")
    df.columns = df.columns.str.replace(r"values_0", "Val_score")
    # df.columns = df.columns.str.replace(r"values_1", "Train_score")
    base_d = pd.date_range("2023-10-25 00:00:00.000000", periods=n_trials)
    df["duration"] = (base_d + df["duration"]).dt.strftime("%M:%S")

    logger.info(f"\n {df}")

    best_numbers = (trial.number for trial in study.best_trials)
    df_best = df.loc[best_numbers]

    print("done")
    logger.removeHandler(logger.handlers[0])
    return df_best


"""
def log_study_results(study):
    STUDYNAME = study.study_name
    DIRECTION = study.user_attrs["DIRECTION"]
    TRIALS = study.user_attrs["TRIALS"]
    TARGET = study.user_attrs["TARGET"]
    log_file = (
        f"/mnt/work/workbench/hedvigs/snake_book/econ/logs/parameters/{TARGET}/net/{STUDYNAME}.log"
    )

    logger = make_logger(log_file, name="optuna_study")

    logger.info("Study finished")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info(f"Study statistics: ")

    for key, value in study.user_attrs.items():
        logger.info("  {}: {}".format(key, value))
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials:  {len(pruned_trials)}")
    logger.info(f"  Number of complete trials:  {len(complete_trials)}")
    logger.info(f"  Number of trials on the Pareto front: {len(study.best_trials)}")

    logger.info("Best trials:")
    for trial in study.best_trials:
        logger.info(f"  Trial {trial.number}:  {trial.values}")
    n_direction = DIRECTION == "maximize"
    # best_test_trial = sorted(
    #     study.best_trials, key=lambda t: t.values[0], reverse=n_direction
    # )[0]
    # logger.info(f"  Best Test:  {best_test_trial.number}")
    best_val_trial = sorted(
        study.best_trials, key=lambda t: t.values[0], reverse=n_direction
    )[0]
    logger.info(f"  Best Val: {best_val_trial.number}")
    best_train_trial = sorted(
        study.best_trials, key=lambda t: t.values[1], reverse=n_direction
    )[0]
    logger.info(f"  Best Train: {best_train_trial.number}")
    # best_pval_trial = sorted(
    #     study.best_trials, key=lambda t: t.values[3], reverse=(n_direction == False)
    # )[0]
    # logger.info(f"  Best pvalue: {best_pval_trial.number}")

    trial = sorted(
        study.best_trials,
        key=lambda t: sorted(t.values[:], reverse=(n_direction == False)),
        reverse=n_direction,
    )[0]

    logger.info(f"Statistics from trial {trial.number}: ")

    # logger.info(f"  Test value: {trial.values[0]}")
    logger.info(f"  Val value:  {trial.values[0]}")
    logger.info(f"  Train value: {trial.values[1]}")
#    logger.info(f"  p-value: {trial.values[3]}")

    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info(
            "    {}: {}".format(
                key, round(value, 5) if isinstance(value, float) else value
            )
        )

    first_values = [trial.values[0] for trial in study.trials]
    if np.var(first_values) != 0:
        logger.info(f"  Param importance: ")
        par_imp = optuna.importance.get_param_importances(
            study, target=lambda t: t.values[0]
        )
        for key, value in par_imp.items():
            logger.info(
                "    {}: {}".format(
                    key, round(value, 5) if isinstance(value, float) else value
                )
            )
    else:
        logger.info(f"All suck")
        return pd.DataFrame()

    pd.set_option("display.max_columns", None)
    df = study.trials_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == TRIALS  # n_trials.
    df.drop(columns=["datetime_start", "datetime_complete", "state"], inplace=True)
    df.columns = df.columns.str.replace(r"params_", "")
    df.columns = df.columns.str.replace(r"values_0", "Test_score")
    df.columns = df.columns.str.replace(r"values_1", "Valid_score")
    df.columns = df.columns.str.replace(r"values_2", "Train_score")
    base_d = pd.date_range("2023-10-25 00:00:00.000000", periods=TRIALS)
    df["duration"] = (base_d + df["duration"]).dt.strftime("%M:%S")
    logger.info(f"\n {df}")
    best_numbers = (trial.number for trial in study.best_trials)
    df_best = df.loc[best_numbers]
    # feather.write_feather(df_best, par_file)
    print("done")
    logger.removeHandler(logger.handlers[0])
    return df_best

def log_classic_results(study):
    # TODO: if user params = params remove user params
    STUDYNAME = study.study_name
    DIRECTION = study.user_attrs["DIRECTION"]
    TRIALS = study.user_attrs["TRIALS"]
    TARGET = study.user_attrs["TARGET"]
    log_file = (
        f"/mnt/work/workbench/hedvigs/snake_book/econ/logs/parameters/{TARGET}/classic/{STUDYNAME}.log"
    )

    logger = make_logger(log_file, name="optuna_study")

    logger.info("Study finished")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info(f"Study statistics: ")

    for key, value in study.user_attrs.items():
        logger.info("  {}: {}".format(key, value))
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials:  {len(pruned_trials)}")
    logger.info(f"  Number of complete trials:  {len(complete_trials)}")
    logger.info(f"  Number of trials on the Pareto front: {len(study.best_trials)}")

    logger.info("Best trials:")
    for trial in study.best_trials:
        logger.info(f"  Trial {trial.number}:  {trial.values}")
    n_direction = DIRECTION == "maximize"
    best_test_trial = sorted(
        study.best_trials, key=lambda t: t.values[0], reverse=n_direction
    )[0]
    logger.info(f"  Best val score:  {best_test_trial.number}")
    best_train_trial = sorted(
        study.best_trials, key=lambda t: t.values[1], reverse=n_direction
    )[0]
    logger.info(f"  Best train mean: {best_train_trial.number}")
    # best_p_trial = sorted(
    #     study.best_trials, key=lambda t: t.values[2], reverse=(n_direction == False)
    # )[0]
    # logger.info(f"  Best pvalue: {best_p_trial.number}")

    # trial = sorted(
    #     study.best_trials,
    #     key=lambda t: sorted(t.values[:], reverse=n_direction), #(n_direction == False)),
    #     reverse=n_direction,
    # )[0]
    trial = sorted(
        study.best_trials,
        key=lambda t: sorted(t.values[:], reverse=(n_direction == False)),
        reverse=n_direction,
    )[0]
    logger.info(f"Statistics from trial {trial.number}: ")

    logger.info(f"  val value: {trial.values[0]}")
    logger.info(f"  Train mean value:  {trial.values[1]}")
    # logger.info(f"  p value: {trial.values[2]}")
    logger.info(f"  User Params: ")
    for key, value in trial.user_attrs.items():
        logger.info(
            "    {}: {}".format(
                key, round(value, 5) if isinstance(value, float) else value
            )
        )
    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info(
            "    {}: {}".format(
                key, round(value, 5) if isinstance(value, float) else value
            )
        )
    first_values = [trial.values[0] for trial in study.trials]
    if np.var(first_values) != 0:
        logger.info(f"  Param importance: ")
        par_imp = optuna.importance.get_param_importances(
            study, target=lambda t: t.values[0]
        )
        for key, value in par_imp.items():
            logger.info(
                "    {}: {}".format(
                    key, round(value, 5) if isinstance(value, float) else value
                )
            )
    else:
        logger.info(f"All suck")
        return pd.DataFrame()

    pd.set_option("display.max_columns", None)
    df = study.trials_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == TRIALS  # n_trials.
    df.drop(columns=["datetime_start", "datetime_complete", "state"], inplace=True)
    df.columns = df.columns.str.replace(r"params_", "")
    df.columns = df.columns.str.replace(r"user_attrs", "")
    df.columns = df.columns.str.replace(r"values_0", "val_score")
    df.columns = df.columns.str.replace(r"values_1", "train_score")
    # df.columns = df.columns.str.replace(r"values_2", "p_value")
    base_d = pd.date_range("2023-10-25 00:00:00.000000", periods=TRIALS)
    df["duration"] = (base_d + df["duration"]).dt.strftime("%M:%S")
    logger.info(f"\n {df}")
    best_numbers = (trial.number for trial in study.best_trials)
    df_best = df.loc[best_numbers]
    print("done")
    logger.removeHandler(logger.handlers[0])
    return df_best
    """
