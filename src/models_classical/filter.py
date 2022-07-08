import pandas as pd
from pathlib import Path
import argparse
import logging
import os
import matplotlib

# run matplotlib without display
# https://stackoverflow.com/a/4706614/9214620
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models_classical.train import train_single_model
from src.models_classical.utils import get_model_metrics_df
from ast import literal_eval
from src.visualization.visualize import plot_pr_roc_curves_kfolds
import pickle


def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    final_dir_name = args.final_dir_name

    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        print("Assume on HPC")

        path_final_dir = scratch_path / "arxiv-code-search/models" / final_dir_name

    else:
        print("Assume on local compute")
        path_final_dir = proj_dir / "models" / final_dir_name

    return proj_dir, path_data_dir, path_final_dir


def filter_results_df(df, keep_top_n=None):
    dfr = df[
        (df["precision_score_min"] > 0)
        & (df["precision_score_max"] < 1)
        & (df["precision_score_std"] > 0)
        & (df["recall_score_min"] > 0)
        & (df["recall_score_max"] < 1)
        & (df["recall_score_std"] > 0)
        & (df["f1_score_min"] > 0)
        & (df["f1_score_max"] < 1)
        & (df["f1_score_std"] > 0)
        & (df["rocauc_min"] < 1)
        & (df["rocauc_max"] < 1)
        & (df["rocauc_avg"] < 1)
        & (df["rocauc_std"] > 0)
        & (df["prauc_min"] < 1)
        & (df["prauc_max"] < 1)
        & (df["prauc_avg"] < 1)
        & (df["prauc_std"] > 0)
        & (df["accuracy_min"] < 1)
        & (df["accuracy_max"] < 1)
        & (df["accuracy_avg"] < 1)
        & (df["accuracy_std"] > 0)
        & (df["n_thresholds_min"] > 3)
        & (df["n_thresholds_max"] > 3)
    ].sort_values(by=["prauc_avg", "rocauc_avg", "accuracy_avg"], ascending=False)

    if keep_top_n is not None:
        return dfr[:keep_top_n].reset_index(drop=True)
    else:
        return dfr.reset_index(drop=True)


def rebuild_params_clf(df, row_idx):
    classifier_string = df.iloc[row_idx]["classifier"]
    if classifier_string == "rf":
        prefix = "RandomForestClassifier"

    elif classifier_string == "xgb":
        prefix = "XGB"

    elif classifier_string == "knn":
        prefix = "KNeighborsClassifier"

    elif classifier_string == "lr":
        prefix = "LogisticRegression"

    elif classifier_string == "sgd":
        prefix = "SGDClassifier"

    elif classifier_string == "ridge":
        prefix = "RidgeClassifier"

    elif classifier_string == "svm":
        prefix = "SVC"

    elif classifier_string == "nb":
        prefix = "GaussianNB"

    params_clf = {
        c.replace(f"{prefix}_", ""): df.iloc[row_idx][c]
        for c in df.iloc[row_idx].dropna().index
        if c.startswith(prefix)
    }

    # convert any whole numbers in clf_cols to int
    for k in params_clf.keys():
        if isinstance(params_clf[k], float) and params_clf[k].is_integer():
            params_clf[k] = int(params_clf[k])

    return {k: [params_clf[k]] for k in params_clf.keys()}  # put each value in a list


def rebuild_general_params(df, row_idx, general_param_keys=None):
    if general_param_keys is None:
        general_param_keys = [
            "scaler_method",
            "oversamp_method",
            "undersamp_method",
            "oversamp_ratio",
            "undersamp_ratio",
            "classifier",
        ]
    return {k: [df.iloc[row_idx][k]] for k in general_param_keys}


def main(args):

    proj_dir, path_data_dir, path_final_dir = set_directories(args)

    df = pd.read_csv(
        path_final_dir / args.compiled_csv_name,
    )
    df = filter_results_df(df)

    # df = df[:500]

    # use this is you want to only select the top models by model type (e.g. top SVM, RF, etc.)
    sort_by = 'prauc_avg'
    df = df.groupby(['classifier']).head(args.keep_top_n).sort_values(by=sort_by, ascending=False)

    # save the top models to a csv
    df.to_csv(path_final_dir / args.filtered_csv_name, index=False)

    # save a certain number of PR-AUC and ROC-AUC curves
    if args.save_n_figures > 0:

        embedding_dir = path_data_dir / "processed/embeddings"
        with open(embedding_dir / "df_embeddings.pkl", "rb") as f:
            df_feat = pickle.load(f)

        percent_true = df_feat["label"].sum() / len(df_feat)
        print(f"Percent imbalance: {percent_true:.2%}")

        path_model_curves = path_final_dir / "model_curves"
        Path(path_model_curves).mkdir(parents=True, exist_ok=True)

        if args.save_models == "True":
            model_save_path = path_final_dir / "model_files"
            model_save_path.mkdir(parents=True, exist_ok=True)
            save_models = True

        for row_idx in range(args.save_n_figures):

            params_clf = rebuild_params_clf(df, row_idx)
            general_params = rebuild_general_params(df, row_idx)

            META_LABEL_COLS = ["para"]
            STRATIFICATION_GROUPING_COL = (
                "id"  # either arxiv paper id, or another unique id (such as the doi)
            )
            Y_LABEL_COL = "label"
            sampler_seed = int(df.iloc[row_idx]["sampler_seed"])
            id = df.iloc[row_idx]["id"]  # unique id for the specific model run

            print(args.save_models)
            (
                model_metrics_dict,
                params_dict_clf_named,
                params_dict_train_setup,
            ) = train_single_model(
                df_feat,
                sampler_seed,
                META_LABEL_COLS,
                STRATIFICATION_GROUPING_COL,
                Y_LABEL_COL,
                general_params=general_params,
                params_clf=params_clf,
                save_model=save_models,
                model_save_name=f"{id}.pkl",
                model_save_path=model_save_path,
                dataset_name=df.iloc[row_idx]["dataset"],
            )

            plot_pr_roc_curves_kfolds(
                model_metrics_dict["precisions_array"],
                model_metrics_dict["recalls_array"],
                model_metrics_dict["fpr_array"],
                model_metrics_dict["tpr_array"],
                model_metrics_dict["rocauc_array"],
                model_metrics_dict["prauc_array"],
                percent_anomalies_truth=percent_true,
                path_save_name=path_model_curves / f"curve_{id}.png",
                save_plot=True,
                dpi=300,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build data sets for analysis")

    parser.add_argument(
        "--keep_top_n",
        type=int,
        default=1,
        help="Keep the top N models in the filtered results CSV.",
    )

    parser.add_argument(
        "--save_n_figures",
        type=int,
        default=0,
        help="Keep the top N models in the filtered results CSV.",
    )

    parser.add_argument(
        "--final_dir_name",
        type=str,
        help="Folder name containing compiled csv.",
    )

    parser.add_argument(
        "--path_data_dir",
        dest="path_data_dir",
        type=str,
        help="Location of the data folder, containing the raw, interim, and processed folders",
    )

    parser.add_argument(
        "--compiled_csv_name",
        type=str,
        default="compiled_results.csv",
        help="The compiled csv name that has not yet been filtered.",
    )

    parser.add_argument(
        "--filtered_csv_name",
        type=str,
        default="compiled_results_filtered.csv",
        help="The name of the compiled and filtered csv.",
    )

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--interim_dir_name",
        type=str,
        help="Folder name containing all the interim result csv's that will be compiled into one.",
    )

    parser.add_argument(
        "--save_models",
        type=str,
        default="False",
        help="Save the models, and scaler, to disk.",
    )

    args = parser.parse_args()

    main(args)
