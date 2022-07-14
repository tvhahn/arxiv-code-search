import pandas as pd
from pathlib import Path
import argparse
import logging
import os
import matplotlib
import shutil

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

    if args.path_emb_dir:
        path_emb_dir = Path(args.path_emb_dir)
    else:
        path_emb_dir = path_data_dir / "processed" / "embeddings"

    final_dir_name = args.final_dir_name

    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        print("Assume on HPC")

        path_final_dir = scratch_path / "arxiv-code-search/models" / final_dir_name

    else:
        print("Assume on local compute")
        path_final_dir = proj_dir / "models" / final_dir_name

    return proj_dir, path_data_dir, path_emb_dir, path_final_dir


def filter_results_df(df, keep_top_n=None):
    dfr = df[
        (df["prauc_min"] < 1)
        & (df["prauc_max"] < 1)
        & (df["prauc_avg"] < 1)
        & (df["prauc_std"] > 0)
        & (df["rocauc_min"] < 1)
        & (df["rocauc_max"] < 1)
        & (df["rocauc_avg"] < 1)
        & (df["rocauc_std"] > 0)
        & (df["accuracy_min"] < 1)
        & (df["accuracy_max"] < 1)
        & (df["accuracy_avg"] < 1)
        & (df["accuracy_std"] > 0)
        & (df["precision_score_min"] > 0)
        & (df["precision_score_max"] < 1)
        & (df["precision_score_std"] > 0)
        & (df["recall_score_min"] > 0)
        & (df["recall_score_max"] < 1)
        & (df["recall_score_std"] > 0)
        & (df["f1_score_min"] > 0)
        & (df["f1_score_max"] < 1)
        & (df["f1_score_std"] > 0)
        & (df["n_thresholds_min"] > 3)
        & (df["n_thresholds_max"] > 3)
    ].sort_values(by=["prauc_avg", "rocauc_avg", "accuracy_avg"], ascending=False)

    if keep_top_n is not None:
        return dfr[:keep_top_n].reset_index(drop=True)
    else:
        return dfr.reset_index(drop=True)

def order_columns_on_results_df(df):

    primary_cols = [
            "classifier",
            "sampler_seed",
            "date_time",
            "dataset",
            "id",
            "scaler_method",
            "oversamp_method",
            "oversamp_ratio",
            "undersamp_method",
            "undersamp_ratio",
            "early_stopping_rounds",
            "prauc_min",
            "prauc_max",
            "prauc_avg",
            "prauc_std",
            "rocauc_min",
            "rocauc_max",
            "rocauc_avg",
            "rocauc_std",
            "accuracy_min",
            "accuracy_max",
            "accuracy_avg",
            "accuracy_std",
            "precision_score_min",
            "precision_score_max",
            "precision_score_avg",
            "precision_score_std",
            "recall_score_min",
            "recall_score_max",
            "recall_score_avg",
            "recall_score_std",
            "f1_score_min",
            "f1_score_max",
            "f1_score_avg",
            "f1_score_std",
            "n_thresholds_min",
            "n_thresholds_max",
        ]

    # remove any columns names from primary_cols that are not in df
    primary_cols = [col for col in primary_cols if col in df.columns]

    # get secondary columns, which are all the remaining columns from the df
    secondary_cols = [col for col in df.columns if col not in primary_cols]

    # return the df with the primary columns first, then the secondary columns
    return df[primary_cols + secondary_cols]


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
            "early_stopping_rounds",
        ]

    # remove any keys from general_param_keys that are not in df
    general_param_keys = [col for col in general_param_keys if col in df.columns]

    return {k: [df.iloc[row_idx][k]] for k in general_param_keys}


def main(args):

    proj_dir, path_data_dir, path_emb_dir, path_final_dir = set_directories(args)

    df = pd.read_csv(
        path_final_dir / args.compiled_csv_name,
    )
    df = filter_results_df(df)

    # df = df[:500]

    # use this is you want to only select the top models by model type (e.g. top SVM, RF, etc.)
    sort_by = 'prauc_avg'
    # sort_by = 'prauc_min'
    df = df.groupby(['classifier']).head(args.keep_top_n).sort_values(by=sort_by, ascending=False)

    df = order_columns_on_results_df(df)

    # save the top models to a csv
    df.to_csv(path_final_dir / args.filtered_csv_name, index=False)

    # save a certain number of PR-AUC and ROC-AUC curves
    if args.save_n_figures > 0:
        print(f"Using embedding file: {args.emb_file_name}")

        with open(path_emb_dir / args.emb_file_name, "rb") as f:
            df_feat = pickle.load(f)

        percent_true = df_feat["label"].sum() / len(df_feat)
        print(f"Percent imbalance: {percent_true:.2%}")

        path_model_curves = path_final_dir / "model_curves"
        Path(path_model_curves).mkdir(parents=True, exist_ok=True)

        if args.save_models == "True":
            model_save_path = path_final_dir / "model_files"
            model_save_path.mkdir(parents=True, exist_ok=True)
            save_models = True
        else:
            save_models = False
            model_save_path = None

        for row_idx in range(args.save_n_figures):

            params_clf = rebuild_params_clf(df, row_idx)
            general_params = rebuild_general_params(df, row_idx)

            print("Classifier params:\n")
            print(params_clf)
            print("General params:\n")
            print(general_params)

            META_LABEL_COLS = ["para"]
            STRATIFICATION_GROUPING_COL = (
                "id"  # either arxiv paper id, or another unique id (such as the doi)
            )
            Y_LABEL_COL = "label"
            sampler_seed = int(df.iloc[row_idx]["sampler_seed"])
            id = df.iloc[row_idx]["id"]  # unique id for the specific model run

            # copy the random_search_setup.py file to path_final_dir using shutil if it doesn't exist
            Path(path_final_dir / "setup_files").mkdir(parents=True, exist_ok=True)
            if (path_final_dir / "setup_files" / "models_classical" / "random_search_setup.py").exists():
                pass
            else:
                shutil.copytree(
                    proj_dir / "src" / "models_classical",
                    path_final_dir / "setup_files" / "models_classical",
                )
                shutil.copy(
                    proj_dir / "Makefile",
                    path_final_dir / "setup_files" / "Makefile",
                )

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
        "--path_emb_dir",
        type=str,
        help="Path to the folder that contains all the embedding pickle files",
    )

    parser.add_argument(
        "--emb_file_name",
        type=str,
        default="df_embeddings.pkl",
        help="Name of the embedding file to save",
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
