import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from pathlib import Path
import random
import argparse
import logging
import shutil
import pickle
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from src.models_classical.utils import (
    under_over_sampler,
    scale_data,
    calculate_scores,
    collate_scores_binary_classification,
    get_classifier_and_params,
    get_model_metrics_df,
)
from src.models_classical.random_search_setup import general_params

from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

# from src.models_classical.random_search_setup import (
#     rf_params,
#     xgb_params,
#     knn_params,
#     lr_params,
#     sgd_params,
#     ridge_params,
#     svm_params,
#     nb_params,
# )

from src.visualization.visualize import plot_pr_roc_curves_kfolds





def kfold_cv(
    df,
    clf,
    oversamp_method,
    undersamp_method,
    scaler_method,
    oversamp_ratio,
    undersamp_ratio,
    meta_label_cols,
    stratification_grouping_col=None,
    y_label_col="y",
    n_splits=5,
):

    scores_list = []

    # perform stratified k-fold cross validation using the grouping of the y-label and another column
    if (
        stratification_grouping_col is not None
        and stratification_grouping_col is not y_label_col
    ):
        df_strat = df[[stratification_grouping_col, y_label_col]].drop_duplicates()

        skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # use clone to do a deep copy of model without copying attached data
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html

        for i, (train_index, test_index) in enumerate(
            skfolds.split(
                df_strat[[stratification_grouping_col]], df_strat[[y_label_col]]
            )
        ):
            clone_clf = clone(clf)
            train_strat_vals = df_strat.iloc[train_index][
                stratification_grouping_col
            ].values

            test_strat_vals = df_strat.iloc[test_index][
                stratification_grouping_col
            ].values

            # train
            df_train = df[df[stratification_grouping_col].isin(train_strat_vals)]
            y_train = df_train[y_label_col].values.astype(int)
            df_train = df_train.drop(meta_label_cols + [y_label_col], axis=1)
            x_train_cols = df_train.columns
            x_train = np.array([e for e in df_train["h"].values])

            # test
            df_test = df[df[stratification_grouping_col].isin(test_strat_vals)]
            y_test = df_test[y_label_col].values.astype(int)
            df_test = df_test.drop(meta_label_cols + [y_label_col], axis=1)
            x_test_cols = df_test.columns
            x_test = np.array([e for e in df_test["h"].values])

            # over-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=oversamp_method, ratio=oversamp_ratio
            )

            # under-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=undersamp_method, ratio=undersamp_ratio
            )

            # scale the data
            x_train, x_test, scaler = scale_data(x_train, x_test, scaler_method)


            # train model
            print("x_train shape:", x_train.shape)
            clone_clf.fit(x_train, y_train)

            # calculate the scores for each individual model train in the cross validation
            # save as a dictionary: "ind_score_dict"
            ind_score_dict = calculate_scores(clone_clf, x_test, y_test)
            scores_list.append(ind_score_dict)

    # perform stratified k-fold cross if only using the y-label for stratification
    else:
        skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(
            skfolds.split(df, df[[y_label_col]])
        ):
            clone_clf = clone(clf)

            df_train = df.iloc[train_index]
            df_test = df.iloc[test_index]

            y_train = df_train[y_label_col].values.astype(int)
            df_train = df_train.drop(meta_label_cols + [y_label_col], axis=1)
            x_train_cols = df_train.columns
            x_train = np.array([e for e in df_train["h"].values])

            y_test = df_test[y_label_col].values.astype(int)
            df_test = df_test.drop(meta_label_cols + [y_label_col], axis=1)
            x_test_cols = df_test.columns
            x_test = np.array([e for e in df_test["h"].values])

            # over-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=oversamp_method, ratio=oversamp_ratio
            )

            # under-sample the data
            x_train, y_train = under_over_sampler(
                x_train, y_train, method=undersamp_method, ratio=undersamp_ratio
            )

            # scale the data
            x_train, x_test, scaler = scale_data(x_train, x_test, scaler_method)

            # train model
            clone_clf.fit(x_train, y_train)

            # calculate the scores for each k in the cross validation
            # save as a dictionary: "ind_score_dict"
            ind_score_dict = calculate_scores(clone_clf, x_test, y_test)
            scores_list.append(ind_score_dict)

    trained_result_dict = collate_scores_binary_classification(scores_list)

    return trained_result_dict, scaler, clone_clf


# TO-DO: need to add the general_params dictionary to the functions.
def train_single_model(
    df,
    sampler_seed,
    meta_label_cols,
    stratification_grouping_col=None,
    y_label_col="y",
    general_params=None,
    params_clf=None,
    save_model=False,
    model_save_name=None,
    model_save_path=None,
    now_str=None, # for saving the model - the current time
    dataset_name="dataset", # for saving the model - the dataset name

):
    # generate the list of parameters to sample over
    params_dict_train_setup = list(
        ParameterSampler(general_params, n_iter=1, random_state=sampler_seed)
    )[0]

    oversamp_method = params_dict_train_setup["oversamp_method"]
    undersamp_method = params_dict_train_setup["undersamp_method"]
    scaler_method = params_dict_train_setup["scaler_method"]
    oversamp_ratio = params_dict_train_setup["oversamp_ratio"]
    undersamp_ratio = params_dict_train_setup["undersamp_ratio"]
    classifier = params_dict_train_setup["classifier"]

    print(
        f"classifier: {classifier}, oversamp_method: {oversamp_method}, undersamp_method: {undersamp_method}"
    )

    # get classifier and its parameters
    clf_function, params_clf_generated = get_classifier_and_params(classifier)

    if params_clf is None:
        params_clf = params_clf_generated

    # instantiate the model
    clf, param_dict_clf_raw, params_dict_clf_named = clf_function(
        sampler_seed, params_clf
    )
    print("\n", params_dict_clf_named)

    model_metrics_dict, scaler_fitted, clf_trained = kfold_cv(
        df,
        clf,
        oversamp_method,
        undersamp_method,
        scaler_method,
        oversamp_ratio,
        undersamp_ratio,
        meta_label_cols,
        stratification_grouping_col,
        y_label_col,
        n_splits=5,
    )

    # added additional parameters to the training setup dictionary
    params_dict_train_setup["sampler_seed"] = sampler_seed
    # params_dict_train_setup["classifier"]

    # save the model if requested
    if save_model:
        if model_save_name is None:
            model_save_name = f"model_{sampler_seed}_{classifier}_{now_str}_{dataset_name}.pkl"
            scaler_save_name = f"scaler_{sampler_seed}_{classifier}_{now_str}_{dataset_name}.pkl"
        else:
            scaler_save_name = "scaler_" + model_save_name
            model_save_name = "model_" + model_save_name

        # save the model and scaler with pickle
        with open(model_save_path / model_save_name, "wb") as f:
            pickle.dump(clf_trained, f)

        with open(model_save_path / scaler_save_name, "wb") as f:
            pickle.dump(scaler_fitted, f)


    return model_metrics_dict, params_dict_clf_named, params_dict_train_setup


def random_search_runner(
    df,
    rand_search_iter,
    meta_label_cols,
    stratification_grouping_col,
    proj_dir,
    path_save_dir,
    dataset_name=None,
    y_label_col="y",
    save_freq=1,
    debug=False,
):

    results_list = []
    for i in range(rand_search_iter):
        # set random sample seed
        sample_seed = random.randint(0, 2**25)
        # sample_seed = 13

        if i == 0:
            file_name_results = f"results_{sample_seed}.csv"

            # copy the random_search_setup.py file to path_save_dir using shutil if it doesn't exist
            if (path_save_dir / "setup_files" / "random_search_setup.py").exists():
                pass
            else:
                shutil.copy(
                    proj_dir / "src/models_classical/random_search_setup.py",
                    path_save_dir / "setup_files" / "random_search_setup.py",
                )

        try:

            if args.date_time:
                now_str = str(args.date_time)
            else:
                now = datetime.now()
                now_str = now.strftime("%Y-%m-%d-%H%M-%S")

            (
                model_metrics_dict,
                params_dict_clf_named,
                params_dict_train_setup,
            ) = train_single_model(
                df,
                sample_seed,
                meta_label_cols,
                stratification_grouping_col,
                y_label_col,
                general_params=general_params,
                params_clf=None,
                save_model=False,
                model_save_name=None,
                model_save_path=path_save_dir,
                now_str=now_str, # for saving the model - the current time
                dataset_name=dataset_name,
            )

            # train setup params
            df_t = pd.DataFrame.from_dict(params_dict_train_setup, orient="index").T

            df_t["date_time"] = now_str
            df_t["dataset"] = dataset_name
            classifier_used = params_dict_train_setup["classifier"]
            df_t["id"] = f"{sample_seed}_{classifier_used}_{now_str}_{dataset_name}"

            # classifier params
            df_c = pd.DataFrame.from_dict(params_dict_clf_named, orient="index").T

            # model metric results
            df_m = get_model_metrics_df(model_metrics_dict)

            results_list.append(pd.concat([df_t, df_m, df_c], axis=1))

            if i % save_freq == 0:
                df_results = pd.concat(results_list)

                if path_save_dir is not None:
                    df_results.to_csv(path_save_dir / file_name_results, index=False)
                else:
                    df_results.to_csv(file_name_results, index=False)

        # except Exception as e and log the exception
        except Exception as e:
            if debug:
                print("####### Exception #######")
                print(e)
                logging.exception(f"##### Exception in random_search_runner:\n{e}\n\n")
            pass


def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    save_dir_name = args.save_dir_name

    # check if "scratch" path exists in the home directory
    # if it does, assume we are on HPC
    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        print("Assume on HPC")

        path_save_dir = scratch_path / "arxiv-code-search/models" / args.save_dir_name
        Path(path_save_dir / "setup_files").mkdir(parents=True, exist_ok=True)

    else:
        print("Assume on local compute")
        path_save_dir = proj_dir / "models" / save_dir_name
        Path(path_save_dir / "setup_files").mkdir(parents=True, exist_ok=True)

    return proj_dir, path_data_dir, path_save_dir


def main(args):

    # set directories
    proj_dir, path_data_dir, path_save_dir = set_directories(args)

    embedding_dir = path_data_dir / "processed/embeddings"

    RAND_SEARCH_ITER = args.rand_search_iter

    # set a seed for the parameter sampler
    # SAMPLER_SEED = random.randint(0, 2 ** 16)

    # load dfh.pickle
    with open(embedding_dir / "df_embeddings.pkl", "rb") as f:
        df = pickle.load(f)

    Y_LABEL_COL = "label"

    # identify if there is another column you want to
    # stratify on, besides the y label
    STRATIFICATION_GROUPING_COL = "id"
    # STRATIFICATION_GROUPING_COL = None

    # list of the columns that are not features columns
    # (not including the y-label column)
    META_LABEL_COLS = ["para"]

    LOG_FILENAME = path_save_dir / "logging_example.out"
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    random_search_runner(
        df,
        RAND_SEARCH_ITER,
        META_LABEL_COLS,
        STRATIFICATION_GROUPING_COL,
        proj_dir,
        path_save_dir,
        dataset_name="papers1",
        y_label_col=Y_LABEL_COL,
        save_freq=1,
        debug=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Random search through various models and parameters"
    )

    parser.add_argument(
        "--n_cores",
        type=int,
        default=1,
        help="Number of cores to use for multiprocessing",
    )

    parser.add_argument(
        "--rand_search_iter",
        type=int,
        default=2,
        help="Number number of randem search iterations",
    )

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--path_data_dir",
        dest="path_data_dir",
        type=str,
        help="Location of the data folder, containing the raw, interim, and processed folders",
    )

    parser.add_argument(
        "--save_dir_name",
        default="interim_results",
        type=str,
        help="Name of the save directory. Used to store the results of the random search",
    )

    parser.add_argument(
        "--date_time",
        type=str,
        help="Date and time that random search was executed.",
    )

    args = parser.parse_args()

    main(args)
