"""
Utility functions used in the random search training.
"""

from sklearn.model_selection import ParameterSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

from src.models_classical.random_search_setup import (
    rf_params,
    xgb_params,
    knn_params,
    lr_params,
    sgd_params,
    ridge_params,
    svm_params,
    nb_params,
)

from src.models_classical.classifiers import (
    rf_classifier,
    xgb_classifier,
    knn_classifier,
    lr_classifier,
    sgd_classifier,
    ridge_classifier,
    svm_classifier,
    nb_classifier,
)

from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    accuracy_score,
)

def under_over_sampler(x, y, method=None, ratio=0.5):
    """
    Returns an undersampled or oversampled data set. Implemented using imbalanced-learn package.
    ['random_over','random_under','random_under_bootstrap','smote', 'adasyn']
    """
    print("method: ", method)

    # if method == None:
    #     return x, y

    # oversample methods: https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
    if method == "random_over":
        # print('before:',sorted(Counter(y).items()))
        ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
        x_resampled, y_resampled = ros.fit_resample(x, y)
        # print('after:',sorted(Counter(y_resampled).items()))
        return x_resampled, y_resampled

    elif method == "random_under":
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
        x_resampled, y_resampled = rus.fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "random_under_bootstrap":
        rus = RandomUnderSampler(
            sampling_strategy=ratio, random_state=0, replacement=True
        )
        x_resampled, y_resampled = rus.fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "smote":
        x_resampled, y_resampled = SMOTE(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "borderline_smote":
        x_resampled, y_resampled = BorderlineSMOTE(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "kmeans_smote":
        x_resampled, y_resampled = KMeansSMOTE(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "svm_smote":
        x_resampled, y_resampled = SVMSMOTE(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "smote_enn":
        x_resampled, y_resampled = SMOTEENN(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "smote_tomek":
        x_resampled, y_resampled = SMOTETomek(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    elif method == "adasyn":
        x_resampled, y_resampled = ADASYN(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(x, y)
        return x_resampled, y_resampled

    else:
        return x, y


def get_classifier_and_params(classifier_string):
    if classifier_string == "rf":
        return rf_classifier, rf_params

    elif classifier_string == "xgb":
        return xgb_classifier, xgb_params

    elif classifier_string == "knn":
        return knn_classifier, knn_params

    elif classifier_string == "lr":
        return lr_classifier, lr_params

    elif classifier_string == "sgd":
        return sgd_classifier, sgd_params

    elif classifier_string == "ridge":
        return ridge_classifier, ridge_params

    elif classifier_string == "svm":
        return svm_classifier, svm_params

    elif classifier_string == "nb":
        return nb_classifier, nb_params

    else:
        raise ValueError("Classifier string not recognized")


def collate_scores_binary_classification(scores_list):
    """
    Collate the scores from the k-fold cross-validation
    """

    n_thresholds_list = []
    precisions_list = []
    recalls_list = []
    precision_score_list = []
    recall_score_list = []
    fpr_list = []
    tpr_list = []
    prauc_list = []
    rocauc_list = []
    f1_list = []
    accuracy_list = []

    for ind_score_dict in scores_list:
            n_thresholds_list.append(ind_score_dict["n_thresholds"])
            precisions_list.append(ind_score_dict["precisions"])
            recalls_list.append(ind_score_dict["recalls"])
            precision_score_list.append(ind_score_dict["precision_result"])
            recall_score_list.append(ind_score_dict["recall_result"])
            fpr_list.append(ind_score_dict["fpr"])
            tpr_list.append(ind_score_dict["tpr"])
            prauc_list.append(ind_score_dict["prauc_result"])
            rocauc_list.append(ind_score_dict["rocauc_result"])
            f1_list.append(ind_score_dict["f1_result"])
            accuracy_list.append(ind_score_dict["accuracy_result"])

    result_dict = {
        "precisions_array": np.array(precisions_list, dtype=object),
        "recalls_array": np.array(recalls_list, dtype=object),
        "precision_score_array": np.array(precision_score_list, dtype=object),
        "recall_score_array": np.array(recall_score_list, dtype=object),
        "fpr_array": np.array(fpr_list, dtype=object),
        "tpr_array": np.array(tpr_list, dtype=object),
        "prauc_array": np.array(prauc_list, dtype=object),
        "rocauc_array": np.array(rocauc_list, dtype=object),
        "f1_score_array": np.array(f1_list, dtype=object),
        "n_thresholds_array": np.array(n_thresholds_list, dtype=int),
        "accuracy_array": np.array(accuracy_list, dtype=object),
    }

    return result_dict



def calculate_scores(clf, x_test, y_test,):
    """Helper function for calculating a bunch of scores"""

    y_pred = clf.predict(x_test)

    # need decision function or probability
    # should probably remove the try-except at a later date
    try:
        y_scores = clf.decision_function(x_test)
    except:
        y_scores = clf.predict_proba(x_test)[:, 1]

    n_thresholds = len(np.unique(y_scores))

    n_correct = sum(y_pred == y_test)

    # calculate the percent accuracy
    accuracy_result = accuracy_score(y_test, y_pred)
    
    # need to use decision scores, or probabilities, in roc_score
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_scores)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)

    # calculate the precision recall curve and roc_auc curve
    # when to use ROC vs. precision-recall curves, Jason Brownlee http://bit.ly/38vEgnW
    # https://stats.stackexchange.com/questions/113326/what-is-a-good-auc-for-a-precision-recall-curve
    prauc_result = auc(recalls, precisions)
    rocauc_result = roc_auc_score(y_test, y_scores)

    # calculate precision, recall, f1 scores
    precision_result = precision_score(y_test, y_pred)
    recall_result = recall_score(y_test, y_pred)
    f1_result = f1_score(y_test, y_pred)

    # create a dictionary of all the scores
    scores = {"n_correct": n_correct, "n_thresholds": n_thresholds, "prauc_result": prauc_result, "rocauc_result": rocauc_result,
                "precision_result": precision_result, "recall_result": recall_result, "f1_result": f1_result, 
                "precisions": precisions, "recalls": recalls, "pr_thresholds": pr_thresholds,
                "fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thresholds, "y_scores": y_scores, "accuracy_result": accuracy_result}


    return scores


def scale_data(x_train, x_test, scaler_method=None):
    if scaler_method == "standard":
        print("scaling - standard")
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    elif scaler_method == "minmax":
        print("scaling - min/max")
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    else:
        print("no scaling")
        scaler = None
        pass
    return x_train, x_test, scaler


def get_model_metrics_df(model_metrics_dict):

    selected_metrics_dict = {
        'precision_score_min': np.min(model_metrics_dict['precision_score_array']),
        'precision_score_max': np.max(model_metrics_dict['precision_score_array']),
        'precision_score_avg': np.mean(model_metrics_dict['precision_score_array']),
        'precision_score_std': np.std(model_metrics_dict['precision_score_array']),
        'recall_score_min': np.min(model_metrics_dict['recall_score_array']),
        'recall_score_max': np.max(model_metrics_dict['recall_score_array']),
        'recall_score_avg': np.mean(model_metrics_dict['recall_score_array']),
        'recall_score_std': np.std(model_metrics_dict['recall_score_array']),
        'f1_score_min': np.min(model_metrics_dict['f1_score_array']),
        'f1_score_max': np.max(model_metrics_dict['f1_score_array']),
        'f1_score_avg': np.mean(model_metrics_dict['f1_score_array']),
        'f1_score_std': np.std(model_metrics_dict['f1_score_array']),
        'rocauc_min': np.min(model_metrics_dict['rocauc_array']),
        'rocauc_max': np.max(model_metrics_dict['rocauc_array']),
        'rocauc_avg': np.mean(model_metrics_dict['rocauc_array']),
        'rocauc_std': np.std(model_metrics_dict['rocauc_array']),
        'prauc_min': np.min(model_metrics_dict['prauc_array']),
        'prauc_max': np.max(model_metrics_dict['prauc_array']),
        'prauc_avg': np.mean(model_metrics_dict['prauc_array']),
        'prauc_std': np.std(model_metrics_dict['prauc_array']),
        'accuracy_min': np.min(model_metrics_dict['accuracy_array']),
        'accuracy_max': np.max(model_metrics_dict['accuracy_array']),
        'accuracy_avg': np.mean(model_metrics_dict['accuracy_array']),
        'accuracy_std': np.std(model_metrics_dict['accuracy_array']),
        'n_thresholds_min': np.min(model_metrics_dict['n_thresholds_array']),
        'n_thresholds_max': np.max(model_metrics_dict['n_thresholds_array']),
    }

    df_m = pd.DataFrame.from_dict(selected_metrics_dict, orient="index").T

    return df_m
    