import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from numpy.lib.stride_tricks import sliding_window_view
import math


def segment_line_by_sliding_window(x, y, x_axis_n=1000):

    y_adj = []  # y values, adjusted
    x_adj = []  # x values, adjusted

    # break the x-axis up by moving a sliding window along it
    for j, i in enumerate(sliding_window_view(np.linspace(0, 1, x_axis_n), 2)):
        window_segment = np.where((x >= i[0]) & (x <= i[1]))[0]

        # if there are no values in the window, then
        # use the last precision value in the window
        if len(window_segment) == 0:
            x_adj.append(np.mean(i))
            y_adj.append(y_adj[-1])
        else:
            x_adj.append(np.mean(i))
            y_adj.append(np.mean(y[window_segment]))

    return np.array(x_adj), np.array(y_adj)


def plot_pr_roc_curves_kfolds(
    precision_array,
    recall_array,
    fpr_array,
    tpr_array,
    rocauc_array,
    prauc_array,
    percent_anomalies_truth=0.073,
    path_save_name="model_curves.pdf",
    save_plot=False,
    dpi=300,
):
    """
    Plot the precision-recall curves and the ROC curves for the different k-folds used in
    cross-validation. Also show the average PR and ROC curves.

    :param precision_array: array of precision values for each k-fold
    :param recall_array: array of recall values for each k-fold
    :param fpr_array: array of false positive rate values for each k-fold
    :param tpr_array: array of true positive rate values for each k-fold
    :param rocauc_array: array of ROC AUC values for each k-fold
    :param prauc_array: array of PR AUC values for each k-fold
    :param percent_anomalies_truth: the percentage of anomalies in the dataset

    """

    # sns whitegrid context
    sns.set(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, dpi=150)
    fig.tight_layout(pad=5.0)

    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)

    ######
    # plot the precision recall curves
    precisions_all_segmented = []
    for p, r in zip(precision_array, recall_array):
        r_adj, p_adj = segment_line_by_sliding_window(r, p, x_axis_n=10000)
        precisions_all_segmented.append(p_adj)

    precisions_all_segmented = np.array(precisions_all_segmented)

    for i, (p, r) in enumerate(zip(precision_array, recall_array)):
        if i == np.shape(precision_array)[0] - 1:
            axes[0].plot(
                r[:], p[:], label="k-fold model", color="gray", alpha=0.5, linewidth=1
            )
        else:
            axes[0].plot(r[:], p[:], color="grey", alpha=0.5, linewidth=1)

    axes[0].plot(
        r_adj,
        precisions_all_segmented.mean(axis=0),
        label="Average model",
        color=pal[5],
        linewidth=2,
    )

    axes[0].plot(
        np.array([0, 1]),
        np.array([percent_anomalies_truth, percent_anomalies_truth]),
        marker="",
        linestyle="--",
        label="No skill model",
        color="orange",
        linewidth=2,
        zorder=0,
    )

    axes[0].legend()
    axes[0].title.set_text("Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].text(
        x=-0.05,
        y=-0.3,
        s=f"PR AUC: {prauc_array.mean():.3f} (avg), {prauc_array.min():.3f} (min), {prauc_array.max():.3f} (max)",
        # s=f"PR AUC: {math.floor(prauc_array.mean() * 1000)/1000.0:.3f} (avg), {math.floor(prauc_array.min() * 1000)/1000.0:.3f} (min), {math.floor(prauc_array.max() * 1000)/1000.0:.3f} (max)",
        # fontsize=10,
        horizontalalignment="left",
        verticalalignment="center",
        rotation="horizontal",
        alpha=1,
    )

    ######
    # plot the ROC curves
    roc_all_segmented = []
    for t, f in zip(tpr_array, fpr_array):
        f_adj, t_adj = segment_line_by_sliding_window(f, t, x_axis_n=10000)
        roc_all_segmented.append(t_adj)

    roc_all_segmented = np.array(roc_all_segmented)

    for i, (t, f) in enumerate(zip(tpr_array, fpr_array)):
        if i == np.shape(tpr_array)[0] - 1:
            axes[1].plot(
                f[:], t[:], label="k-fold models", color="gray", alpha=0.5, linewidth=1
            )
        else:
            axes[1].plot(f[:], t[:], color="grey", alpha=0.5, linewidth=1)

    axes[1].plot(
        f_adj,
        roc_all_segmented.mean(axis=0),
        label="Average of k-folds",
        color=pal[5],
        linewidth=2,
    )

    axes[1].plot(
        np.array([0, 1]),
        np.array([0, 1]),
        marker="",
        linestyle="--",
        label="No skill",
        color="orange",
        linewidth=2,
        zorder=0,
    )

    axes[1].title.set_text("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].text(
        x=-0.05,
        y=-0.3,
        s=f"ROC AUC: {rocauc_array.mean():.3f} (avg), {rocauc_array.min():.3f} (min), {rocauc_array.max():.3f} (max)",
        # s=f"ROC AUC: {math.floor(rocauc_array.mean() * 1000)/1000.0:.3f} (avg), {math.floor(rocauc_array.min() * 1000)/1000.0:.3f} (min), {math.floor(rocauc_array.max() * 1000)/1000.0:.3f} (max)",
        # fontsize=10,
        horizontalalignment="left",
        verticalalignment="center",
        rotation="horizontal",
        alpha=1,
    )

    for ax in axes.flatten():
        ax.yaxis.set_tick_params(labelleft=True, which="major")
        ax.grid(False)

    if save_plot:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
        plt.cla()
        plt.close()
    else:
        plt.show()


# use numpy doc string formatting
def summarize_final_label_file(file_path, sample_size, publisher_name):
    """
    Function to summarize the final label file.

    Parameters
    ----------
    file_path : str
        Path to the final label file.
    sample_size : int
        Number of articles in the sample.
    publisher_name : str
        Name of the publisher. Use nice formatting for eventual plotting.

    Returns
    -------
    df_label_count : pandas.DataFrame
        Dataframe with the summary of the final label file.
    df_label_count_all : pandas.DataFrame
        Dataframe with the summary of the final label file.


    """

    file_name = file_path.stem

    if file_name.endswith(".csv"):
        df = pd.read_csv(file_path, parse_dates=["update_date"])
    else:
        df = pd.read_excel(
            file_path,
            parse_dates=["update_date"],
            engine="odf",
        )
        df = df[["id", "pattern", "token_count", "update_date", "label", "para"]]

    # group id and aggregate by max label
    dfr = df.groupby(["id"]).agg({"label": "max"}).reset_index()

    # replace any NaN in "label" column with 0
    dfr.label.fillna(0, inplace=True)

    # the number of articles may have been more than what is contained in the file (because no keywords may have been found)
    sample_size_df = len(dfr)

    df_label_count = dfr.groupby("label").count().reset_index().astype(int)
    df_label_count = df_label_count.iloc[:, :2]
    df_label_count.columns = ["label", "count"]

    # the difference between sample_size and sample_size_df is the number of articles that had no keywords in the search
    df_label_count.loc[0, "count"] += sample_size - sample_size_df

    missing_category = set(range(4)).difference(df_label_count.label.unique())

    for category in missing_category:
        df_label_count.loc[len(df_label_count)] = [category, 0]

    df_label_count_all = df_label_count.copy()

    def label_name(x):
        if x == 0:
            return "Data and Code Not\nPublicly Available"
        elif x == 1:
            return "Only Data\nPublicly Available"
        elif x == 2:
            return "Only Code\nPublicly Available"
        elif x == 3:
            return "Both Data and Code\nPublicly Available"
        else:
            return "Data and Code\nNot Available"

    df_label_count_all["label_name"] = df_label_count_all.label.apply(label_name)
    df_label_count_all["percentage"] = (
        df_label_count_all["count"] / df_label_count_all["count"].sum() * 100
    )
    df_label_count_all = df_label_count_all[
        ["label_name", "label", "count", "percentage"]
    ].sort_values(by="label")
    df_label_count_all["publisher"] = publisher_name
    df_label_count_all["sample_size"] = sample_size

    # create new column called "label_name" where the value is 0 if the label is 0, 1 if the label is greater than 0
    df_label_count["label_name"] = df_label_count.label.apply(
        lambda x: "Data and Code Not Available"
        if x == 0
        else "Data or Code Publicly Available"
    )
    df_label_count["label"] = df_label_count.label.apply(lambda x: 0 if x == 0 else 1)
    df_label_count = (
        df_label_count.groupby(["label_name", "label"])
        .agg({"count": "sum"})
        .reset_index()
    )
    df_label_count["percentage"] = (
        df_label_count["count"] / df_label_count["count"].sum() * 100
    )
    df_label_count["publisher"] = publisher_name
    df_label_count["sample_size"] = sample_size

    return df_label_count, df_label_count_all


def plot_percent_articles_by_publisher(
    df,
    title="Data and Code Availablility for Articles\nat Various Venues (from 2015-2021)",
    path_save_dir=None,
    save_name="article_pcts_by_publisher",
    dpi=300,
    save_plot=False,
):
    df["publisher_size"] = df.publisher + "\n(n=" + df.sample_size.astype(str) + ")"
    df = df[df["label"] == 1].sort_values(by="percentage", ascending=False)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(4, 7),
    )

    sns.set(style="whitegrid", font="DejaVu Sans")
    pal = sns.color_palette("Greys_r")
    font_size = 14

    # define color palette
    custom_pal = []
    for publisher in df["publisher"]:
        if publisher == "PHM Conf.":
            # this palette https://www.color-hex.com/color-palette/94510
            custom_pal.append("#bd0c0c")
        else:
            custom_pal.append(pal[2])

    ax = sns.barplot(
        x="percentage",
        y="publisher_size",
        data=df,
        palette=custom_pal,
    )

    for p in ax.patches:
        # help from https://stackoverflow.com/a/56780852/9214620
        space = df["percentage"].max() * 0.02
        _x = p.get_x() + p.get_width() + float(space)
        _y = p.get_y() + p.get_height() / 1.93
        value = p.get_width()

        ax.text(
            _x,
            _y,
            f"{int(np.round(value))}%",
            ha="left",
            va="center",
            weight="normal",
            size=font_size,
        )

    ax.spines["bottom"].set_visible(True)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.grid(alpha=0.7, linewidth=1, axis="x")
    ax.set_xticks([0])
    ax.set_xticklabels([])

    ax.text(
        0.35,
        1.10,
        title,
        verticalalignment="top",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontsize=14,
    )

    # ax.set_title(title, fontsize=font_size, loc='right', pad=20, x=-0.25)
    ax.tick_params(axis="y", labelsize=font_size)
    plt.subplots_adjust(wspace=0.3)
    sns.despine(left=True, bottom=True)

    # save plot as pdf and png
    if save_plot:
        # matplotlib.use('Agg')
        if path_save_dir is None:
            path_save_dir = Path.cwd().parent.parent

        plt.savefig(
            path_save_dir / f"{save_name}.pdf",
            bbox_inches="tight",
        )

        plt.savefig(path_save_dir / f"{save_name}.png", bbox_inches="tight", dpi=dpi)
        plt.cla()
        plt.close()
    else:
        plt.show()


def plot_individual_publisher(
    df,
    publisher_name="PHM Conf.",
    title="Data and Code Availablility for Articles\nat the PHM Conf. (from 2015-2021)",
    path_save_dir=None,
    save_name="article_pcts_phm_conf",
    bar_color=None,
    dpi=300,
    save_plot=False,
):
    df = df[df["publisher"] == publisher_name]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(4, 7),
    )

    sns.set(style="whitegrid", font="DejaVu Sans")
    pal = sns.color_palette("Greys_r")
    font_size = 14

    if bar_color is None:
        bar_color = pal[2]

    ax = sns.barplot(
        x="percentage",
        y="label_name",
        data=df,
        color=bar_color,
    )

    for p in ax.patches:
        # help from https://stackoverflow.com/a/56780852/9214620
        space = df["percentage"].max() * 0.04
        _x = p.get_x() + p.get_width() + float(space)
        _y = p.get_y() + p.get_height() / 1.93
        value = p.get_width()

        ax.text(
            _x,
            _y,
            f"{int(np.round(value))}%",
            ha="left",
            va="center",
            weight="normal",
            size=font_size,
        )

    ax.spines["bottom"].set_visible(True)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.grid(alpha=0.7, linewidth=1, axis="x")
    ax.set_xticks([0])
    ax.set_xticklabels([])

    ax.text(
        0.2,
        1.10,
        title,
        verticalalignment="top",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontsize=14,
    )

    # ax.set_title(title, fontsize=font_size, loc="center", pad=20, x=-0.05)
    ax.tick_params(axis="y", labelsize=font_size)
    plt.subplots_adjust(wspace=0.3)
    sns.despine(left=True, bottom=True)

    # save plot as pdf and png
    if save_plot:
        # matplotlib.use('Agg')
        if path_save_dir is None:
            path_save_dir = Path.cwd().parent.parent

        plt.savefig(
            path_save_dir / f"{save_name}.pdf",
            bbox_inches="tight",
        )

        plt.savefig(path_save_dir / f"{save_name}.png", bbox_inches="tight", dpi=dpi)
        plt.cla()
        plt.close()
    else:
        plt.show()


def main():
    logger = logging.getLogger(__name__)
    logger.info("making figures from label data")

    path_label_dir = proj_dir / "data/processed/labels/labels_complete_for_viz"
    path_save_dir = proj_dir / "reports/figures/"
    path_save_dir.mkdir(parents=True, exist_ok=True)

    # make sure the styling of the plots is consistent
    sns.set(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    # prep label data
    file_names_list = [
        "labels_phm_97_0-150.ods",
        "labels_energies_98_0-150.ods",
        "labels_mssp_99_0-100.ods",
        "labels_1.ods",
    ]
    publisher_names_list = ["PHM Conf.", "Energies", "MSSP", "arXiv"]
    sample_size_list = [150, 124, 100, 100]

    df_all_list = []
    df_list = []
    for file_name, publisher, sample_size in zip(
        file_names_list, publisher_names_list, sample_size_list
    ):
        df, df_all = summarize_final_label_file(
            path_label_dir / file_name, sample_size, publisher
        )
        df_list.append(df)
        df_all_list.append(df_all)

    df_all = pd.concat(df_all_list)
    df = pd.concat(df_list)

    # All publications
    plot_percent_articles_by_publisher(
        df,
        title="Data and Code Availablility for Different\nPublications (articles from 2015-2021)",
        path_save_dir=path_save_dir,
        save_name="article_pcts_by_publisher",
        dpi=300,
        save_plot=True,
    )

    # PHM Conf.
    plot_individual_publisher(
        df_all,
        publisher_name="PHM Conf.",
        title="PHM Conf. Data and Code Availablility\n(articles from 2015-2021)",
        path_save_dir=path_save_dir,
        save_name="article_pcts_phm_conf",
        bar_color="#bd0c0c",  # red-ish color
        dpi=300,
        save_plot=True,
    )

    # MSSP
    plot_individual_publisher(
        df_all,
        publisher_name="MSSP",
        title="MSSP Data and Code Availablility\n(articles from 2015-2021)",
        path_save_dir=path_save_dir,
        save_name="article_pcts_mssp",
        dpi=300,
        save_plot=True,
    )

    # Energies
    plot_individual_publisher(
        df_all,
        publisher_name="Energies",
        title="Energies Data and Code Availablility\n(articles from 2015-2021)",
        path_save_dir=path_save_dir,
        save_name="article_pcts_energies",
        dpi=300,
        save_plot=True,
    )

    # arXiv
    plot_individual_publisher(
        df_all,
        publisher_name="arXiv",
        title="arXiv Data and Code Availablility\n(articles from 2015-2021)",
        path_save_dir=path_save_dir,
        save_name="article_pcts_arxiv",
        dpi=300,
        save_plot=True,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    proj_dir = Path(__file__).resolve().parents[2]

    main()
