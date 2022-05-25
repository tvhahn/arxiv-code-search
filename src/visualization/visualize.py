import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def summarize_final_label_file(file_path, sample_size, publisher_name):

    # get file_name from file_path
    file_name = file_path.stem

    if file_name.endswith(".csv"):
        df = pd.read_csv(file_path, parse_dates=['update_date'])
    else:
        df = pd.read_excel(file_path, parse_dates=['update_date'], 
                            engine='odf',)
        df = df[["id", "pattern", "token_count", "update_date", "label", "para"]]
    
    # group id and aggregate by max label
    dfr = df.groupby(["id"]).agg({"label": "max"}).reset_index()

    # replace any NaN in "label" column with 0
    dfr.label.fillna(0, inplace=True)

    # the number of articles may have been more than what is contained in the file (because no keywords may have been found)
    sample_size_df = len(dfr)

    df_label_count = dfr.groupby('label').count().reset_index().astype(int)
    df_label_count = df_label_count.iloc[:,:2]
    df_label_count.columns = ['label', 'count']

    # the difference between sample_size and sample_size_df is the number of articles that had no keywords in the search
    df_label_count.loc[0, 'count'] += sample_size - sample_size_df

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

    df_label_count_all['label_name'] = df_label_count_all.label.apply(label_name)
    df_label_count_all['percentage'] = df_label_count_all['count'] / df_label_count_all['count'].sum() * 100
    df_label_count_all = df_label_count_all[['label_name', 'label', 'count', 'percentage']].sort_values(by='label')
    df_label_count_all['publisher'] = publisher_name
    df_label_count_all['sample_size'] = sample_size

    # create new column called "label_name" where the value is 0 if the label is 0, 1 if the label is greater than 0
    df_label_count['label_name'] = df_label_count.label.apply(
        lambda x: "Data and Code Not Available" if x == 0 else "Data or Code Publicly Available"
        )
    df_label_count['label'] = df_label_count.label.apply(
        lambda x: 0 if x == 0 else 1
        )
    df_label_count = df_label_count.groupby(["label_name", "label"]).agg({"count": "sum"}).reset_index()
    df_label_count['percentage'] = df_label_count['count'] / df_label_count['count'].sum() * 100
    df_label_count['publisher'] = publisher_name
    df_label_count['sample_size'] = sample_size

    return df_label_count, df_label_count_all


def plot_percent_articles_by_venue(
    df, path_save_dir=None, save_name="article_pcts_by_venue", dpi=300, save_plot=True
):
    df = df[df["label"] == 1].sort_values(by="percentage", ascending=False)

    title = "Percentage of articles (by publishing venue)\nwith data or code publicly available"

    fig, axes = plt.subplots(
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
        y="publisher",
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
    ax.set_title(title, fontsize=font_size, loc="left", pad=20, x=-0.25)
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


def plot_individual_venue(
    df,
    publisher_name="PHM Conf.",
    path_save_dir=None,
    save_name="article_pcts_phm_conf",
    dpi=300,
    save_plot=True,
):
    df = df[df["publisher"] == publisher_name]

    title = "Percentage of articles (by publishing venue)\nwith data or code publicly available"

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(4, 7),
    )

    sns.set(style="whitegrid", font="DejaVu Sans")
    pal = sns.color_palette("Greys_r")
    font_size = 14

    ax = sns.barplot(
        x="percentage",
        y="label_name",
        data=df,
        color="#bd0c0c",
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
    #     ax.text(-0.4, 1.06, title,
    #     verticalalignment='top', horizontalalignment='left',
    #     transform=ax.transAxes,
    #     color='black', fontsize=12)

    ax.set_title(title, fontsize=font_size, loc="left", pad=20, x=-0.25)
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
    publisher_names_list = ["PHM Conf.", "Energies", "MSSP", "arxiv"]
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

    # plot
    plot_percent_articles_by_venue(
        df,
        path_save_dir=path_save_dir,
        save_name="article_pcts_by_venue",
        dpi=300,
        save_plot=True,
    )

    plot_individual_venue(
        df_all,
        publisher_name="PHM Conf.",
        path_save_dir=path_save_dir,
        save_name="article_pcts_phm_conf",
        dpi=300,
        save_plot=True,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    proj_dir = Path(__file__).resolve().parents[2]

    main()
