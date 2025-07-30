from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from spark import SPARK, Inst, Feat

app = typer.Typer()


@app.command()
def initial_pheno(
    feature_set: Annotated[
        str,
        typer.Argument(help="The question set to use", envvar="PHENOTYPIC_FEATURE_SET"),
    ] = ".",
    spark_pathname: Annotated[
        str,
        typer.Argument(
            help="The SPARK data release directory pathname", envvar="SPARK_PATHNAME"
        ),
    ] = ".",
    cache_pathname: Annotated[
        str,
        typer.Argument(help="The cache directory pathname", envvar="CACHE_PATHNAME"),
    ] = ".",
    output_pathname: Annotated[
        str,
        typer.Argument(help="The output directory pathname", envvar="OUTPUT_PATHNAME"),
    ] = ".",
) -> None:
    """
    Runs clustering algorithms on phenotypic features from SPARK data and generates metrics plots.

    :param feature_set: A string indicating which set of features to use for clustering.
    :param spark_pathname: The SPARK data release directory pathname.
    :param cache_pathname: The cache directory pathname.
    :param output_pathname: The output directory pathname where plots will be saved.
    """
    cache_path = Path(cache_pathname) / "initial_pheno"
    cache_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_pathname)

    feature_sets: dict[str, list[Feat]] = {
        "questions": (
            Inst.SCQ.question_features
            + Inst.CBCL_6_18.question_features
            + Inst.RBSR.question_features
            + Inst.DCDQ.question_features
        ),
        "final_score": [
            Inst.SCQ.final_score_feature,
            Inst.CBCL_6_18.final_score_feature,
            Inst.RBSR.final_score_feature,
            Inst.DCDQ.final_score_feature,
        ],
    }

    features = feature_sets[feature_set]

    ds, df, instruments = SPARK.init_and_join(
        spark_pathname=spark_pathname,
        features=features,
    )

    df = df.dropna()

    k_mean_metrics_filepath = cache_path / f"{feature_set}_k_means_metrics.feather"
    if k_mean_metrics_filepath.exists():
        k_means_metrics_df = pd.read_feather(k_mean_metrics_filepath)
    else:
        k_means_metrics_df = run_k_means(df)
        k_means_metrics_df.to_feather(k_mean_metrics_filepath)

    fig = plot_metrics_against_k(
        metrics_df=k_means_metrics_df,
        metric_keys=[
            "inertia",
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
        ],
        title="K-mean metrics on SCQ, CBCL/6-18, RBS-R, and DCDQ final score features",
    )
    k_means_metrics_plot_filepath = output_path / f"{feature_set}_k_means_metrics.png"
    fig.savefig(k_means_metrics_plot_filepath)

    gmm_metrics_filepath = cache_path / f"{feature_set}_gmm_metrics.feather"
    if gmm_metrics_filepath.exists():
        gmm_metrics_df = pd.read_feather(gmm_metrics_filepath)
    else:
        gmm_metrics_df = run_gmm(df)
        gmm_metrics_df.to_feather(gmm_metrics_filepath)

    gmm_metrics_plot_filepath = output_path / f"{feature_set}_gmm_metrics.png"
    fig = plot_metrics_against_k(
        metrics_df=gmm_metrics_df,
        metric_keys=[
            "log_likelihood",
            "bic",
            "aic",
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
        ],
        title="GMM metrics on SCQ, CBCL/6-18, RBS-R, and DCDQ final score features",
    )
    fig.savefig(gmm_metrics_plot_filepath)

    return


def run_gmm(df: pd.DataFrame, repeats: int = 5) -> pd.DataFrame:
    """
    Runs Gaussian Mixture Model (GMM) clustering on the provided DataFrame and computes various clustering metrics.

    :param df: A dataframe containing the features to be clustered.
    :param repeats: Number of times to repeat the GMM clustering for each k value.
    :return: A dataframe containing the clustering metrics for each k value.
    """
    k_values = range(2, 15)
    df = StandardScaler().fit_transform(df)
    results: list[dict[str, float]] = []

    n_trials = repeats * len(k_values)

    with tqdm(total=n_trials, desc="Running GMM") as pbar:
        for k in k_values:
            log_likelihoods = np.ndarray(shape=(repeats,), dtype=float)
            bics = np.ndarray(shape=(repeats,), dtype=float)
            aics = np.ndarray(shape=(repeats,), dtype=float)
            silhouette_scores = np.ndarray(shape=(repeats,), dtype=float)
            davies_bouldin_scores = np.ndarray(shape=(repeats,), dtype=float)
            calinski_harabasz_scores = np.ndarray(shape=(repeats,), dtype=float)

            for r in range(repeats):
                pbar.set_postfix({"k": k, "repeat": r + 1})

                gmm = GaussianMixture(
                    n_components=k,
                )
                labels = gmm.fit_predict(df)
                log_likelihoods[r] = gmm.score(df)
                bics[r] = gmm.bic(df)
                aics[r] = gmm.aic(df)
                silhouette_scores[r] = silhouette_score(df, labels)
                davies_bouldin_scores[r] = davies_bouldin_score(df, labels)
                calinski_harabasz_scores[r] = calinski_harabasz_score(df, labels)

                pbar.update(1)

            results.append(
                {
                    "k": k,
                    "mean_silhouette_score": silhouette_scores.mean(),
                    "mean_davies_bouldin_score": davies_bouldin_scores.mean(),
                    "mean_calinski_harabasz_score": calinski_harabasz_scores.mean(),
                    "mean_log_likelihood": log_likelihoods.mean(),
                    "mean_bic": bics.mean(),
                    "mean_aic": aics.mean(),
                    "std_log_likelihood": log_likelihoods.std(),
                    "std_bic": bics.std(),
                    "std_aic": aics.std(),
                    "std_silhouette_score": silhouette_scores.std(),
                    "std_davies_bouldin_score": davies_bouldin_scores.std(),
                    "std_calinski_harabasz_score": calinski_harabasz_scores.std(),
                }
            )

    metrics = pd.DataFrame(results)

    return metrics


def run_k_means(df: pd.DataFrame, repeats: int = 5) -> pd.DataFrame:
    """
    Runs k-means clustering on the provided DataFrame and computes various clustering metrics.

    :param df: A dataframe containing the features to be clustered.
    :param repeats: Number of times to repeat the k-means clustering for each k value.
    :return: A dataframe containing the clustering metrics for each k value.
    """
    k_values = range(2, 15)

    df = StandardScaler().fit_transform(df)
    results: list[dict[str, float]] = []

    n_trials = repeats * len(k_values)

    with tqdm(total=n_trials, desc="Running k-means") as pbar:
        for k in k_values:
            inertias = np.ndarray(shape=(repeats,), dtype=float)
            silhouette_scores = np.ndarray(shape=(repeats,), dtype=float)
            davies_bouldin_scores = np.ndarray(shape=(repeats,), dtype=float)
            calinski_harabasz_scores = np.ndarray(shape=(repeats,), dtype=float)

            for r in range(repeats):
                pbar.set_postfix({"k": k, "repeat": r + 1})

                kmeans = KMeans(
                    n_clusters=k,
                    n_init="auto",
                )
                labels = kmeans.fit_predict(df)
                inertias[r] = kmeans.inertia_
                silhouette_scores[r] = silhouette_score(df, labels)
                davies_bouldin_scores[r] = davies_bouldin_score(df, labels)
                calinski_harabasz_scores[r] = calinski_harabasz_score(df, labels)

                pbar.update(1)

            results.append(
                {
                    "k": k,
                    "mean_inertia": inertias.mean(),
                    "mean_silhouette_score": silhouette_scores.mean(),
                    "mean_davies_bouldin_score": davies_bouldin_scores.mean(),
                    "mean_calinski_harabasz_score": calinski_harabasz_scores.mean(),
                    "std_inertia": inertias.std(),
                    "std_silhouette_score": silhouette_scores.std(),
                    "std_davies_bouldin_score": davies_bouldin_scores.std(),
                    "std_calinski_harabasz_score": calinski_harabasz_scores.std(),
                }
            )

    metrics = pd.DataFrame(results)

    return metrics


def plot_metrics_against_k(
    metrics_df: pd.DataFrame, metric_keys: list[str], title: str
) -> Figure:
    """
    Generates a plot of specified metrics against different values of k, where each metric is displayed
    in its own subplot panel. The function allows for error bars based on the corresponding standard
    deviations.

    :param metrics_df: A pandas DataFrame containing k values, the mean and standard deviation
        of each specified metric. The DataFrame must include columns named 'k',
        'mean_<metric>', and 'std_<metric>' for all the specified metrics in `metric_keys`.
    :param metric_keys: A list of strings, where each string is the name of a metric to be plotted.
        Metrics must have corresponding columns in `metrics_df`.
    :param title: A string to specify the title of the generated figure.
    :return: The plot figure.
    """
    custom_ylim = {
        "inertia": (0, None),
        "silhouette_score": (-1, 1),
        "davies_bouldin_score": (0, None),
        "calinski_harabasz_score": (0, None),
    }

    num_metrics = len(metric_keys)

    ncols = min(2, num_metrics)
    nrows = int(np.ceil(num_metrics / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metric_keys):
        metric_label = metric.replace("_", " ").title()
        ax.errorbar(
            metrics_df["k"],
            metrics_df[f"mean_{metric}"],
            yerr=metrics_df[f"std_{metric}"],
            marker="x",
            capsize=5,
        )
        ax.set_xlabel("k")
        ax.set_ylabel(f"Mean {metric_label}")
        ax.set_ylim(custom_ylim.get(metric, (None, None)))
        ax.grid(True)

    for ax in axes_flat[len(metric_keys) :]:
        fig.delaxes(ax)

    # fig.suptitle(title, wrap=True)
    plt.tight_layout()
    return fig
