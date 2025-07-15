from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import stats, t
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from spark import SPARK, Inst

app = typer.Typer()


@app.command()
def regression(
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
    Perform regression analysis using multiple models on specified instruments.

    Saves the plot of regression results to disk.

    :param spark_pathname: The SPARK data release directory pathname.
    :param cache_pathname: The cache pathname.
    :param output_pathname: The output pathname.
    """
    # Initialise the models to use for analysis:
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "HistGBRT": HistGradientBoostingRegressor(),
        "Random Forest": RandomForestRegressor(),
    }

    # Select the instruments to use for analysis:
    instruments = [
        Inst.CBCL_6_18,
        Inst.DCDQ,
        Inst.RBSR,
        Inst.SCQ,
    ]

    # Run regression analysis or load from cache:
    cache_path = Path(cache_pathname) / "regression"
    cache_path.mkdir(parents=True, exist_ok=True)
    r2_mean_filepath = cache_path / "regression_results.npz"
    r2_ci_lower_filepath = cache_path / "r2_ci_lower.npz"
    r2_ci_upper_filepath = cache_path / "r2_ci_upper.npz"
    subject_counts_filepath = cache_path / "subject_counts.npz"

    if r2_mean_filepath.exists() and subject_counts_filepath.exists():
        r2_mean_data = np.load(r2_mean_filepath)
        r2_ci_lower_data = np.load(r2_ci_lower_filepath)
        r2_ci_upper_data = np.load(r2_ci_upper_filepath)
        r2_mean = {key: r2_mean_data[key] for key in r2_mean_data.files}
        r2_ci_lower = {key: r2_ci_lower_data[key] for key in r2_ci_lower_data.files}
        r2_ci_upper = {key: r2_ci_upper_data[key] for key in r2_ci_upper_data.files}
        subject_counts_data = np.load(subject_counts_filepath)
        subject_counts = subject_counts_data["subject_counts"]
    else:
        r2_mean, r2_ci_lower, r2_ci_upper, subject_counts = run_regression(
            spark_pathname=spark_pathname, models=models, instruments=instruments
        )
        np.savez(r2_mean_filepath, **r2_mean)
        np.savez(r2_ci_lower_filepath, **r2_ci_lower)
        np.savez(r2_ci_upper_filepath, **r2_ci_upper)
        np.savez(subject_counts_filepath, subject_counts=subject_counts)

    # Plot the analysis results:
    fig_regression_results = plot_regression_results(
        instruments=instruments,
        r2_mean=r2_mean,
        r2_ci_lower=r2_ci_lower,
        r2_ci_upper=r2_ci_upper,
        models=models,
    )
    fig_subject_counts = plot_subject_counts(
        instruments=instruments,
        subject_counts=subject_counts,
    )

    # Save the figures to the output directory:
    fig_regression_results.savefig(Path(output_pathname) / "regression_results.png")
    fig_subject_counts.savefig(Path(output_pathname) / "subject_counts.png")

    return


def plot_subject_counts(
    instruments: list[Inst],
    subject_counts: np.ndarray,
) -> Figure:
    """
    Plots a heatmap of subject counts for each combination of instruments.

    :param instruments: A list of instruments used in the analysis.
    :param subject_counts: A 2-dimensional array where each cell represents the number of subjects for the corresponding
    instrument pair.
    :return:
    """
    # Calculate the number of instruments:
    n = len(instruments)

    # Precalculate labels from instrument codes:
    index = [inst.code for inst in instruments]

    # Initialise the figure:
    fig, ax = plt.subplots()

    # Display the matrix as an image:
    im = ax.imshow(subject_counts, cmap="Blues", vmin=0)

    # Set the axes ticks and labels:
    ax.set_yticks(range(n), labels=index)
    ax.set_xticks(range(n), labels=index, rotation=90)
    ax.set_xlabel("Question")
    ax.set_ylabel("Score")

    # Add text for each cell:
    for x, col_x in enumerate(index):
        for y, col_y in enumerate(index):
            if x == y:
                continue
            ax.text(
                y,
                x,
                f"{subject_counts[x][y]/1_000:.1f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if subject_counts[x][y] > max(subject_counts.flatten()) / 2
                    else "black"
                ),
            )

    # Add a colorbar:
    fig.colorbar(im, shrink=0.5)

    # Add a title :
    fig.suptitle(r"Subject count/$\times 10^3$")

    plt.tight_layout()

    return fig


def plot_regression_results(
    instruments: list[Inst],
    r2_mean: dict[str, np.ndarray],
    r2_ci_lower: dict[str, np.ndarray],
    r2_ci_upper: dict[str, np.ndarray],
    models: dict[str, Any],
    nrows: int = 3,
    ncols: int = 2,
    figsize: tuple[int, int] = (10, 16),
) -> Figure:
    """
    Plots a grid of heatmaps. Each heatmap plots the regression results for a model. Each cell is coloured based on the
    mean :math:`R^2` achieved by the model when tasked with using the x-axis instrument's question features to predict
    the y-axis instrument's score.

    :param instruments: The instruments under analysis.
    :param r2_mean: A dictionary of :math:`R^2` matrices keyed by a model.
    :param r2_ci_lower: A dictionary of confidence interval lower bound matrices keyed by a model.
    :param r2_ci_upper: A dictionary of confidence interval upper bound matrices keyed by a model.
    :param models: The models used for analysis.
    :param nrows: The number of rows in the heatmap grid.
    :param ncols: The number of columns in the heatmap grid.
    :param figsize: The size of the figure.
    :return: The plotted figure.
    """
    # Calculate the number of instruments:
    n = len(instruments)

    # Precalculate labels from instrument codes:
    index = [inst.code for inst in instruments]

    # Initialise the figure:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Ensure axes is an array of axes:
    if len(models) == 1:
        axes = [axes]

    # Convert the grid of axes into one array for easier iteration:
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        r2_mean_matrix = r2_mean[model]
        r2_ci_lower_matrix = r2_ci_lower[model]
        r2_ci_upper_matrix = r2_ci_upper[model]

        # Display the matrix as an image:
        ax.imshow(r2_mean_matrix, cmap="Blues", vmin=0, vmax=1)

        # Set the title to the model:
        ax.set_title(model)

        # Set the axes ticks and labels:
        ax.set_yticks(range(n), labels=index)
        ax.set_xticks(range(n), labels=index, rotation=90)
        ax.set_xlabel("Question")
        ax.set_ylabel("Score")

        # Add text for each cell:
        for x, col_x in enumerate(index):
            for y, col_y in enumerate(index):
                if x == y:
                    continue
                r2_mean_value = r2_mean_matrix[x][y]
                r2_ci_lower_value = r2_ci_lower_matrix[x][y]
                r2_ci_upper_value = r2_ci_upper_matrix[x][y]
                ax.text(
                    y,
                    x,
                    f"{r2_mean_value:.2f}\n[{r2_ci_lower_value:.2f}, {r2_ci_upper_value:.2f}]",
                    ha="center",
                    va="center",
                )

    # Remove any extra axes:
    for ax in axes[n + 1 :]:
        fig.delaxes(ax)

    # Add a title to the whole figure:
    fig.suptitle("$R^2$ mean and 95% confidence interval below")

    plt.tight_layout()

    return fig


def run_regression(
    spark_pathname: str,
    models: dict[str, Any],
    instruments: list[Inst],
    kf=KFold(shuffle=True),
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray
]:
    """
    Executed regression analysis for each combination of instruments and models provided, evaluating the
    cross-validation mean :math:`R^2`, confidence interval lower and upper bounds, and subject counts.

    :param kf: A K-Fold cross-validator.
    :param spark_pathname: The SPARK data release pathname.
    :param models: A dictionary of models keyed by their name.
    :param instruments: A list of instruments to carry out regression on.
    :return: A dictionary of :math:`R^2` matrices keyed by model, a dictionary of confidence interval lower bounds keyed
    by model, a dictionary of confidence interval upper bounds keyed by model, and a matrix of subject counts.
    """

    # Initialise the dataset with the appropriate instruments:
    ds = SPARK(
        spark_pathname=spark_pathname,
        instruments=instruments,
    )

    # Calculate the number of trials:
    n_instruments = len(instruments)
    n_models = len(models)
    n_trials = n_models * (n_instruments**2 - n_instruments)

    # Initialise dictionaries:
    regression_results = {
        model: np.zeros(
            shape=(
                n_instruments,
                n_instruments,
            ),
        )
        for model in models.keys()
    }
    r2_ci_lower = {
        model: np.zeros(
            shape=(
                n_instruments,
                n_instruments,
            ),
        )
        for model in models.keys()
    }
    r2_ci_upper = {
        model: np.zeros(
            shape=(
                n_instruments,
                n_instruments,
            ),
        )
        for model in models.keys()
    }
    subject_counts = np.zeros(shape=(n_instruments, n_instruments))

    # Set up the progress bar:
    with tqdm(total=n_trials, desc="Running regression") as pbar:
        for x_index, x_inst in enumerate(instruments):
            for y_index, y_inst in enumerate(instruments):
                # Early continue if the instruments are identical or invalid:
                if (
                    x_index == y_index
                    or x_inst.question_features is None
                    or y_inst.final_score_feature is None
                ):
                    continue

                # Join the appropriate features:
                df = ds.join(
                    features=x_inst.question_features + [y_inst.final_score_feature],
                    how="inner",
                )

                # Drop None values from the resulting dataframe:
                df = df.dropna()

                n_subjects = len(df)
                subject_counts[x_index, y_index] = n_subjects

                # Split the dataframe into questions and scores dataframes:
                questions = df.drop(columns=[y_inst.final_score_feature.col])
                scores = df[y_inst.final_score_feature.col]

                for model_key, model in models.items():
                    # Update the progress bar with the current instruments and models:
                    pbar.set_postfix(
                        {"x": x_inst.code, "y": y_inst.code, "model": model_key}
                    )

                    # Prepend the model with a standard scaler to form a pipeline:
                    pipeline = make_pipeline(StandardScaler(), model)

                    # Run cross-validation:
                    cv_results = cross_validate(
                        pipeline, questions, scores, cv=kf, scoring="r2", n_jobs=-1
                    )

                    # Extract the scores and calculate the confidence intervals:
                    test_scores = cv_results["test_score"]
                    mean = np.mean(test_scores)
                    sem = stats.sem(test_scores)
                    confidence = 0.95
                    ci = t.interval(
                        confidence, df=len(test_scores) - 1, loc=mean, scale=sem
                    )

                    r2_ci_lower[model_key][x_index, y_index] = ci[0]
                    r2_ci_upper[model_key][x_index, y_index] = ci[1]
                    regression_results[model_key][x_index, y_index] = mean

                    # Update the progress bar:
                    pbar.update(1)

    return regression_results, r2_ci_lower, r2_ci_upper, subject_counts
