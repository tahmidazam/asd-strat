import pickle
import time
from pathlib import Path
from typing import Annotated, Optional, Hashable, Callable

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from tqdm import tqdm

from spark import Feat, SPARK, Inst
from .paper_reproduction_utils.constants import (
    BHC_FEATURES,
    BHS_FEATURES,
    EPHEMERAL_FEATURES,
    SCQ_FEATURES,
    RBSR_FEATURES,
    CBCL_6_18_FEATURES,
    CATEGORY_MAP,
)

app = typer.Typer()


@app.command()
def paper_reproduction(
    use_original_ids: Annotated[
        bool,
        typer.Argument(
            help="Use the subjects from the original paper",
            envvar="PAPER_REPRODUCTION_USE_ORIGINAL_IDS",
        ),
    ] = True,
    sample_frac: Annotated[
        float,
        typer.Argument(
            help="The proportion of the dataset to sample",
            envvar="PAPER_REPRODUCTION_SAMPLE_FRAC",
        ),
    ] = True,
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

    cache_path = Path(cache_pathname) / "paper_reproduction"
    cache_path.mkdir(parents=True, exist_ok=True)

    df, df_descriptor, Z_p, ds, instruments = run_preprocessing(
        spark_pathname=spark_pathname,
        use_original_ids=use_original_ids,
        sample_frac=sample_frac,
    )

    model, labels = run_model(
        df=df,
        df_descriptor=df_descriptor,
        Z_p=Z_p,
        n_components=4,
        n_init=1,
        progress_bar=2,
        verbose=1,
    )

    with open(cache_path / f"model_{time.time() }.pkl", "wb") as file_handle:
        pickle.dump(model, file_handle)

    return


def find_optimal_k(df, df_descriptor, Z_p, ds, instruments, cache_path):
    k_range = range(2, 10)
    len_k_range = len(k_range)
    repeats = 5
    metrics: dict[str, Callable[[StepMix], float]] = {
        "aic": lambda m: m.aic(df),
        "bic": lambda m: m.bic(df),
        "caic": lambda m: m.caic(df),
        "entropy": lambda m: m.entropy(df),
        "scaled_entropy": lambda m: m.relative_entropy(df),
        "log_likelihood": lambda m: m.lower_bound_,
    }
    results: dict[str, np.ndarray] = {
        key: np.zeros(shape=(len_k_range, repeats)) for key in metrics.keys()
    }

    if (cache_path / f"{list(metrics.keys())[0]}.npz").exists():
        for key in metrics.keys():
            data = np.load(cache_path / f"{key}.npz")
            results[key] = data["arr_0"]
    else:
        with tqdm(
            total=len_k_range * repeats,
            desc=f"Running models",
        ) as pbar:
            for k_index, k in enumerate(k_range):
                for r_index in range(repeats):
                    pbar.set_postfix({"k": k, "repeat": r_index + 1})
                    model, labels = run_model(
                        df,
                        df_descriptor,
                        Z_p,
                        n_components=k,
                        n_init=20,
                    )

                    for key, f in metrics.items():
                        results[key][k_index, r_index] = f(model)

                    pbar.update(1)

        for key in metrics.keys():
            np.savez(cache_path / f"{key}.npz", results[key])

    ks = np.array(list(k_range))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, matrix) in zip(axes, results.items()):
        mean_vals = matrix.mean(axis=1)
        std_vals = matrix.std(axis=1)
        ax.errorbar(ks, mean_vals, yerr=std_vals, fmt="o-", capsize=4)
        ax.set_title(key)
        ax.set_xlabel("k")
        ax.set_ylabel(key)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def run_model(
    df,
    df_descriptor,
    Z_p,
    n_components: int = 4,
    n_init: int = 20,
    progress_bar: int = 0,
    verbose: int = 0,
) -> tuple[StepMix, pd.Series]:
    model = StepMix(
        n_components=n_components,
        measurement=df_descriptor,
        structural="covariate",
        n_steps=1,
        n_init=n_init,
        progress_bar=progress_bar,
        verbose=verbose,
    )

    model.fit(df, Z_p)

    labels = model.predict(df)
    labels = pd.Series(labels)

    return model, labels


def run_preprocessing(
    spark_pathname: str,
    use_original_ids: bool = True,
    round_values: bool = True,
    sample_frac: Optional[float] = 0.01,
) -> tuple[Hashable, Hashable, pd.DataFrame, SPARK, list[Inst]]:
    """
    Reproduces the paper's preprocessing.

    :param spark_pathname: The SPARK data release directory pathname.
    :param use_original_ids: A flag indicating whether to filter only original subject IDs. Defaults to True.
    :type use_original_ids: bool, optional
    :param round_values: A flag indicating whether to round the dataframe to the nearest integer. Defaults to True.
    :param sample_frac: The proportion of the dataset to sample. Defaults to 0.05 (5%).
    :return: A tuple containing the preprocessed dataframe, the dataset, and a list of instruments used during processing.
    :rtype: tuple[pd.DataFrame, SPARK, list[Inst]]
    """
    # Define instruments to be used in the dataset:
    instruments: list[Inst] = [Inst.SCQ, Inst.BHC, Inst.BHS, Inst.RBSR, Inst.CBCL_6_18]

    # Initialize the SPARK dataset with the specified instruments:
    ds = SPARK(spark_pathname=spark_pathname, instruments=instruments)

    # Load BHC and BHS dataframes, filter by age, concatenate, and rename columns:
    bhc_df = ds.join(features=BHC_FEATURES, rename=False)
    bhc_df = bhc_df.loc[
        (bhc_df[Feat.BHC_AGE_AT_EVAL_YEARS.source_col] >= 4)
        & (bhc_df[Feat.BHC_AGE_AT_EVAL_YEARS.source_col] <= 18)
    ]
    bhs_df = ds.join(features=BHS_FEATURES, rename=False)
    bhs_df = bhs_df.loc[
        (bhs_df[Feat.BHS_AGE_AT_EVAL_YEARS.source_col] >= 4)
        & (bhs_df[Feat.BHS_AGE_AT_EVAL_YEARS.source_col] <= 18)
    ]
    bh_df = pd.concat(
        [
            bhc_df,
            bhs_df,
        ],
        join="inner",
    )
    bh_df = bh_df.rename(columns=lambda col: f"BHC_{col}")

    # Join the dataset with remaining features:
    df = ds.join(features=SCQ_FEATURES + RBSR_FEATURES + CBCL_6_18_FEATURES).join(bh_df)

    # Filter the dataframe based on age and missing values:
    df = df.loc[
        (df[Feat.SCQ_AGE_AT_EVAL_YEARS.col] <= 18)
        & (df[Feat.SCQ_MISSING_VALUES.col] < 1)
        & (df[Feat.SCQ_AGE_AT_EVAL_YEARS.col] >= 4)
        & (df[Feat.RBSR_AGE_AT_EVAL_YEARS.col] <= 18)
        & (df[Feat.RBSR_MISSING_VALUES.col] < 1)
        & (df[Feat.RBSR_AGE_AT_EVAL_YEARS.col] >= 4)
    ]

    # Drop ephemeral features:
    df = df.drop(columns=[feat.col for feat in EPHEMERAL_FEATURES], axis=1)

    # Replace categorical values with integers:
    for features, replace_dict in CATEGORY_MAP:
        cols = [feat.col for feat in features]
        df[cols] = df[cols].replace(replace_dict)

    # Use original subject IDs if specified:
    if use_original_ids:
        original_subject_sp_ids = get_original_subject_sp_ids()
        df = df.loc[df.index.isin(original_subject_sp_ids)]

    # Remove columns with more than 10% missing values and drop rows with any NaN values:
    df = df.loc[:, df.isna().sum() / len(df) < 0.1].dropna(axis=0)

    # Convert all columns to float64 type:
    df = df.astype("float64")

    # Round the dataframe if specified:
    if round_values:
        df = df.round()

    # Sample a fraction of the dataframe if specified:
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=1)

    # Define and select covariates, Z_p:
    Z_p_cols = [Feat.SCQ_SEX.col, Feat.SCQ_AGE_AT_EVAL_YEARS.col]
    Z_p = df[Z_p_cols]

    # Drop covariates from the dataframe:
    df = df.drop(Z_p_cols, axis=1)

    # Split the dataframe columns into binary, continuous, and categorical:
    binary_cols, continuous_cols, categorical_cols = split_dataframe_columns_by_type(df)

    # Get the mixed data and descriptor:
    df, df_descriptor = get_mixed_descriptor(
        dataframe=df,
        continuous=continuous_cols,
        binary=binary_cols,
        categorical=categorical_cols,
    )

    print(
        f"Preprocessing complete (shape={df.shape}, use_original_ids={use_original_ids}, sample_frac={sample_frac})"
    )

    return df, df_descriptor, Z_p, ds, instruments


def get_original_subject_sp_ids() -> set[str]:
    """
    Retrieves the original subject identifiers used by the paper from a text file.

    :return: A set of unique subject identifiers read from the file.
    :rtype: set[str]
    """
    with open("data/original_subject_sp_ids.txt", "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def split_dataframe_columns_by_type(
    df: pd.DataFrame, cat_unique_threshold: int = 10
) -> tuple[list[str], list[str], list[str]]:
    """
    Classify the columns of a DataFrame as binary, categorical, or continuous.

    Binary columns are those with unique non-null values subset of {0.0, 1.0}.
    Categorical columns have unique values less than or equal to the threshold.
    Continuous columns are the remaining ones.

    :param df: The DataFrame whose columns are to be classified.
    :type df: pd.DataFrame
    :param cat_unique_threshold: The maximum number of unique values for a column to be considered categorical.
    :type cat_unique_threshold: int
    :return: A tuple of three lists: (binary columns, continuous columns, categorical columns)
    :rtype: tuple[list[str], list[str], list[str]]
    """

    binary_cols = []
    continuous_cols = []
    categorical_cols = []

    for col in df.columns:
        s = df[col].dropna()
        unique_vals = s.unique()

        if set(unique_vals).issubset({0.0, 1.0}):
            binary_cols.append(col)
        elif len(unique_vals) <= cat_unique_threshold:
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    return binary_cols, continuous_cols, categorical_cols
