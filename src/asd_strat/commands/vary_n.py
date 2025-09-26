import pickle
from pathlib import Path
from typing import Hashable

import matplotlib
import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from tqdm import tqdm
from typing_extensions import Annotated

from asd_strat.commands.paper_reproduction import split_dataframe_columns_by_type
from spark import SPARK, Feat

app = typer.Typer()


@app.command()
def vary_n(
    spark_pathname: Annotated[str, typer.Argument(envvar="SPARK_PATHNAME")] = ".",
    output_pathname: Annotated[str, typer.Argument(envvar="OUTPUT_PATHNAME")] = ".",
    cache_pathname: Annotated[str, typer.Argument(envvar="CACHE_PATHNAME")] = ".",
    k_lower: int = 2,
    k_upper: int = 6,
    repeats: int = 10,
    threshold_lower: float = 0.4,
    threshold_upper: float = 0.95,
    threshold_step: float = 0.05,
):
    def serialise_params() -> str:
        return f"t_{threshold_lower}-{threshold_upper}_t_step_{threshold_step}_k_{k_lower}-{k_upper}_repeats_{repeats}.npz"

    cache_dir = Path(cache_pathname) / "vary_n"
    cache_dir.mkdir(exist_ok=True, parents=True)
    filepath = cache_dir / serialise_params()

    df = get_df(spark_pathname, output_pathname, cache_pathname)

    output_dir = Path(output_pathname)
    data_retention_vs_threshold_filepath = (
        output_dir / "data_retention_vs_threshold.png"
    )
    # if not data_retention_vs_threshold_filepath.exists():
    data_retention_vs_threshold_fig = plot_data_retention_vs_threshold(df)
    data_retention_vs_threshold_fig.savefig(data_retention_vs_threshold_filepath)

    k_means_model = {
        "name": "k_means",
        "class": lambda k, d: KMeans(n_clusters=k),
        "metrics": {
            "inertia": lambda m, d, l: m.inertia_,
            "silhouette": lambda m, d, l: silhouette_score(d, l),
            "davies_bouldin": lambda m, d, l: davies_bouldin_score(d, l),
            "calinski_harabasz": lambda m, d, l: calinski_harabasz_score(d, l),
        },
    }
    stepmix_model = {
        "name": "stepmix",
        "class": lambda k, d: make_stepmix_model(k, df),
        "metrics": {
            "log_likelihood": lambda m, d, l: m.lower_bound_,
            "average_log_likelihood": lambda m, d, l: m.score(d),
            "aic": lambda m, d, l: m.aic(d),
            "bic": lambda m, d, l: m.bic(d),
            "caic": lambda m, d, l: m.caic(d),
            "sabic": lambda m, d, l: m.sabic(d),
            "entropy": lambda m, d, l: m.entropy(d),
            "scaled_relative_entropy": lambda m, d, l: m.relative_entropy(d),
        },
    }
    gmm_model = {
        "name": "gmm",
        "class": lambda k, d: GaussianMixture(n_components=k),
        "metrics": {
            "log_likelihood": lambda m, d, l: m.score(d),
            "silhouette": lambda m, d, l: silhouette_score(d, l),
            "davies_bouldin": lambda m, d, l: davies_bouldin_score(d, l),
            "calinski_harabasz": lambda m, d, l: calinski_harabasz_score(d, l),
            "bic": lambda m, d, l: m.bic(d),
            "aic": lambda m, d, l: m.aic(d),
        },
    }
    birch_model = {
        "name": "birch",
        "class": lambda k, d: Birch(n_clusters=k),
        "metrics": {
            "silhouette": lambda m, d, l: silhouette_score(d, l),
            "davies_bouldin": lambda m, d, l: davies_bouldin_score(d, l),
            "calinski_harabasz": lambda m, d, l: calinski_harabasz_score(d, l),
        },
    }
    hierarchical_model = {
        "name": "hierarchical",
        "class": lambda k, d: AgglomerativeClustering(n_clusters=k),
        "metrics": {
            "silhouette": lambda m, d, l: silhouette_score(d, l),
            "davies_bouldin": lambda m, d, l: davies_bouldin_score(d, l),
            "calinski_harabasz": lambda m, d, l: calinski_harabasz_score(d, l),
        },
    }
    models = [
        gmm_model,
    ]

    k_range = range(k_lower, k_upper + 1)
    t_range = np.arange(
        threshold_lower, threshold_upper + threshold_step, threshold_step
    )

    def serialise_t(t: np.floating) -> str:
        return f"{t:.2f}"

    shape = (len(k_range), repeats)

    results = {
        model["name"]: {
            serialise_t(t): {
                metric_key: np.ndarray(shape=shape, dtype=float)
                for metric_key in model["metrics"].keys()
            }
            for t in t_range
        }
        for model in models
    }

    total = len(models) * len(k_range) * len(t_range) * repeats
    with tqdm(total=total, desc="Clustering") as pbar:
        for model in models:
            model_dir = cache_dir / model["name"]
            model_dir.mkdir(exist_ok=True, parents=True)
            for t in t_range:
                df_subset = df.loc[:, df.isna().mean() < t].dropna(axis=0)
                df_subset = StandardScaler().fit_transform(df_subset)
                for k in k_range:
                    for r in range(repeats):
                        pbar.set_postfix(
                            {
                                "model": model["name"],
                                "threshold": t,
                                "shape": df_subset.shape,
                                "k": k,
                                "repeat": r + 1,
                            }
                        )
                        filepath = model_dir / f"{serialise_t(t)}_{k}_{r}"

                        if filepath.exists():
                            with open(filepath, "rb") as f:
                                intermediate_results = pickle.load(f)
                        else:
                            model_class_init_result = model["class"](k, df_subset)

                            if isinstance(model_class_init_result, tuple):
                                model_class, df_to_use = model_class_init_result
                                df_to_use = df_to_use.loc[
                                    :, df_to_use.isna().mean() < t
                                ].dropna(axis=0)
                                df_to_use = StandardScaler().fit_transform(df_to_use)
                            else:
                                model_class = model_class_init_result
                                df_to_use = df_subset

                            model_class.fit(df_to_use)
                            labels = model_class.predict(df_to_use)

                            intermediate_results = {
                                metric_key: metric_fn(model_class, df_to_use, labels)
                                for metric_key, metric_fn in model["metrics"].items()
                            }

                            with open(filepath, "wb") as f:
                                pickle.dump(intermediate_results, f)  # type: ignore

                        for metric_key, metric_fn in model["metrics"].items():
                            results[model["name"]][serialise_t(t)][metric_key][
                                k - k_range[0], r
                            ] = intermediate_results[metric_key]

                        pbar.update(1)
            plot_model_metrics(
                results,
                model["name"],
                k_range,
                n_cols=5,
                n_rows=1,
                exclude=["log_likelihood"],
            )


def plot_model_metrics(results, model_name, k_range, n_cols=2, n_rows=2, exclude=None):
    cmap = LinearSegmentedColormap.from_list("mycmap", ["white", "#426665"])
    metric_labels = {
        "inertia": "Inertia",
        "silhouette": "Silhouette Score",
        "davies_bouldin": "Davies-Bouldin Index",
        "calinski_harabasz": "Calinski-Harabasz Index",
        "log_likelihood": "Log Likelihood",
        "bic": "BIC",
        "aic": "AIC",
    }
    metric_subtitle = {
        "inertia": "(lower is better)",
        "silhouette": "(higher is better)",
        "davies_bouldin": "(lower is better)",
        "calinski_harabasz": "(higher is better)",
        "log_likelihood": "(higher is better)",
        "bic": "(lower is better)",
        "aic": "(lower is better)",
    }
    metric_ylims = {
        "inertia": (0, None),
        "silhouette": (-1, 1),
        "davies_bouldin": (0, None),
        "calinski_harabasz": (0, None),
    }
    model_data = results[model_name]

    thresholds = list(model_data.keys())
    metrics = list(model_data[thresholds[0]].keys())
    if exclude:
        metrics = [i for i in metrics if i not in exclude]
    n_metrics = len(metrics)

    px = 1 / matplotlib.rcParams["figure.dpi"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1359 * px, 810 * px))

    # Flatten axes for simpler indexing:
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    threshold_values = [float(t) for t in thresholds]
    min_thresh, max_thresh = min(threshold_values), max(threshold_values)
    normalized_values = [
        1.0 - 0.75 * (t - min_thresh) / (max_thresh - min_thresh)
        for t in threshold_values
    ]
    colors = cmap(normalized_values)

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]

        for j, threshold in enumerate(thresholds):
            metric_data = model_data[threshold][metric]

            ax.errorbar(
                k_range,
                metric_data.mean(axis=1),
                yerr=metric_data.std(axis=1),
                label=(f"$t={float(threshold):.2f}$"),
                color=colors[j],
                marker="x",
                capsize=5,
            )

        ax.set_xlabel("$k$")
        if metric in metric_ylims.keys():
            ax.set_ylim(metric_ylims[metric])
        ax.set_title(f"{metric_labels[metric]}\n{metric_subtitle[metric]}")
        ax.grid(True, alpha=0.3)

    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def make_stepmix_model(k: int, df: pd.DataFrame) -> tuple[StepMix, Hashable]:
    binary_cols, continuous_cols, categorical_cols = split_dataframe_columns_by_type(df)

    df, descriptor = get_mixed_descriptor(
        dataframe=df,
        continuous=continuous_cols,
        binary=binary_cols,
        categorical=categorical_cols,
    )

    return (
        StepMix(
            n_components=k,
            n_steps=2,
            measurement=descriptor,
            verbose=0,
        ),
        df,
    )


def get_df(
    spark_pathname: Annotated[str, typer.Argument(envvar="SPARK_PATHNAME")] = ".",
    output_pathname: Annotated[str, typer.Argument(envvar="OUTPUT_PATHNAME")] = ".",
    cache_pathname: Annotated[str, typer.Argument(envvar="CACHE_PATHNAME")] = ".",
):
    cache_dir = Path(cache_pathname) / "vary_n"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_filepath = cache_dir / "df.feather"
    if not df_filepath.exists():
        ds, df, instruments = SPARK.init_and_join(
            spark_pathname=spark_pathname,
            features=[
                Feat.BHC_CHILD_LIVES_WITH,
                Feat.BHC_CHILD_LIVES_WITH_V2,
                Feat.BHC_MARITAL_STATUS_BIOLOGICAL_PARENTS,
                Feat.BHC_MARITAL_STATUS_PARENT_GUARDIAN_V2,
                Feat.BHC_MOTHER_HIGHEST_EDUCATION,
                Feat.BHC_FATHER_HIGHEST_EDUCATION,
                Feat.BHC_MOTHER_OCCUPATION,
                Feat.BHC_FATHER_OCCUPATION,
                Feat.BHC_ANNUAL_HOUSEHOLD_INCOME,
                Feat.BHC_SPED_Y_N,
                Feat.BHC_SPED_BIRTH_TO_THREE,
                Feat.BHC_SPED_PRESCHOOL,
                Feat.BHC_SPED_IEP_ASD,
                Feat.BHC_SPED_IEP_OTHER,
                Feat.BHC_SPED_504,
                Feat.BHC_SPED_ASD_CLASS_FT,
                Feat.BHC_SPED_ASD_CLASS_PT,
                Feat.BHC_SPED_OTHER_CLASS_FT,
                Feat.BHC_SPED_OTHER_CLASS_PT,
                Feat.BHC_SPED_REVERSE,
                Feat.BHC_SPED_OTHER_SUPPORT,
                Feat.BHC_SPED_AIDE,
                Feat.BHC_SPED_SPEECH,
                Feat.BHC_SPED_OT,
                Feat.BHC_SPED_PT,
                Feat.BHC_SPED_BEHAVIOR,
                Feat.BHC_SPED_SOC_SKILLS,
                Feat.BHC_SPED_PRIV_ASD,
                Feat.BHC_SPED_OTHER,
                Feat.BHC_INTERVENTION_MEDICATION,
                Feat.BHC_INTERVENTION_BIOMEDICAL,
                Feat.BHC_INTERVENTION_BEHAVIORAL,
                Feat.BHC_INTERVENTION_OTHER_DEV,
                Feat.BHC_INTERVENTION_SPEECH_LANGUAGE,
                Feat.BHC_INTERVENTION_OT_SENSORY,
                Feat.BHC_INTERVENTION_OT_FINE_MOT,
                Feat.BHC_INTERVENTION_PT,
                Feat.BHC_INTERVENTION_RECREATIONAL_THERAPY,
                Feat.BHC_INTERVENTION_SOCIAL_SKILLS,
                Feat.BHC_INTERVENTION_COUNSELING,
                Feat.BHC_INTERVENTION_OTHER,
                Feat.BHC_TWIN_MULT_BIRTH,
                Feat.BHC_ZYGOSITY,
                Feat.BHC_TWIN_ASD,
                Feat.BHC_SMILED_AGE_MOS,
                Feat.BHC_SAT_WO_SUPPORT_AGE_MOS,
                Feat.BHC_CRAWLED_AGE_MOS,
                Feat.BHC_WALKED_AGE_MOS,
                Feat.BHC_FED_SELF_SPOON_AGE_MOS,
                Feat.BHC_USED_WORDS_AGE_MOS,
                Feat.BHC_COMBINED_WORDS_AGE_MOS,
                Feat.BHC_COMBINED_PHRASES_AGE_MOS,
                Feat.BHC_BLADDER_TRAINED_AGE_MOS,
                Feat.BHC_BOWEL_TRAINED_AGE_MOS,
                Feat.BHC_HAND,
                Feat.BHC_COG_AGE_LEVEL,
                Feat.BHC_COG_AGE_EQUIVALENT,
                Feat.BHC_COG_TEST_SCORE,
                Feat.BHC_FUNCTION_AGE_LEVEL,
                Feat.BHC_LANGUAGE_AGE_LEVEL,
                Feat.BHC_AGE_ONSET_MOS,
                Feat.BHC_ONSET_CONCERN,
                Feat.BHC_PLATEAU_Y_N,
                Feat.BHC_REGRESS_LANG_Y_N,
                Feat.BHC_REGRESS_LANG_AGE_MOS,
                Feat.BHC_REGRESS_LANG_RETURN_Y_N,
                Feat.BHC_REGRESS_LANG_TIME_RETURN,
                Feat.BHC_REGRESS_OTHER_Y_N,
                Feat.BHC_REGRESS_OTHER_SKILL_SOCIAL,
                Feat.BHC_REGRESS_OTHER_SKILL_PLAY,
                Feat.BHC_REGRESS_OTHER_SKILL_OTHER_DEV,
                Feat.BHC_REGRESS_OTHER_AGE_MOS,
                Feat.BHC_REGRESS_OTHER_RETURN_Y_N,
                Feat.BHC_REGRESS_OTHER_TIME_RETURN,
                Feat.BHC_CHILD_GRADE_SCHOOL,
                Feat.BHC_REPEAT_GRADE,
                Feat.BMS_ATTN_BEHAV,
                Feat.BMS_BEHAV_ADHD,
                Feat.BMS_BEHAV_CONDUCT,
                Feat.BMS_BEHAV_INTERMITT_EXPLOS,
                Feat.BMS_BEHAV_ODD,
                Feat.BMS_BIRTH_DEF_BONE,
                Feat.BMS_BIRTH_DEF_BONE_CLUB,
                Feat.BMS_BIRTH_DEF_BONE_MISS,
                Feat.BMS_BIRTH_DEF_BONE_POLYDACT,
                Feat.BMS_BIRTH_DEF_BONE_SPINE,
                Feat.BMS_BIRTH_DEF_CLEFT_LIP,
                Feat.BMS_BIRTH_DEF_CLEFT_PALATE,
                Feat.BMS_BIRTH_DEF_CNS,
                Feat.BMS_BIRTH_DEF_CNS_BRAIN,
                Feat.BMS_BIRTH_DEF_CNS_MYELO,
                Feat.BMS_BIRTH_DEF_FAC,
                Feat.BMS_BIRTH_DEF_GASTRO,
                Feat.BMS_BIRTH_DEF_GI_ESOPH_ATRES,
                Feat.BMS_BIRTH_DEF_GI_HIRSCHPRUNG,
                Feat.BMS_BIRTH_DEF_GI_INTEST_MALROT,
                Feat.BMS_BIRTH_DEF_GI_PYLOR_STEN,
                Feat.BMS_BIRTH_DEF_THORAC,
                Feat.BMS_BIRTH_DEF_THORAC_CDH,
                Feat.BMS_BIRTH_DEF_THORAC_HEART,
                Feat.BMS_BIRTH_DEF_THORAC_LUNG,
                Feat.BMS_BIRTH_DEF_UROGEN,
                Feat.BMS_BIRTH_DEF_UROGEN_HYPOSPAD,
                Feat.BMS_BIRTH_DEF_UROGEN_RENAL,
                Feat.BMS_BIRTH_DEF_UROGEN_RENAL_AGEN,
                Feat.BMS_BIRTH_DEF_UROGEN_UTER_AGEN,
                Feat.BMS_BIRTH_ETOH_SUBST,
                Feat.BMS_BIRTH_OXYGEN,
                Feat.BMS_BIRTH_PG_INF,
                Feat.BMS_BIRTH_PREM,
                Feat.BMS_COG_MED,
                Feat.BMS_DEV_ID,
                Feat.BMS_DEV_LANG_DIS,
                Feat.BMS_DEV_LD,
                Feat.BMS_DEV_MOTOR,
                Feat.BMS_DEV_MUTISM,
                Feat.BMS_DEV_SOC_PRAG,
                Feat.BMS_DEV_SPEECH,
                Feat.BMS_EATING_DISORDER,
                Feat.BMS_ENCOPRES,
                Feat.BMS_ENURES,
                Feat.BMS_FEEDING_DX,
                Feat.BMS_GROWTH_LOW_WT,
                Feat.BMS_GROWTH_MACROCEPH,
                Feat.BMS_GROWTH_MICROCEPH,
                Feat.BMS_GROWTH_OBES,
                Feat.BMS_GROWTH_SHORT,
                Feat.BMS_MED_COND_BIRTH,
                Feat.BMS_MED_COND_BIRTH_DEF,
                Feat.BMS_MED_COND_GROWTH,
                Feat.BMS_MED_COND_NEURO,
                Feat.BMS_MED_COND_VISAUD,
                Feat.BMS_MOOD_ANX,
                Feat.BMS_MOOD_BIPOL,
                Feat.BMS_MOOD_DEP,
                Feat.BMS_MOOD_DMD,
                Feat.BMS_MOOD_HOARD,
                Feat.BMS_MOOD_OCD,
                Feat.BMS_MOOD_OR_ANX,
                Feat.BMS_MOOD_SEP_ANX,
                Feat.BMS_MOOD_SOC_ANX,
                Feat.BMS_NEURO_INF,
                Feat.BMS_NEURO_LEAD,
                Feat.BMS_NEURO_SZ,
                Feat.BMS_NEURO_TBI,
                Feat.BMS_PERS_DIS,
                Feat.BMS_SCHIZ,
                Feat.BMS_SLEEP_DX,
                Feat.BMS_SLEEP_EAT_TOILET,
                Feat.BMS_TICS,
                Feat.BMS_VISAUD_BLIND,
                Feat.BMS_VISAUD_CATAR,
                Feat.BMS_VISAUD_DEAF,
                Feat.BMS_VISAUD_STRAB,
                Feat.CBCL_6_18_CLOSE_FRIENDS,
                Feat.CBCL_6_18_CONTACT_FRIENDS_OUTSIDE_SCHOOL,
                Feat.CBCL_6_18_GETS_ALONG_SIBLINGS,
                Feat.CBCL_6_18_GETS_ALONG_OTHER_KIDS,
                Feat.CBCL_6_18_BEHAVE_WITH_PARENTS,
                Feat.CBCL_6_18_PLAY_WORK_ALONE,
                Feat.CBCL_6_18_READING_ENG_LANGUAGE,
                Feat.CBCL_6_18_HISTORY_SOCIAL_STUDIES,
                Feat.CBCL_6_18_ARITHMETIC_MATH,
                Feat.CBCL_6_18_SCIENCE,
                Feat.CBCL_6_18_Q001_ACTS_YOUNG,
                Feat.CBCL_6_18_Q002_DRINKS_ALCOHOL,
                Feat.CBCL_6_18_Q003_ARGUES,
                Feat.CBCL_6_18_Q004_FAILS_TO_FINISH,
                Feat.CBCL_6_18_Q005_VERY_LITTLE_ENJOYMENT,
                Feat.CBCL_6_18_Q006_BOWEL_MOVEMENTS_OUTSIDE,
                Feat.CBCL_6_18_Q007_BRAG_BOAST,
                Feat.CBCL_6_18_Q008_CONCENTRATE,
                Feat.CBCL_6_18_Q009_OBSESSIONS,
                Feat.CBCL_6_18_Q010_RESTLESS,
                Feat.CBCL_6_18_Q011_TOO_DEPENDENT,
                Feat.CBCL_6_18_Q012_LONELINESS,
                Feat.CBCL_6_18_Q013_CONFUSED,
                Feat.CBCL_6_18_Q014_CRIES_A_LOT,
                Feat.CBCL_6_18_Q015_CRUELTY_ANIMALS,
                Feat.CBCL_6_18_Q016_CRUELTY_OTHERS,
                Feat.CBCL_6_18_Q017_DAYDREAMS,
                Feat.CBCL_6_18_Q018_HARMS_SELF,
                Feat.CBCL_6_18_Q019_DEMANDS_ATTENTION,
                Feat.CBCL_6_18_Q020_DESTROYS_OWN_THINGS,
                Feat.CBCL_6_18_Q021_DESTROYS_OTHERS_THINGS,
                Feat.CBCL_6_18_Q022_DISOBEDIENT_HOME,
                Feat.CBCL_6_18_Q023_DISOBEDIENT_SCHOOL,
                Feat.CBCL_6_18_Q024_DOESNT_EAT_WELL,
                Feat.CBCL_6_18_Q025_DOESNT_GET_ALONG_OTHERS,
                Feat.CBCL_6_18_Q026_GUILTY_MISBEHAVING,
                Feat.CBCL_6_18_Q027_JEALOUS,
                Feat.CBCL_6_18_Q028_BREAKS_RULES,
                Feat.CBCL_6_18_Q029_FEARS,
                Feat.CBCL_6_18_Q030_FEARS_SCHOOL,
                Feat.CBCL_6_18_Q031_FEARS_BAD,
                Feat.CBCL_6_18_Q032_PERFECT,
                Feat.CBCL_6_18_Q033_FEARS_NO_ONE_LOVES,
                Feat.CBCL_6_18_Q034_OUT_TO_GET,
                Feat.CBCL_6_18_Q035_FEELS_WORTHLESS,
                Feat.CBCL_6_18_Q036_ACCIDENT_PRONE,
                Feat.CBCL_6_18_Q037_FIGHTS,
                Feat.CBCL_6_18_Q038_TEASED,
                Feat.CBCL_6_18_Q039_HANGS_AROUND_TROUBLE,
                Feat.CBCL_6_18_Q040_HEARS_VOICES,
                Feat.CBCL_6_18_Q041_IMPULSIVE,
                Feat.CBCL_6_18_Q042_RATHER_ALONE,
                Feat.CBCL_6_18_Q043_LYING,
                Feat.CBCL_6_18_Q044_BITES_FINGERNAILS,
                Feat.CBCL_6_18_Q045_NERVOUS_TENSE,
                Feat.CBCL_6_18_Q046_TWITCHING,
                Feat.CBCL_6_18_Q047_NIGHTMARES,
                Feat.CBCL_6_18_Q048_NOT_LIKED,
                Feat.CBCL_6_18_Q049_CONSTIPATED,
                Feat.CBCL_6_18_Q050_ANXIOUS,
                Feat.CBCL_6_18_Q051_DIZZY,
                Feat.CBCL_6_18_Q052_FEELS_TOO_GUILTY,
                Feat.CBCL_6_18_Q053_OVEREATING,
                Feat.CBCL_6_18_Q054_OVERTIRED,
                Feat.CBCL_6_18_Q055_OVERWEIGHT,
                Feat.CBCL_6_18_Q056_A_ACHES,
                Feat.CBCL_6_18_Q056_B_HEADACHE,
                Feat.CBCL_6_18_Q056_C_NAUSEA,
                Feat.CBCL_6_18_Q056_D_EYES,
                Feat.CBCL_6_18_Q056_E_RASHES,
                Feat.CBCL_6_18_Q056_F_STOMACHACHES,
                Feat.CBCL_6_18_Q056_G_VOMITING,
                Feat.CBCL_6_18_Q056_H_OTHER,
                Feat.CBCL_6_18_Q057_ATTACKS,
                Feat.CBCL_6_18_Q058_PICKS_SKIN,
                Feat.CBCL_6_18_Q059_SEX_PARTS_PUBLIC,
                Feat.CBCL_6_18_Q060_SEX_PARTS_TOO_MUCH,
                Feat.CBCL_6_18_Q061_POOR_WORK,
                Feat.CBCL_6_18_Q062_CLUMSY,
                Feat.CBCL_6_18_Q063_RATHER_OLDER_KIDS,
                Feat.CBCL_6_18_Q064_RATHER_YOUNGER_KIDS,
                Feat.CBCL_6_18_Q065_REFUSES_TO_TALK,
                Feat.CBCL_6_18_Q066_REPEATS_ACTS,
                Feat.CBCL_6_18_Q067_RUNS_AWAY_HOME,
                Feat.CBCL_6_18_Q068_SCREAMS_A_LOT,
                Feat.CBCL_6_18_Q069_SECRETIVE,
                Feat.CBCL_6_18_Q070_SEES_THINGS,
                Feat.CBCL_6_18_Q071_SELF_CONSCIOUS,
                Feat.CBCL_6_18_Q072_SETS_FIRES,
                Feat.CBCL_6_18_Q073_SEXUAL_PROBLEMS,
                Feat.CBCL_6_18_Q074_CLOWNING,
                Feat.CBCL_6_18_Q075_TOO_SHY,
                Feat.CBCL_6_18_Q076_SLEEPS_LESS,
                Feat.CBCL_6_18_Q077_SLEEPS_MORE,
                Feat.CBCL_6_18_Q078_EASILY_DISTRACTED,
                Feat.CBCL_6_18_Q079_SPEECH_PROBLEM,
                Feat.CBCL_6_18_Q080_STARES_BLANKLY,
                Feat.CBCL_6_18_Q081_STEALS_HOME,
                Feat.CBCL_6_18_Q082_STEALS_OUTSIDE,
                Feat.CBCL_6_18_Q083_STORES_MANY_THINGS,
                Feat.CBCL_6_18_Q084_STRANGE_BEHAVIOR,
                Feat.CBCL_6_18_Q085_STRANGE_IDEAS,
                Feat.CBCL_6_18_Q086_STUBBORN,
                Feat.CBCL_6_18_Q087_CHANGES_MOOD,
                Feat.CBCL_6_18_Q088_SULKS,
                Feat.CBCL_6_18_Q089_SUSPICIOUS,
                Feat.CBCL_6_18_Q090_OBSCENE_LANGUAGE,
                Feat.CBCL_6_18_Q091_TALKS_KILLING_SELF,
                Feat.CBCL_6_18_Q092_TALKS_WALKS_SLEEP,
                Feat.CBCL_6_18_Q093_TALKS_TOO_MUCH,
                Feat.CBCL_6_18_Q094_TEASES_A_LOT,
                Feat.CBCL_6_18_Q095_TANTRUMS,
                Feat.CBCL_6_18_Q096_THINKS_SEX_TOO_MUCH,
                Feat.CBCL_6_18_Q097_THREATENS,
                Feat.CBCL_6_18_Q098_THUMB_SUCKING,
                Feat.CBCL_6_18_Q099_TOBACCO,
                Feat.CBCL_6_18_Q100_TROUBLE_SLEEPING,
                Feat.CBCL_6_18_Q101_SKIPS_SCHOOL,
                Feat.CBCL_6_18_Q102_UNDERACTIVE,
                Feat.CBCL_6_18_Q103_UNHAPPY,
                Feat.CBCL_6_18_Q104_UNUSUALLY_LOUD,
                Feat.CBCL_6_18_Q105_DRUGS,
                Feat.CBCL_6_18_Q106_VANDALISM,
                Feat.CBCL_6_18_Q107_WETS_SELF,
                Feat.CBCL_6_18_Q108_WETS_BED,
                Feat.CBCL_6_18_Q109_WHINING,
                Feat.CBCL_6_18_Q110_WISHES_TO_BE_OPP_SEX,
                Feat.CBCL_6_18_Q111_WITHDRAWN,
                Feat.CBCL_6_18_Q112_WORRIES,
                Feat.CBCL_6_18_ANXIOUS_DEPRESSED_RAW_SCORE,
                Feat.CBCL_6_18_ANXIOUS_DEPRESSED_T_SCORE,
                Feat.CBCL_6_18_ANXIOUS_DEPRESSED_PERCENTILE,
                Feat.CBCL_6_18_ANXIOUS_DEPRESSED_RANGE,
                Feat.CBCL_6_18_WITHDRAWN_DEPRESSED_RAW_SCORE,
                Feat.CBCL_6_18_WITHDRAWN_DEPRESSED_T_SCORE,
                Feat.CBCL_6_18_WITHDRAWN_DEPRESSED_PERCENTILE,
                Feat.CBCL_6_18_WITHDRAWN_DEPRESSED_RANGE,
                Feat.CBCL_6_18_SOMATIC_COMPLAINTS_RAW_SCORE,
                Feat.CBCL_6_18_SOMATIC_COMPLAINTS_T_SCORE,
                Feat.CBCL_6_18_SOMATIC_COMPLAINTS_PERCENTILE,
                Feat.CBCL_6_18_SOMATIC_COMPLAINTS_RANGE,
                Feat.CBCL_6_18_SOCIAL_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_SOCIAL_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_SOCIAL_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_SOCIAL_PROBLEMS_RANGE,
                Feat.CBCL_6_18_THOUGHT_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_THOUGHT_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_THOUGHT_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_THOUGHT_PROBLEMS_RANGE,
                Feat.CBCL_6_18_ATTENTION_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_ATTENTION_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_ATTENTION_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_ATTENTION_PROBLEMS_RANGE,
                Feat.CBCL_6_18_RULE_BREAKING_BEHAVIOR_RAW_SCORE,
                Feat.CBCL_6_18_RULE_BREAKING_BEHAVIOR_T_SCORE,
                Feat.CBCL_6_18_RULE_BREAKING_BEHAVIOR_PERCENTILE,
                Feat.CBCL_6_18_RULE_BREAKING_BEHAVIOR_RANGE,
                Feat.CBCL_6_18_AGGRESSIVE_BEHAVIOR_RAW_SCORE,
                Feat.CBCL_6_18_AGGRESSIVE_BEHAVIOR_T_SCORE,
                Feat.CBCL_6_18_AGGRESSIVE_BEHAVIOR_PERCENTILE,
                Feat.CBCL_6_18_AGGRESSIVE_BEHAVIOR_RANGE,
                Feat.CBCL_6_18_INTERNALIZING_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_INTERNALIZING_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_INTERNALIZING_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_INTERNALIZING_PROBLEMS_RANGE,
                Feat.CBCL_6_18_EXTERNALIZING_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_EXTERNALIZING_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_EXTERNALIZING_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_EXTERNALIZING_PROBLEMS_RANGE,
                Feat.CBCL_6_18_TOTAL_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_TOTAL_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_TOTAL_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_TOTAL_PROBLEMS_RANGE,
                Feat.CBCL_6_18_OTHER_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_OBSESSIVE_COMPULSIVE_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_OBSESSIVE_COMPULSIVE_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_OBSESSIVE_COMPULSIVE_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_OBSESSIVE_COMPULSIVE_PROBLEMS_RANGE,
                Feat.CBCL_6_18_SLUGGISH_COGNITIVE_TEMPO_RAW_SCORE,
                Feat.CBCL_6_18_SLUGGISH_COGNITIVE_TEMPO_T_SCORE,
                Feat.CBCL_6_18_SLUGGISH_COGNITIVE_TEMPO_PERCENTILE,
                Feat.CBCL_6_18_SLUGGISH_COGNITIVE_TEMPO_RANGE,
                Feat.CBCL_6_18_STRESS_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_STRESS_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_STRESS_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_STRESS_PROBLEMS_RANGE,
                Feat.CBCL_6_18_DSM5_CONDUCT_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_DSM5_CONDUCT_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_DSM5_CONDUCT_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_DSM5_CONDUCT_PROBLEMS_RANGE,
                Feat.CBCL_6_18_DSM5_SOMATIC_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_DSM5_SOMATIC_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_DSM5_SOMATIC_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_DSM5_SOMATIC_PROBLEMS_RANGE,
                Feat.CBCL_6_18_DSM5_OPPOSITIONAL_DEFIANT_RAW_SCORE,
                Feat.CBCL_6_18_DSM5_OPPOSITIONAL_DEFIANT_T_SCORE,
                Feat.CBCL_6_18_DSM5_OPPOSITIONAL_DEFIANT_PERCENTILE,
                Feat.CBCL_6_18_DSM5_OPPOSITIONAL_DEFIANT_RANGE,
                Feat.CBCL_6_18_DSM5_ATTENTION_DEFICIT_HYPERACTIVITY_RAW_SCORE,
                Feat.CBCL_6_18_DSM5_ATTENTION_DEFICIT_HYPERACTIVITY_T_SCORE,
                Feat.CBCL_6_18_DSM5_ATTENTION_DEFICIT_HYPERACTIVITY_PERCENTILE,
                Feat.CBCL_6_18_DSM5_ATTENTION_DEFICIT_HYPERACTIVITY_RANGE,
                Feat.CBCL_6_18_DSM5_ANXIETY_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_DSM5_ANXIETY_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_DSM5_ANXIETY_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_DSM5_ANXIETY_PROBLEMS_RANGE,
                Feat.CBCL_6_18_DSM5_DEPRESSIVE_PROBLEMS_RAW_SCORE,
                Feat.CBCL_6_18_DSM5_DEPRESSIVE_PROBLEMS_T_SCORE,
                Feat.CBCL_6_18_DSM5_DEPRESSIVE_PROBLEMS_PERCENTILE,
                Feat.CBCL_6_18_DSM5_DEPRESSIVE_PROBLEMS_RANGE,
                Feat.DCDQ_Q01_THROW_BALL,
                Feat.DCDQ_Q02_CATCH_BALL,
                Feat.DCDQ_Q03_HIT_BALL,
                Feat.DCDQ_Q04_JUMP_OBSTACLES,
                Feat.DCDQ_Q05_RUN_FAST_SIMILAR,
                Feat.DCDQ_Q06_PLAN_MOTOR_ACTIVITY,
                Feat.DCDQ_Q07_PRINTING_WRITING_DRAWING_FAST,
                Feat.DCDQ_Q08_PRINTING_LETTERS_LEGIBLE,
                Feat.DCDQ_Q09_APPROPRIATE_TENSION_PRINTING_WRITING,
                Feat.DCDQ_Q10_CUTS_PICTURES_SHAPES,
                Feat.DCDQ_Q11_LIKES_SPORTS_MOTORS_SKILLS,
                Feat.DCDQ_Q12_LEARNS_NEW_MOTOR_TASKS,
                Feat.DCDQ_Q13_QUICK_COMPETENT_TIDYING_UP,
                Feat.DCDQ_Q14_BULL_IN_CHINA_SHOP,
                Feat.DCDQ_Q15_FATIGUE_EASILY,
                Feat.DCDQ_CONTROL_DURING_MOVEMENT,
                Feat.DCDQ_FINAL_SCORE,
                Feat.DCDQ_FINE_MOTOR_HANDWRITING,
                Feat.DCDQ_GENERAL_COORDINATION,
                Feat.DCDQ_MOTOR_ABLE,
                Feat.IR_SEX,
                Feat.IR_ASD,
                Feat.IR_AGE_AT_REGISTRATION_YEARS,
                Feat.RBSR_Q01_WHOLE_BODY,
                Feat.RBSR_Q02_HEAD,
                Feat.RBSR_Q03_HAND_FINGER,
                Feat.RBSR_Q04_LOCOMOTION,
                Feat.RBSR_Q05_OBJECT_USAGE,
                Feat.RBSR_Q06_SENSORY,
                Feat.RBSR_Q07_HITS_SELF_BODY,
                Feat.RBSR_Q08_HITS_SELF_AGAINST_OBJECT,
                Feat.RBSR_Q09_HITS_SELF_WITH_OBJECT,
                Feat.RBSR_Q10_BITES_SELF,
                Feat.RBSR_Q11_PULLS,
                Feat.RBSR_Q12_RUBS,
                Feat.RBSR_Q13_INSERTS_FINGER,
                Feat.RBSR_Q14_SKIN_PICKING,
                Feat.RBSR_Q15_ARRANGING,
                Feat.RBSR_Q16_COMPLETE,
                Feat.RBSR_Q17_WASHING,
                Feat.RBSR_Q18_CHECKING,
                Feat.RBSR_Q19_COUNTING,
                Feat.RBSR_Q20_HOARDING,
                Feat.RBSR_Q21_REPEATING,
                Feat.RBSR_Q22_TOUCH_TAP,
                Feat.RBSR_Q23_EATING,
                Feat.RBSR_Q24_SLEEP,
                Feat.RBSR_Q25_SELF_CARE,
                Feat.RBSR_Q26_TRAVEL,
                Feat.RBSR_Q27_PLAY,
                Feat.RBSR_Q28_COMMUNICATION,
                Feat.RBSR_Q29_THINGS_SAME_PLACE,
                Feat.RBSR_Q30_OBJECTS,
                Feat.RBSR_Q31_BECOMES_UPSET,
                Feat.RBSR_Q32_INSISTS_WALKING,
                Feat.RBSR_Q33_INSISTS_SITTING,
                Feat.RBSR_Q34_DISLIKES_CHANGES,
                Feat.RBSR_Q35_INSISTS_DOOR,
                Feat.RBSR_Q36_LIKES_PIECE_MUSIC,
                Feat.RBSR_Q37_RESISTS_CHANGE,
                Feat.RBSR_Q38_INSISTS_ROUTINE,
                Feat.RBSR_Q39_INSISTS_TIME,
                Feat.RBSR_Q40_FASCINATION_SUBJECT,
                Feat.RBSR_Q41_STRONGLY_ATTACHED,
                Feat.RBSR_Q42_PREOCCUPATION,
                Feat.RBSR_Q43_FASCINATION_MOVEMENT,
                Feat.RBSR_I_STEREOTYPED_BEHAVIOR_SCORE,
                Feat.RBSR_II_SELF_INJURIOUS_SCORE,
                Feat.RBSR_III_COMPULSIVE_BEHAVIOR_SCORE,
                Feat.RBSR_IV_RITUALISTIC_BEHAVIOR_SCORE,
                Feat.RBSR_V_SAMENESS_BEHAVIOR_SCORE,
                Feat.RBSR_VI_RESTRICTED_BEHAVIOR_SCORE,
                Feat.RBSR_OVERALL_SCORE,
                Feat.RBSR_OVERALL_NUMBER_ITEMS,
                Feat.RBSR_TOTAL_FINAL_SCORE,
                Feat.SCQ_Q01_PHRASES,
                Feat.SCQ_Q02_CONVERSATION,
                Feat.SCQ_Q03_ODD_PHRASE,
                Feat.SCQ_Q04_INAPPROPRIATE_QUESTION,
                Feat.SCQ_Q05_PRONOUNS_MIXED,
                Feat.SCQ_Q06_INVENTED_WORDS,
                Feat.SCQ_Q07_SAME_OVER,
                Feat.SCQ_Q08_PARTICULAR_WAY,
                Feat.SCQ_Q09_EXPRESSIONS_APPROPRIATE,
                Feat.SCQ_Q10_HAND_TOOL,
                Feat.SCQ_Q11_INTEREST_PREOCCUPY,
                Feat.SCQ_Q12_PARTS_OBJECT,
                Feat.SCQ_Q13_INTERESTS_INTENSITY,
                Feat.SCQ_Q14_SENSES,
                Feat.SCQ_Q15_ODD_WAYS,
                Feat.SCQ_Q16_COMPLICATED_MOVEMENTS,
                Feat.SCQ_Q17_INJURED_DELIBERATELY,
                Feat.SCQ_Q18_OBJECTS_CARRY,
                Feat.SCQ_Q19_BEST_FRIEND,
                Feat.SCQ_Q20_TALK_FRIENDLY,
                Feat.SCQ_Q21_COPY_YOU,
                Feat.SCQ_Q22_POINT_THINGS,
                Feat.SCQ_Q23_GESTURES_WANTED,
                Feat.SCQ_Q24_NOD_HEAD,
                Feat.SCQ_Q25_SHAKE_HEAD,
                Feat.SCQ_Q26_LOOK_DIRECTLY,
                Feat.SCQ_Q27_SMILE_BACK,
                Feat.SCQ_Q28_THINGS_INTERESTED,
                Feat.SCQ_Q29_SHARE,
                Feat.SCQ_Q30_JOIN_ENJOYMENT,
                Feat.SCQ_Q31_COMFORT,
                Feat.SCQ_Q32_HELP_ATTENTION,
                Feat.SCQ_Q33_RANGE_EXPRESSIONS,
                Feat.SCQ_Q34_COPY_ACTIONS,
                Feat.SCQ_Q35_MAKE_BELIEVE,
                Feat.SCQ_Q36_SAME_AGE,
                Feat.SCQ_Q37_RESPOND_POSITIVELY,
                Feat.SCQ_Q38_PAY_ATTENTION,
                Feat.SCQ_Q39_IMAGINATIVE_GAMES,
                Feat.SCQ_Q40_COOPERATIVELY_GAMES,
                Feat.SCQ_FINAL_SCORE,
                Feat.SRS_2_SA_Q01_FIDGETY,
                Feat.SRS_2_SA_Q02_EXPRESSIONS_MATCH,
                Feat.SRS_2_SA_Q03_SELF_CONFIDENT,
                Feat.SRS_2_SA_Q04_UNDER_STRESS,
                Feat.SRS_2_SA_Q05_RECOGNIZE_TAKE_ADVANTAGE,
                Feat.SRS_2_SA_Q06_RATHER_ALONE,
                Feat.SRS_2_SA_Q07_AWARE_OTHERS,
                Feat.SRS_2_SA_Q08_BEHAVES_STRANGE,
                Feat.SRS_2_SA_Q09_CLINGS_DEPENDENT,
                Feat.SRS_2_SA_Q10_THINGS_LITERALLY,
                Feat.SRS_2_SA_Q11_SELF_CONFIDENCE,
                Feat.SRS_2_SA_Q12_COMMUNICATE_FEELINGS,
                Feat.SRS_2_SA_Q13_TURN_TAKING,
                Feat.SRS_2_SA_Q14_WELL_COORDINATED,
                Feat.SRS_2_SA_Q15_MEANING_TONE,
                Feat.SRS_2_SA_Q16_EYE_CONTACT,
                Feat.SRS_2_SA_Q17_RECOGNIZES_UNFAIR,
                Feat.SRS_2_SA_Q18_MAKING_FRIENDS,
                Feat.SRS_2_SA_Q19_FRUSTRATED_CONVERSATION,
                Feat.SRS_2_SA_Q20_SENSORY_INTERESTS,
                Feat.SRS_2_SA_Q21_IMITATE_OTHERS,
                Feat.SRS_2_SA_Q22_PLAYS_APPROPRIATELY,
                Feat.SRS_2_SA_Q23_JOIN_ACTIVITIES,
                Feat.SRS_2_SA_Q24_CHANGES_ROUTINE,
                Feat.SRS_2_SA_Q25_SAME_WAVELENGTH,
                Feat.SRS_2_SA_Q26_COMFORTS_OTHERS,
                Feat.SRS_2_SA_Q27_AVOIDS_SOCIAL,
                Feat.SRS_2_SA_Q28_SAME_THING,
                Feat.SRS_2_SA_Q29_REGARDED_ODD,
                Feat.SRS_2_SA_Q30_UPSET,
                Feat.SRS_2_SA_Q31_MIND_SOMETHING,
                Feat.SRS_2_SA_Q32_HYGIENE,
                Feat.SRS_2_SA_Q33_SOCIALLY_AWKWARD,
                Feat.SRS_2_SA_Q34_AVOIDS_PEOPLE,
                Feat.SRS_2_SA_Q35_NORMAL_CONVERSATIONS,
                Feat.SRS_2_SA_Q36_RELATING_ADULTS,
                Feat.SRS_2_SA_Q37_RELATING_PEERS,
                Feat.SRS_2_SA_Q38_MOOD_OTHERS,
                Feat.SRS_2_SA_Q39_NARROW_INTERESTS,
                Feat.SRS_2_SA_Q40_IMAGINATIVE,
                Feat.SRS_2_SA_Q41_WANDERS_AIMLESSLY,
                Feat.SRS_2_SA_Q42_SENSITIVE_SOUNDS,
                Feat.SRS_2_SA_Q43_SEPARATES_CAREGIVERS,
                Feat.SRS_2_SA_Q44_EVENTS_RELATED,
                Feat.SRS_2_SA_Q45_ATTENTION_OTHERS,
                Feat.SRS_2_SA_Q46_OVERLY_SERIOUS,
                Feat.SRS_2_SA_Q47_LAUGHS_INAPPROPRIATELY,
                Feat.SRS_2_SA_Q48_HUMOR,
                Feat.SRS_2_SA_Q49_FEW_TASKS,
                Feat.SRS_2_SA_Q50_REPETITIVE_BEHAVIOR,
                Feat.SRS_2_SA_Q51_QUESTIONS_DIRECTLY,
                Feat.SRS_2_SA_Q52_TOO_LOUD,
                Feat.SRS_2_SA_Q53_UNUSUAL_TONE,
                Feat.SRS_2_SA_Q54_PEOPLE_OBJECTS,
                Feat.SRS_2_SA_Q55_KNOWS_TOO_CLOSE,
                Feat.SRS_2_SA_Q56_WALKS_BETWEEN,
                Feat.SRS_2_SA_Q57_TEASED,
                Feat.SRS_2_SA_Q58_CONCENTRATES_PARTS,
                Feat.SRS_2_SA_Q59_OVERLY_SUSPICIOUS,
                Feat.SRS_2_SA_Q60_EMOTIONALLY_DISTANT,
                Feat.SRS_2_SA_Q61_INFLEXIBLE,
                Feat.SRS_2_SA_Q62_ILLOGICAL_REASONS,
                Feat.SRS_2_SA_Q63_TOUCHS_UNUSUAL,
                Feat.SRS_2_SA_Q64_TENSE_SOCIAL,
                Feat.SRS_2_SA_Q65_STARES_SPACE,
                Feat.SRS_2_SA_RRB_T_SCORE,
                Feat.SRS_2_SA_RRB_RAW_SCORE,
                Feat.SRS_2_SA_AWR_T_SCORE,
                Feat.SRS_2_SA_AWR_RAW_SCORE,
                Feat.SRS_2_SA_SOC_COG_T_SCORE,
                Feat.SRS_2_SA_SOC_COG_RAW_SCORE,
                Feat.SRS_2_SA_COM_T_SCORE,
                Feat.SRS_2_SA_COM_RAW_SCORE,
                Feat.SRS_2_SA_MOT_T_SCORE,
                Feat.SRS_2_SA_MOT_RAW_SCORE,
                Feat.SRS_2_SA_SCI_T_SCORE,
                Feat.SRS_2_SA_SCI_RAW_SCORE,
                Feat.SRS_2_SA_TOTAL_T_SCORE,
                Feat.SRS_2_SA_TOTAL_RAW_SCORE,
                Feat.SRS_2_SA_TOTAL_SCORE_RANGE,
            ],
        )

        def convert_one_or_null_to_bool(df: pd.DataFrame) -> pd.DataFrame:
            cols_to_convert = [
                col for col in df.columns if df[col].dropna().isin([1]).all()
            ]
            df[cols_to_convert] = df[cols_to_convert].notna().astype(bool)
            return df

        def convert_object_to_one_hot_vector(df: pd.DataFrame) -> pd.DataFrame:
            df = pd.get_dummies(
                df, columns=list(df.select_dtypes(include="object").columns)
            )
            return df

        def convert_binary_like_to_bool(df: pd.DataFrame) -> pd.DataFrame:
            binary_like_cols = [
                col for col in df.columns if df[col].dropna().isin([0, 1]).all()
            ]
            df[binary_like_cols] = df[binary_like_cols].astype("boolean")
            return df

        df = df.replace({"-": None})

        df[Feat.IR_ASD.col] = df[Feat.IR_ASD.col].astype("boolean")

        df = df[df[Feat.IR_ASD.col] == True]
        df = df[df[Feat.IR_AGE_AT_REGISTRATION_YEARS.col] >= 4]
        df = df[df[Feat.IR_AGE_AT_REGISTRATION_YEARS.col] <= 18]

        df = convert_one_or_null_to_bool(df)
        df = convert_binary_like_to_bool(df)
        df = convert_object_to_one_hot_vector(df)

        df.to_feather(df_filepath)
    else:
        df = pd.read_feather(df_filepath)

    return df


def plot_data_retention_vs_threshold(
    df,
    increment=0.01,
    figsize=None,
    threshold_lower=0.4,
    threshold_upper=0.95,
    threshold_step=0.05,
):
    thresholds = np.arange(0, 1 + increment, increment)
    t_range = np.arange(
        threshold_lower, threshold_upper + threshold_step, threshold_step
    )
    shapes = []

    for threshold in tqdm(thresholds, desc="Analyzing thresholds"):
        shapes.append(df.loc[:, df.isna().mean() < threshold].dropna(axis=0).shape)

    rows, cols = zip(*shapes)

    px = 1 / matplotlib.rcParams["figure.dpi"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1812 * px, 567 * px))

    ax1.plot(thresholds, rows, color="#426665")
    ax1.set_ylabel("Subject count")
    ax1.set_xlabel("Missing Value Threshold")
    ax1.grid(True, which="both", axis="both", alpha=0.3)
    for t in t_range:
        ax1.axvline(t, linestyle="--", alpha=0.5, color="#426665")
        ax2.axvline(t, linestyle="--", alpha=0.5, color="#426665")

    ax2.plot(thresholds, cols, color="#426665")
    ax2.set_xlabel("Missing Value Threshold")
    ax2.set_ylabel("Feature count")
    ax2.grid(True, which="both", axis="both", alpha=0.3)

    plt.suptitle("Data Retention against Missing Value Threshold, $t$")
    fig.tight_layout()

    return fig
