from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing_extensions import Annotated

from spark import Inst, SPARK

app = typer.Typer()


@app.command()
def linear_regression(
    spark_pathname: Annotated[str, typer.Argument(envvar="SPARK_PATHNAME")] = ".",
    cache_pathname: Annotated[str, typer.Argument(envvar="CACHE_PATHNAME")] = ".",
    n_splits: int = 10,
):
    def calc_mean_ci(
        array: list[float],
    ) -> tuple[np.floating, np.floating, np.floating]:
        n_bootstrap = 1000
        rng = np.random.default_rng()
        boot_means = [
            np.mean(rng.choice(array, size=len(array), replace=True))
            for _ in range(n_bootstrap)
        ]
        return (
            np.mean(array),
            np.percentile(boot_means, 2.5),
            np.percentile(boot_means, 97.5),
        )

    cache_dir = Path(cache_pathname) / "linear_regression"
    cache_dir.mkdir(exist_ok=True, parents=True)
    filepath = cache_dir / "results.npz"
    instruments = [
        i
        for i in Inst
        if i.final_score_feature is not None and i.question_features is not None
    ]
    n_instruments = len(instruments)

    ds = SPARK(
        spark_pathname=spark_pathname,
        instruments=instruments,
    )

    shape = (n_instruments, n_instruments)
    results = {
        "n": np.ndarray(shape=shape),
        "mean": np.ndarray(shape=shape),
        "ci_lower": np.ndarray(shape=shape),
        "ci_upper": np.ndarray(shape=shape),
    }

    if filepath.exists():
        file = np.load(filepath)

        for key in results:
            results[key] = file[key]
    else:
        with tqdm(total=len(instruments) * len(instruments)) as pbar:
            for x, x_inst in enumerate(instruments):
                for y, y_inst in enumerate(instruments):
                    pbar.set_postfix(
                        {
                            "x": x_inst.code,
                            "y": y_inst.code,
                        }
                    )

                    df = ds.join(
                        features=x_inst.question_features
                        + [y_inst.final_score_feature],
                        how="inner",
                    ).dropna()

                    questions = df.drop(columns=[y_inst.final_score_feature.col])
                    scores = df[y_inst.final_score_feature.col]

                    lr = LinearRegression()
                    scaler = StandardScaler()
                    pl = make_pipeline(scaler, lr)
                    kf = KFold(n_splits=n_splits, shuffle=True)

                    cv = cross_validate(
                        pl, questions, scores, cv=kf, scoring="r2", n_jobs=-1
                    )
                    test_scores = cv["test_score"]

                    mean, ci_lower, ci_upper = calc_mean_ci(test_scores)

                    results["n"][x, y] = len(df)
                    results["mean"][x, y] = mean
                    results["ci_lower"][x, y] = ci_lower
                    results["ci_upper"][x, y] = ci_upper

                    pbar.update(1)

        np.savez(filepath, **results)

    labels = {
        "CBCL_6_18": "CBCL/6–18",
        "CBCL_1_5": "CBCL/1–5",
    }
    inst_codes = [
        labels[inst.code] if inst.code in labels.keys() else inst.code
        for inst in instruments
    ]

    px = 1 / matplotlib.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(1920 * px, 810 * px))
    im = ax.imshow(
        results["mean"],
        vmin=0,
        vmax=1,
        cmap=LinearSegmentedColormap.from_list("mycmap", ["white", "#426665"]),
    )

    for x, col_x in enumerate(inst_codes):
        for y, col_y in enumerate(inst_codes):
            n = results["n"][x][y]
            mean = results["mean"][x][y]
            ci_lower = results["ci_lower"][x][y]
            ci_upper = results["ci_upper"][x][y]
            ax.text(
                y=y,
                x=x,
                s=(
                    f"${mean:.2f}$"
                    f"\n$[{ci_lower:.2f},\\ {ci_upper:.2f}]$"
                    f"\n${n / 1e3:.2g} \\times 10^3$"
                ),
                ha="center",
                va="center",
                color="white" if mean > 0.5 else "black",
            )

    ax.set_yticks(range(n_instruments), labels=inst_codes)
    ax.set_xticks(range(n_instruments), labels=inst_codes)
    ax.set_xlabel("Question")
    ax.set_ylabel("Score")
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("$R^2$")
    fig.tight_layout()
    fig.show()

    return
