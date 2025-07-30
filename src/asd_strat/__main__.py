import matplotlib
import pandas as pd
import typer
from pyfonts import load_google_font

from .commands.initial_pheno import app as phenotypic_k_means_app
from .commands.paper_reproduction import app as paper_reproduction_app
from .commands.regression import app as regression_app

app = typer.Typer()

app.add_typer(regression_app)
app.add_typer(phenotypic_k_means_app)
app.add_typer(paper_reproduction_app)


if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)

    font = load_google_font("Geist")
    matplotlib.rcParams["font.family"] = font.get_name()
    matplotlib.rcParams["font.size"] = 24

    app()
