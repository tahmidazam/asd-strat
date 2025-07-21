import typer

from .commands.initial_pheno import app as phenotypic_k_means_app
from .commands.regression import app as regression_app

app = typer.Typer()

app.add_typer(regression_app)
app.add_typer(phenotypic_k_means_app)


if __name__ == "__main__":
    app()
