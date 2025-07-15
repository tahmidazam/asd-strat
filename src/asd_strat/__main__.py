import typer

from .commands.regression import app as regression_app

app = typer.Typer()

app.add_typer(regression_app)


if __name__ == "__main__":
    app()
