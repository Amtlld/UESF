"""UESF CLI main application."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

import uesf

console = Console(stderr=True)

app = typer.Typer(
    name="uesf",
    help="Universal EEG Study Framework - CLI for EEG deep learning research management.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"uesf {uesf.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Universal EEG Study Framework."""


# Register sub-apps (imported lazily to avoid circular imports)
def _register_sub_apps() -> None:
    from uesf.cli.config_cmd import config_app
    from uesf.cli.data_cmd import data_app
    from uesf.cli.experiment_cmd import experiment_app
    from uesf.cli.metric_cmd import metric_app
    from uesf.cli.model_cmd import model_app
    from uesf.cli.project_cmd import project_app
    from uesf.cli.trainer_cmd import trainer_app

    app.add_typer(config_app, name="config")
    app.add_typer(data_app, name="data")
    app.add_typer(model_app, name="model")
    app.add_typer(trainer_app, name="trainer")
    app.add_typer(metric_app, name="metric")
    app.add_typer(project_app, name="project")
    app.add_typer(experiment_app, name="experiment")


_register_sub_apps()
