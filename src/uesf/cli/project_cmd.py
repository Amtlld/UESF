"""UESF CLI project commands: uesf project init/info."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from uesf.cli.app import _format_uesf_error
from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import UESFException
from uesf.core.logging import setup_logging
from uesf.managers.project_manager import ProjectManager

console = Console()

project_app = typer.Typer(name="project", help="Manage projects.", no_args_is_help=True)


def _get_manager() -> ProjectManager:
    """Create and initialize ProjectManager."""
    setup_logging()
    db = DatabaseManager()
    db.initialize()
    config = ConfigManager(db)
    return ProjectManager(db, config)


@project_app.command("init")
def project_init(
    path: Path = typer.Argument(".", help="Directory to initialize as a project"),
) -> None:
    """Initialize a new UESF project directory."""
    try:
        manager = _get_manager()
        yml_path = manager.init(path)
        console.print(f"[green]Project initialized at '{yml_path.parent}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@project_app.command("info")
def project_info(
    path: Path = typer.Argument(".", help="Project directory"),
) -> None:
    """Show project information."""
    try:
        manager = _get_manager()
        info = manager.info(path)

        table = Table(title=f"Project: {info['project_name']}", show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Name", info["project_name"])
        table.add_row("Description", info["description"] or "(none)")
        table.add_row("Directory", info["project_dir"])
        table.add_row("Preprocessed Datasets", ", ".join(info["preprocessed_datasets"]) or "(none)")
        table.add_row("Raw Datasets", ", ".join(info["raw_datasets"]) or "(none)")
        table.add_row("Models", ", ".join(info["models"]) or "(none)")
        table.add_row("Trainers", ", ".join(info["trainers"]) or "(none)")
        table.add_row("Metrics", ", ".join(info["metrics"]) or "(none)")

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)
