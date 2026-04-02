"""UESF CLI config commands: uesf config set/show."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from uesf.cli.app import _format_uesf_error
from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import UESFException
from uesf.core.logging import setup_logging

console = Console()

config_app = typer.Typer(
    name="config",
    help="View and modify UESF global configuration.",
    no_args_is_help=True,
)


def _get_manager() -> ConfigManager:
    """Create and initialize ConfigManager with default paths."""
    setup_logging()
    db = DatabaseManager()
    db.initialize()
    return ConfigManager(db)


@config_app.command("show")
def config_show() -> None:
    """Display current effective global configuration."""
    try:
        manager = _get_manager()
        config = manager.get_all()

        table = Table(title="UESF Global Configuration", show_header=True)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        for key in sorted(config.keys()):
            table.add_row(key, str(config[key]))

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(help="Config key to set"),
    value: str = typer.Argument(help="Config value"),
) -> None:
    """Set a global configuration value (writes to <uesf-home>/config.yml)."""
    try:
        manager = _get_manager()
        manager.set(key, value)
        console.print(f"[green]Config '{key}' set to '{value}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)
