"""UESF CLI metric commands: uesf metric add/list/remove/edit."""

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
from uesf.managers.metric_manager import MetricManager

console = Console()

metric_app = typer.Typer(name="metric", help="Manage evaluation metrics.", no_args_is_help=True)


def _get_manager() -> MetricManager:
    """Create and initialize MetricManager."""
    setup_logging()
    db = DatabaseManager()
    db.initialize()
    config = ConfigManager(db)
    return MetricManager(db, config)


@metric_app.command("add")
def metric_add(
    source: Path = typer.Argument(help="Path to the Python file containing the metric function"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the metric"),
    description: str | None = typer.Option(None, "--description", "-d", help="Metric description"),
) -> None:
    """Add a global metric from a source file."""
    try:
        manager = _get_manager()
        record = manager.add_global(source, name, description)
        console.print(f"[green]Added global metric '{record['name']}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@metric_app.command("list")
def metric_list(
    show_obsolete: bool = typer.Option(False, "--show-obsolete", help="Include obsolete metrics"),
) -> None:
    """List all registered metrics."""
    try:
        manager = _get_manager()
        metrics = manager.list(show_obsolete=show_obsolete)

        if not metrics:
            console.print("[dim]No metrics registered.[/dim]")
            return

        table = Table(title="Metrics", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Obsolete", justify="center")
        table.add_column("Description")

        for m in metrics:
            table.add_row(
                m["name"],
                m["metric_type"],
                "Yes" if m["is_obsolete"] else "No",
                m.get("description") or "",
            )

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@metric_app.command("remove")
def metric_remove(
    name: str = typer.Argument(help="Name of the metric to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a metric."""
    try:
        manager = _get_manager()
        manager.get(name)  # Verify it exists

        if not yes and not typer.confirm(f"Remove metric '{name}'?"):
            console.print("[dim]Cancelled.[/dim]")
            return

        manager.remove(name)
        console.print(f"[green]Removed metric '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@metric_app.command("edit")
def metric_edit(
    name: str = typer.Argument(help="Name of the metric to edit"),
    description: str | None = typer.Option(None, "--description", "-d", help="New description"),
) -> None:
    """Edit metric metadata."""
    try:
        manager = _get_manager()

        if description is None:
            console.print("[dim]No fields specified to edit.[/dim]")
            return

        manager.edit(name, description=description)
        console.print(f"[green]Updated metric '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)
