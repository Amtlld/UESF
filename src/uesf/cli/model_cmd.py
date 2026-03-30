"""UESF CLI model commands: uesf model add/list/remove/edit."""

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
from uesf.managers.model_manager import ModelManager

console = Console()

model_app = typer.Typer(name="model", help="Manage models.", no_args_is_help=True)


def _get_manager() -> ModelManager:
    """Create and initialize ModelManager."""
    setup_logging()
    db = DatabaseManager()
    db.initialize()
    config = ConfigManager(db)
    return ModelManager(db, config)


@model_app.command("add")
def model_add(
    source: Path = typer.Argument(help="Path to the Python file containing the model class"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the model"),
    description: str | None = typer.Option(None, "--description", "-d", help="Model description"),
) -> None:
    """Add a global model from a source file."""
    try:
        manager = _get_manager()
        record = manager.add_global(source, name, description)
        console.print(f"[green]Added global model '{record['name']}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@model_app.command("list")
def model_list(
    show_obsolete: bool = typer.Option(False, "--show-obsolete", help="Include obsolete models"),
) -> None:
    """List all registered models."""
    try:
        manager = _get_manager()
        models = manager.list(show_obsolete=show_obsolete)

        if not models:
            console.print("[dim]No models registered.[/dim]")
            return

        table = Table(title="Models", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Obsolete", justify="center")
        table.add_column("Description")

        for m in models:
            table.add_row(
                m["name"],
                m["model_type"],
                "Yes" if m["is_obsolete"] else "No",
                m.get("description") or "",
            )

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@model_app.command("remove")
def model_remove(
    name: str = typer.Argument(help="Name of the model to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a model."""
    try:
        manager = _get_manager()
        manager.get(name)  # Verify it exists

        if not yes and not typer.confirm(f"Remove model '{name}'?"):
            console.print("[dim]Cancelled.[/dim]")
            return

        manager.remove(name)
        console.print(f"[green]Removed model '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@model_app.command("edit")
def model_edit(
    name: str = typer.Argument(help="Name of the model to edit"),
    description: str | None = typer.Option(None, "--description", "-d", help="New description"),
) -> None:
    """Edit model metadata."""
    try:
        manager = _get_manager()

        if description is None:
            console.print("[dim]No fields specified to edit.[/dim]")
            return

        manager.edit(name, description=description)
        console.print(f"[green]Updated model '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)
