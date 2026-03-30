"""UESF CLI trainer commands: uesf trainer add/list/remove/edit."""

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
from uesf.managers.trainer_manager import TrainerManager

console = Console()

trainer_app = typer.Typer(name="trainer", help="Manage trainers.", no_args_is_help=True)


def _get_manager() -> TrainerManager:
    """Create and initialize TrainerManager."""
    setup_logging()
    db = DatabaseManager()
    db.initialize()
    config = ConfigManager(db)
    return TrainerManager(db, config)


@trainer_app.command("add")
def trainer_add(
    source: Path = typer.Argument(help="Path to the Python file containing the trainer class"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the trainer"),
    description: str | None = typer.Option(None, "--description", "-d", help="Trainer description"),
) -> None:
    """Add a global trainer from a source file."""
    try:
        manager = _get_manager()
        record = manager.add_global(source, name, description)
        console.print(f"[green]Added global trainer '{record['name']}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@trainer_app.command("list")
def trainer_list(
    show_obsolete: bool = typer.Option(False, "--show-obsolete", help="Include obsolete trainers"),
) -> None:
    """List all registered trainers."""
    try:
        manager = _get_manager()
        trainers = manager.list(show_obsolete=show_obsolete)

        if not trainers:
            console.print("[dim]No trainers registered.[/dim]")
            return

        table = Table(title="Trainers", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Obsolete", justify="center")
        table.add_column("Description")

        for t in trainers:
            table.add_row(
                t["name"],
                t["trainer_type"],
                "Yes" if t["is_obsolete"] else "No",
                t.get("description") or "",
            )

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@trainer_app.command("remove")
def trainer_remove(
    name: str = typer.Argument(help="Name of the trainer to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a trainer."""
    try:
        manager = _get_manager()
        manager.get(name)  # Verify it exists

        if not yes and not typer.confirm(f"Remove trainer '{name}'?"):
            console.print("[dim]Cancelled.[/dim]")
            return

        manager.remove(name)
        console.print(f"[green]Removed trainer '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@trainer_app.command("edit")
def trainer_edit(
    name: str = typer.Argument(help="Name of the trainer to edit"),
    description: str | None = typer.Option(None, "--description", "-d", help="New description"),
) -> None:
    """Edit trainer metadata."""
    try:
        manager = _get_manager()

        if description is None:
            console.print("[dim]No fields specified to edit.[/dim]")
            return

        manager.edit(name, description=description)
        console.print(f"[green]Updated trainer '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)
