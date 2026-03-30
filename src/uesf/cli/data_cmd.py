"""UESF CLI data commands: uesf data raw/preprocessed."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from uesf.cli.app import _format_uesf_error
from uesf.core.config import ConfigManager
from uesf.core.database import DatabaseManager
from uesf.core.exceptions import UESFException
from uesf.core.logging import setup_logging
from uesf.managers.data_manager import DataManager

console = Console()

data_app = typer.Typer(name="data", help="Manage EEG datasets.", no_args_is_help=True)
raw_app = typer.Typer(name="raw", help="Manage raw datasets.", no_args_is_help=True)
data_app.add_typer(raw_app, name="raw")


def _get_manager() -> DataManager:
    """Create and initialize DataManager."""
    setup_logging()
    db = DatabaseManager()
    db.initialize()
    config = ConfigManager(db)
    return DataManager(db, config)


@raw_app.command("register")
def raw_register(
    path: Path = typer.Argument(help="Path to raw dataset directory containing raw.yml"),
) -> None:
    """Register a raw dataset (user manages storage)."""
    try:
        manager = _get_manager()
        record = manager.register_raw(path)
        console.print(f"[green]Registered raw dataset '{record['name']}' ({record['n_subjects']} subjects)[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@raw_app.command("import")
def raw_import(
    path: Path = typer.Argument(help="Path to raw dataset directory containing raw.yml"),
) -> None:
    """Import a raw dataset (copy files to UESF data directory)."""
    try:
        manager = _get_manager()
        record = manager.import_raw(path)
        console.print(f"[green]Imported raw dataset '{record['name']}' ({record['n_subjects']} subjects)[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@raw_app.command("list")
def raw_list() -> None:
    """List all registered raw datasets."""
    try:
        manager = _get_manager()
        datasets = manager.list_raw()

        if not datasets:
            console.print("[dim]No raw datasets registered.[/dim]")
            return

        table = Table(title="Raw Datasets", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Subjects", justify="right")
        table.add_column("Data Shape")
        table.add_column("Imported", justify="center")
        table.add_column("Description")

        for ds in datasets:
            table.add_row(
                ds["name"],
                str(ds["n_subjects"] or "?"),
                ds.get("data_shape", "?"),
                "Yes" if ds["is_imported"] else "No",
                ds.get("description") or "",
            )

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@raw_app.command("remove")
def raw_remove(
    name: str = typer.Argument(help="Name of the raw dataset to remove"),
    delete_preprocessed: bool = typer.Option(
        False, "--delete-preprocessed", "-d",
        help="Also delete dependent preprocessed datasets",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a raw dataset."""
    try:
        manager = _get_manager()
        record = manager.get_raw(name)

        if not yes:
            # Show dependent preprocessed datasets
            deps = manager.db.fetch_all(
                "SELECT name FROM preprocessed_datasets WHERE source_raw_dataset_id = ?",
                (record["id"],),
            )
            msg = f"Remove raw dataset '{name}'?"
            if deps:
                dep_names = ", ".join(d["name"] for d in deps)
                msg += f"\n  Dependent preprocessed datasets: {dep_names}"
                if delete_preprocessed:
                    msg += "\n  [red]These will also be deleted![/red]"
                else:
                    msg += "\n  [yellow]These will be marked as orphans.[/yellow]"

            if not typer.confirm(msg):
                console.print("[dim]Cancelled.[/dim]")
                return

        manager.remove_raw(name, delete_preprocessed=delete_preprocessed)
        console.print(f"[green]Removed raw dataset '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@raw_app.command("edit")
def raw_edit(
    name: str = typer.Argument(help="Name of the raw dataset to edit"),
    description: str | None = typer.Option(None, "--description", help="New description"),
    sampling_rate: float | None = typer.Option(None, "--sampling-rate", help="Sampling rate in Hz"),
) -> None:
    """Edit metadata of a raw dataset."""
    try:
        manager = _get_manager()
        fields = {}
        if description is not None:
            fields["description"] = description
        if sampling_rate is not None:
            fields["sampling_rate"] = sampling_rate

        if not fields:
            console.print("[dim]No fields specified to edit.[/dim]")
            return

        manager.edit_raw(name, **fields)
        console.print(f"[green]Updated raw dataset '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@raw_app.command("info")
def raw_info(
    name: str = typer.Argument(help="Name of the raw dataset"),
) -> None:
    """Show detailed information about a raw dataset."""
    try:
        manager = _get_manager()
        record = manager.get_raw(name)

        table = Table(title=f"Raw Dataset: {name}", show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        for key in ["name", "description", "is_imported", "data_dir_path",
                     "eeg_data_key", "label_key", "n_subjects", "sampling_rate",
                     "n_sessions", "n_recordings", "n_channels", "n_samples",
                     "electrode_list", "data_shape", "dimension_info",
                     "label_shape", "numeric_to_semantic"]:
            val = record.get(key)
            if val is not None:
                # Pretty-print JSON fields
                if isinstance(val, str) and val.startswith(("[", "{")):
                    try:
                        val = json.dumps(json.loads(val), indent=2, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass
                table.add_row(key, str(val))

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)
