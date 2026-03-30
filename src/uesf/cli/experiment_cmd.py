"""UESF CLI experiment commands: uesf experiment add/list/remove/run/query."""

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
from uesf.managers.experiment_manager import ExperimentManager
from uesf.managers.metric_manager import MetricManager
from uesf.managers.model_manager import ModelManager
from uesf.managers.project_manager import ProjectManager
from uesf.managers.trainer_manager import TrainerManager

console = Console()

experiment_app = typer.Typer(
    name="experiment",
    help="Manage and run experiments.",
    no_args_is_help=True,
)


def _get_manager() -> ExperimentManager:
    """Create and initialize ExperimentManager with all dependencies."""
    setup_logging()
    db = DatabaseManager()
    db.initialize()
    config = ConfigManager(db)
    project_mgr = ProjectManager(db, config)
    model_mgr = ModelManager(db, config)
    trainer_mgr = TrainerManager(db, config)
    metric_mgr = MetricManager(db, config)
    return ExperimentManager(db, config, project_mgr, model_mgr, trainer_mgr, metric_mgr)


@experiment_app.command("add")
def experiment_add(
    project_dir: Path = typer.Option(".", "--project-dir", "-p", help="Project directory"),
    name: str | None = typer.Option(None, "--name", "-n", help="Experiment name"),
    from_existing: str | None = typer.Option(
        None, "--from", "-f", help="Copy config from existing experiment",
    ),
    description: str | None = typer.Option(None, "--description", "-d", help="Description"),
) -> None:
    """Create a new experiment configuration."""
    try:
        manager = _get_manager()
        yml_path = manager.add(project_dir, name, from_existing, description)
        console.print(f"[green]Created experiment config: {yml_path}[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@experiment_app.command("list")
def experiment_list(
    project_dir: Path = typer.Option(".", "--project-dir", "-p", help="Project directory"),
) -> None:
    """List all experiments for the current project."""
    try:
        manager = _get_manager()
        experiments = manager.list(project_dir)

        if not experiments:
            console.print("[dim]No experiments found.[/dim]")
            return

        table = Table(title="Experiments", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Status")
        table.add_column("Created")
        table.add_column("Description")

        for exp in experiments:
            status = exp["status"]
            style = {"COMPLETED": "green", "FAILED": "red", "RUNNING": "yellow"}.get(status, "dim")
            table.add_row(
                exp["experiment_name"],
                f"[{style}]{status}[/{style}]",
                str(exp.get("created_at", "")),
                exp.get("description") or "",
            )

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@experiment_app.command("remove")
def experiment_remove(
    name: str = typer.Argument(help="Experiment name to remove"),
    project_dir: Path = typer.Option(".", "--project-dir", "-p", help="Project directory"),
    results_only: bool = typer.Option(
        False, "--results-only", "-r", help="Only remove results, keep config",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Remove an experiment or just its results."""
    try:
        manager = _get_manager()
        action = "results for" if results_only else ""
        if not yes and not typer.confirm(f"Remove {action} experiment '{name}'?"):
            console.print("[dim]Cancelled.[/dim]")
            return

        manager.remove(project_dir, name, results_only=results_only)
        what = "results" if results_only else "experiment"
        console.print(f"[green]Removed {what} '{name}'[/green]")
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@experiment_app.command("run")
def experiment_run(
    experiment_name: str = typer.Option(
        ..., "--exp", "-e", help="Experiment name to run",
    ),
    project_dir: Path = typer.Option(".", "--project-dir", "-p", help="Project directory"),
) -> None:
    """Run an experiment."""
    try:
        manager = _get_manager()
        console.print(f"[bold]Running experiment '{experiment_name}'...[/bold]")
        results = manager.run(project_dir, experiment_name)

        console.print(f"\n[green bold]Experiment '{experiment_name}' completed![/green bold]")

        # Display key results
        table = Table(title="Results", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        for key, value in results.items():
            if key in ("history", "fold_results"):
                continue
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            elif isinstance(value, dict):
                table.add_row(key, json.dumps(value, indent=2))
            else:
                table.add_row(key, str(value))

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)


@experiment_app.command("query")
def experiment_query(
    project_dir: Path = typer.Option(".", "--project-dir", "-p", help="Project directory"),
    metrics: str | None = typer.Option(
        None, "--metrics", "-m", help="Comma-separated metric names to display",
    ),
    status: str | None = typer.Option(None, "--status", "-s", help="Filter by status"),
) -> None:
    """Query and compare experiment results."""
    try:
        manager = _get_manager()
        project_config = manager.project_manager.load(project_dir)
        project_name = project_config["project-name"]

        metric_list = [m.strip() for m in metrics.split(",")] if metrics else None
        rows = manager.query(project_name=project_name, metrics=metric_list, status=status)

        if not rows:
            console.print("[dim]No matching experiments found.[/dim]")
            return

        table = Table(title="Experiment Results", show_header=True)
        table.add_column("Experiment", style="cyan")
        table.add_column("Status")

        if metric_list:
            for m in metric_list:
                table.add_column(m, justify="right")

        for row in rows:
            status_val = row["status"]
            style = {"COMPLETED": "green", "FAILED": "red"}.get(status_val, "dim")
            values = [row["experiment_name"], f"[{style}]{status_val}[/{style}]"]

            if metric_list and "selected_metrics" in row:
                for m in metric_list:
                    val = row["selected_metrics"].get(m)
                    if isinstance(val, float):
                        values.append(f"{val:.4f}")
                    elif val is not None:
                        values.append(str(val))
                    else:
                        values.append("-")

            table.add_row(*values)

        console.print(table)
    except UESFException as exc:
        console.print(_format_uesf_error(exc))
        raise typer.Exit(code=1)
