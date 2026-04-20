"""Shared CLI error formatting helpers.

Kept in its own module so that sub-command modules can import the
formatter without re-entering ``uesf.cli.app`` (whose module body
eagerly registers every sub-app).
"""

from __future__ import annotations

from rich.panel import Panel

from uesf.core.exceptions import UESFException


def _format_uesf_error(exc: UESFException) -> Panel:
    """Format a UESFException as a Rich Panel for CLI display."""
    error_type = type(exc).__name__
    lines = [f"[bold red]{exc.message}[/bold red]"]

    if exc.context:
        lines.append("")
        lines.append("[dim]Context:[/dim]")
        for k, v in exc.context.items():
            lines.append(f"  [dim]{k}:[/dim] {v}")

    if exc.hint:
        lines.append("")
        lines.append(f"[yellow]Hint: {exc.hint}[/yellow]")

    return Panel(
        "\n".join(lines),
        title=f"[red]{error_type}[/red]",
        border_style="red",
        expand=False,
    )
