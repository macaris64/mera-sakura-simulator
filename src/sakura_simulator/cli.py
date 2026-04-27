"""Typer-based CLI for the SAKURA-II simulator."""

import typer

from sakura_simulator.engine import SakuraEngine

app = typer.Typer(help="SAKURA-II NPU Simulator CLI")


@app.callback()
def callback():
    """SAKURA-II NPU Simulator."""


@app.command()
def hello():
    """Initialize the SAKURA-II engine and print its greeting."""
    engine = SakuraEngine()
    typer.echo(engine.greeting())


if __name__ == "__main__":  # pragma: no cover
    app()
