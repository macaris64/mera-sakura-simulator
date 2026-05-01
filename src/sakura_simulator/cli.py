"""Typer-based CLI for the SAKURA-II simulator."""

import typer
from rich.console import Console
from rich.table import Table

from sakura_simulator.engine import SakuraEngine

app = typer.Typer(help="SAKURA-II NPU Simulator CLI")
models_app = typer.Typer(help="Model registry commands")
app.add_typer(models_app, name="models")


@app.callback()
def callback():
    """SAKURA-II NPU Simulator."""


@app.command()
def hello():
    """Initialize the SAKURA-II engine and print its greeting."""
    engine = SakuraEngine()
    typer.echo(engine.greeting())


@models_app.command("list")
def models_list(
    manifest: str = typer.Option("configs/models.yaml", "--manifest", help="Path to manifest YAML"),
):
    """List all registered models with their Space-Ready status."""
    from sakura_simulator.registry import ModelRegistry

    try:
        registry = ModelRegistry(manifest)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    table = Table("Name", "Version", "Space-Ready")
    for entry in registry.list_models():
        ready = registry.is_space_ready(entry)
        table.add_row(entry.name, entry.version, "YES" if ready else "NO")
    Console().print(table)


@models_app.command("download")
def models_download(
    name: str = typer.Argument(..., help="Model name to download"),
    manifest: str = typer.Option("configs/models.yaml", "--manifest", help="Path to manifest YAML"),
):
    """Download a model file and verify its SHA-256 checksum."""
    from sakura_simulator.registry import ModelRegistry

    try:
        registry = ModelRegistry(manifest)
        path = registry.download(name)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Downloaded: {path}")


@models_app.command("remove")
def models_remove(
    name: str = typer.Argument(..., help="Model name to remove"),
    manifest: str = typer.Option("configs/models.yaml", "--manifest", help="Path to manifest YAML"),
):
    """Delete a downloaded model file from disk."""
    from sakura_simulator.registry import ModelRegistry

    try:
        registry = ModelRegistry(manifest)
        path = registry.remove(name)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Removed: {path}")


@models_app.command("compile")
def models_compile(
    name: str = typer.Argument(..., help="Model name to compile"),
    manifest: str = typer.Option("configs/models.yaml", "--manifest", help="Path to manifest YAML"),
):
    """Compile a source model to SAKURA-II deployment artifacts."""
    from sakura_simulator.compiler import MeraCompiler
    from sakura_simulator.registry import ModelRegistry

    try:
        registry = ModelRegistry(manifest)
        entry = registry.get_model(name)
        if entry is None:
            typer.echo(f"Error: Model not found: {name}", err=True)
            raise typer.Exit(1)
        artifact_path = MeraCompiler().compile(entry)
        typer.echo(f"Compiled: {artifact_path}")
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@models_app.command("run")
def models_run(
    name: str = typer.Argument(..., help="Model name to run inference on"),
    iters: int = typer.Option(1, "--iters", help="Number of inference iterations"),
    manifest: str = typer.Option("configs/models.yaml", "--manifest", help="Path to manifest YAML"),
):
    """Run inference using compiled SAKURA-II artifacts."""
    from sakura_simulator.registry import ModelRegistry
    from sakura_simulator.runtime import MeraRuntime

    try:
        registry = ModelRegistry(manifest)
        entry = registry.get_model(name)
        if entry is None:
            typer.echo(f"Error: Model not found: {name}", err=True)
            raise typer.Exit(1)
        if not registry.is_compiled(entry):
            typer.echo(
                f"Error: '{name}' is not compiled. Run: sakura models compile {name}", err=True
            )
            raise typer.Exit(1)
        result = MeraRuntime().run(entry, entry.artifact_dir, iters=iters)
        typer.echo(f"Avg latency: {result.avg_latency_ms:.2f} ms")
        typer.echo(f"Min latency: {result.min_latency_ms:.2f} ms")
        typer.echo(f"P95 latency: {result.p95_latency_ms:.2f} ms")
        for out in result.outputs:
            typer.echo(f"Output: {out['name']} shape={out['shape']} dtype={out['dtype']}")
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@models_app.command("inspect")
def models_inspect(
    name: str = typer.Argument(..., help="Model name to inspect"),
    manifest: str = typer.Option("configs/models.yaml", "--manifest", help="Path to manifest YAML"),
):
    """Show detailed NPU constraints for a specific model."""
    from sakura_simulator.registry import ModelRegistry

    registry = ModelRegistry(manifest)
    entry = registry.get_model(name)
    if entry is None:
        typer.echo(f"Model not found: {name}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Name:          {entry.name}")
    typer.echo(f"Version:       {entry.version}")
    typer.echo(f"Path:          {entry.path}")
    typer.echo(f"Max Power (W): {entry.npu_constraints.max_power_watts}")
    typer.echo(f"Memory (MB):   {entry.npu_constraints.required_memory_mb}")
    typer.echo(f"Space-Ready:   {'YES' if registry.is_space_ready(entry) else 'NO'}")


if __name__ == "__main__":  # pragma: no cover
    app()
