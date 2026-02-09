"""CLI entry point for memz."""

from pathlib import Path

import typer
import uvicorn

from memz.config import ServiceConfig, load_config

app = typer.Typer(help="memz: continuous PEFT service")


@app.command()
def serve(
    config: Path = typer.Option("config.yaml", help="Path to config YAML"),
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8042, help="Bind port"),
):
    """Start the memz API server."""
    cfg = load_config(config) if config.exists() else ServiceConfig()
    typer.echo(f"Starting memz with backend={cfg.backend}, model={cfg.base_model}")

    # Pass config to server via module-level factory
    from memz.server import create_app

    server_app = create_app(cfg)

    uvicorn.run(server_app, host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
