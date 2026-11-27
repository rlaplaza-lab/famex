"""Cache management commands for FAMEX CLI."""

from __future__ import annotations

import click

from famex.backends.cache import get_model_cache
from famex.utils.logging import MARK_FAIL, MARK_OK


@click.group(help="Manage FAMEX model cache")
def cache() -> None:
    """Manage FAMEX model cache."""


@cache.command(help="Show cache information")
def info() -> None:
    """Show information about cached models."""
    model_cache = get_model_cache()
    info = model_cache.get_cache_info()

    click.echo("FAMEX Model Cache Information")
    click.echo("=" * 40)
    click.echo(f"Cache directory: {info['cache_dir']}")
    click.echo(f"Number of models: {info['model_count']}")
    click.echo(f"Total size: {info['total_size_mb']:.2f} MB")
    click.echo()

    if info["models"]:
        click.echo("Cached models:")
        for model in info["models"]:
            click.echo(f"  - {model['model_name']} ({model['size'] / (1024 * 1024):.2f} MB)")
    else:
        click.echo("No models cached.")


@cache.command(help="Clear model cache")
@click.option("--model", help="Specific model to clear (optional)")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def clear(model: str | None, yes: bool) -> None:
    """Clear model cache."""
    model_cache = get_model_cache()
    if model:
        prompt = f"Clear cache for model '{model}'?"
        success = f"Cleared cache for model '{model}'."
    else:
        prompt = "Clear entire model cache?"
        success = "Cleared entire model cache."

    if not yes and not click.confirm(prompt):
        click.echo("Cancelled.")
        return

    model_cache.clear_cache(model)
    click.echo(success)


@cache.command(help="Verify cache integrity")
def verify() -> None:
    """Verify integrity of cached models."""
    model_cache = get_model_cache()
    info = model_cache.get_cache_info()

    click.echo("Verifying cache integrity...")

    verified = 0
    corrupted = 0

    for model in info["models"]:
        model_path = model_cache.cache_dir / model["filename"]
        if model_path.exists():
            if model_cache._verify_checksum(model_path, model["checksum"]):
                click.echo(f"  {MARK_OK} {model['model_name']}")
                verified += 1
            else:
                click.echo(f"  {MARK_FAIL} {model['model_name']} (corrupted)")
                corrupted += 1
        else:
            click.echo(f"  {MARK_FAIL} {model['model_name']} (missing file)")
            corrupted += 1

    click.echo()
    click.echo(f"Verified: {verified}, Corrupted: {corrupted}")

    if corrupted > 0:
        click.echo("Run 'famex cache clear' to remove corrupted entries.")


if __name__ == "__main__":
    cache()
