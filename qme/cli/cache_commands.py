"""Cache management commands for QME CLI."""

import click

from qme.backends.cache import get_model_cache


@click.group(help="Manage QME model cache")
def cache() -> None:
    """Manage QME model cache."""


@cache.command(help="Show cache information")
def info() -> None:
    """Show information about cached models."""
    cache = get_model_cache()
    info = cache.get_cache_info()

    click.echo("QME Model Cache Information")
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
def clear(model, yes) -> None:
    """Clear model cache."""
    cache = get_model_cache()

    if model:
        if not yes and not click.confirm(f"Clear cache for model '{model}'?"):
            click.echo("Cancelled.")
            return
        cache.clear_cache(model)
        click.echo(f"Cleared cache for model '{model}'.")
    else:
        if not yes and not click.confirm("Clear entire model cache?"):
            click.echo("Cancelled.")
            return
        cache.clear_cache()
        click.echo("Cleared entire model cache.")


@cache.command(help="Verify cache integrity")
def verify() -> None:
    """Verify integrity of cached models."""
    cache = get_model_cache()
    info = cache.get_cache_info()

    click.echo("Verifying cache integrity...")

    verified = 0
    corrupted = 0

    for model in info["models"]:
        model_path = cache.cache_dir / model["filename"]
        if model_path.exists():
            if cache._verify_checksum(model_path, model["checksum"]):
                click.echo(f"  ✅ {model['model_name']}")
                verified += 1
            else:
                click.echo(f"  ❌ {model['model_name']} (corrupted)")
                corrupted += 1
        else:
            click.echo(f"  ❌ {model['model_name']} (missing file)")
            corrupted += 1

    click.echo()
    click.echo(f"Verified: {verified}, Corrupted: {corrupted}")

    if corrupted > 0:
        click.echo("Run 'qme cache clear' to remove corrupted entries.")


if __name__ == "__main__":
    cache()
