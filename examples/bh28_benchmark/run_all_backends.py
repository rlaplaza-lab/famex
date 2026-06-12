#!/usr/bin/env python3
"""Multi-backend BH28 benchmark runner.

This script automatically manages conda environments and runs the BH28 benchmark
across all available backends, handling dependency conflicts by using separate
environments for incompatible backends.

Usage:
    python run_all_backends.py [--quick|--quicker]
    python run_all_backends.py --skip-env-setup  # Use existing environments
    python run_all_backends.py --backends aimnet2,uma  # Run specific backends only
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BackendGroup:
    """Represents a group of backends that can run together."""

    def __init__(self, name: str, backends: list[str], env_name: str) -> None:
        self.name = name
        self.backends = backends
        self.env_name = env_name


class MultiBackendBenchmarkRunner:
    """Runner for BH28 benchmark across multiple backends with environment management."""

    def __init__(
        self,
        benchmark_dir: Path,
        skip_env_setup: bool = False,
        quick: bool = False,
        quicker: bool = False,
    ) -> None:
        """Initialize the multi-backend runner."""
        self.benchmark_dir = Path(benchmark_dir).resolve()
        self.skip_env_setup = skip_env_setup
        self.quick = quick
        self.quicker = quicker
        self.results_dir = self.benchmark_dir / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)

        # Find famex package root (assume we're in examples/bh28_benchmark)
        # Go up to repo root
        self.repo_root = self.benchmark_dir.parent.parent

        # Check if conda is available
        self.conda_available = self._check_conda_available()

    def _check_conda_available(self) -> bool:
        """Check if conda is available."""
        try:
            result = subprocess.run(
                ["conda", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_all_backends(self) -> list[str]:
        """Get list of all backends to benchmark.

        Each backend will run in its own environment.

        Returns
        -------
        List of backend names (may include backend:model format)
        """
        backends = [
            "mace",
            "orb",
            "uma:uma-s-1p2",  # UMA Small
            "uma:uma-m-1p1",  # UMA Medium
            "aimnet2",
            # Optional backends (can be added later)
            # "tblite:GFN2-xTB",
            # "tblite:GFN1-xTB",
        ]
        return backends

    def get_env_name_for_backend(self, backend: str) -> str:
        """Generate conda environment name for a backend.

        Parameters
        ----------
        backend : str
            Backend name (may include backend:model format)

        Returns
        -------
        str
            Environment name like "famex-benchmark-orb" or "famex-benchmark-uma-s-1p1"
        """
        # Normalize backend name for env name (no colons, lowercase, replace special chars)
        env_name = backend.lower().replace(":", "-").replace("_", "-")
        return f"famex-benchmark-{env_name}"

    def filter_available_backends(self, backends: list[str]) -> list[str]:
        """Filter backends to only those that should be tested.

        Note: We can't check actual availability here since we're in the base
        environment. We'll let the benchmark script handle availability checking
        in each environment.
        """
        # All backends from groups are potentially testable
        return backends

    def setup_conda_environment_for_backend(self, backend: str) -> bool:
        """Create and setup a conda environment for a backend group.

        Parameters
        ----------
        backend : str
            Backend name (may include backend:model format)

        Returns
        -------
        bool
            True if setup successful, False otherwise
        """
        if not self.conda_available:
            logger.error("Conda is not available. Cannot setup environments.")
            return False

        env_name = self.get_env_name_for_backend(backend)
        base_backend = backend.split(":")[0].lower()

        # Check if environment already exists
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        env_exists = env_name in result.stdout

        if env_exists:
            logger.info(f"Environment {env_name} already exists.")
            if self.skip_env_setup:
                logger.info(f"Skipping setup for {env_name} (--skip-env-setup)")
                return True
            logger.info(f"Recreating environment {env_name}...")
            subprocess.run(["conda", "env", "remove", "-n", env_name, "-y"], check=False)

        # Create new environment
        logger.info(f"Creating conda environment {env_name}...")
        result = subprocess.run(
            [
                "conda",
                "create",
                "-n",
                env_name,
                "python=3.10",
                "-y",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Failed to create conda environment {env_name}")
            logger.error(result.stderr)
            return False

        # Install famex in editable mode
        logger.info(f"Installing famex in {env_name}...")
        install_commands = [
            ["conda", "run", "-n", env_name, "pip", "install", "-e", str(self.repo_root)],
        ]

        # Install backend-specific dependencies based on base backend name
        if base_backend == "aimnet2":
            # Install torch first, then torch-cluster separately
            install_commands.append(["conda", "run", "-n", env_name, "pip", "install", "torch"])
            install_commands.append(
                ["conda", "run", "-n", env_name, "pip", "install", "torch-cluster"]
            )
        elif base_backend == "orb":
            install_commands.append(
                ["conda", "run", "-n", env_name, "pip", "install", "orb-models", "torch"]
            )
        elif base_backend == "tblite":
            install_commands.append(["conda", "run", "-n", env_name, "pip", "install", "tblite"])
        elif base_backend == "uma":
            install_commands.append(
                ["conda", "run", "-n", env_name, "pip", "install", "fairchem-core"]
            )
        elif base_backend == "mace":
            install_commands.append(
                ["conda", "run", "-n", env_name, "pip", "install", "mace-torch"]
            )

        failed_installations = []
        for cmd in install_commands:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                pkg_name = cmd[-1] if cmd else "unknown"
                logger.warning(f"Installation failed for {pkg_name}")
                logger.warning(result.stderr[:500])  # Limit stderr output
                failed_installations.append(pkg_name)
                # Continue - some packages may be optional

        if failed_installations:
            logger.warning(
                f"Some installations failed in {env_name}: {', '.join(failed_installations)}"
            )
        else:
            logger.info(f"Successfully setup environment {env_name}")
        return True

    def run_benchmark_in_env(self, backend: str) -> Path | None:
        """Run the benchmark for a single backend in its environment.

        Parameters
        ----------
        backend : str
            The backend name to run (can be backend:model format)

        Returns
        -------
        Path | None
            Path to results JSON file if successful, None otherwise
        """
        env_name = self.get_env_name_for_backend(backend)
        benchmark_script = self.benchmark_dir / "bh28_benchmark.py"

        # Build command
        cmd = ["conda", "run", "-n", env_name, "python", str(benchmark_script)]

        # Add backend argument (supports backend:model format)
        cmd.extend(["--backends", backend])

        # Extract base backend name for results file naming
        backend.split(":")[0] if ":" in backend else backend

        # Add quick/quicker flags
        if self.quicker:
            cmd.append("--quicker")
        elif self.quick:
            cmd.append("--quick")

        # Set output directory - each backend gets its own directory
        backend_output_dir = self.results_dir / env_name
        backend_output_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output-dir", str(backend_output_dir)])

        # Add device argument (use GPU if available)
        cmd.extend(["--device", "cuda"])

        logger.info(f"Running benchmark for {backend} in {env_name}...")
        logger.info(f"Command: {' '.join(cmd)}")

        # Run the benchmark with timeout (4 hours max per backend)
        # Use timeout to prevent infinite hangs
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.benchmark_dir),
                capture_output=True,
                text=True,
                check=False,
                timeout=14400,  # 4 hours timeout
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out for {backend} in {env_name} after 4 hours")
            logger.error("This backend may be stuck. Continuing with next backend...")
            return None

        if result.returncode != 0:
            logger.error(f"Benchmark failed for {backend} in {env_name}")
            logger.error(result.stderr)
            return None

        # Find the results file
        results_file = backend_output_dir / "bh28_benchmark_results.json"
        if results_file.exists():
            logger.info(f"Results saved to {results_file}")
            return results_file
        else:
            logger.warning(f"Results file not found: {results_file}")
            return None

    def _normalize_backend_name(self, backend: str) -> str:
        """Normalize backend name for use as result key.

        Parameters
        ----------
        backend : str
            Backend name (may include :model format)

        Returns
        -------
        str
            Normalized backend name for result storage
        """
        # Convert uma:uma-s-1p1 -> uma-small, uma:uma-m-1p1 -> uma-medium
        # Convert tblite:GFN2-xTB -> tblite-gfn2, tblite:GFN1-xTB -> tblite-gfn1
        if ":" in backend:
            parts = backend.split(":", 1)
            base, model = parts[0], parts[1]
            if base == "uma":
                if "uma-s-1p2" in model or "small" in model.lower():
                    return "uma-small"
                elif "uma-m-1p1" in model or "medium" in model.lower():
                    return "uma-medium"
                else:
                    return f"{base}-{model}"
            elif base == "tblite":
                # GFN2-xTB -> gfn2, GFN1-xTB -> gfn1
                method_lower = model.lower()
                if "gfn2" in method_lower or "gf2" in method_lower:
                    return "tblite-gfn2"
                elif "gfn1" in method_lower or "gf1" in method_lower:
                    return "tblite-gfn1"
                else:
                    # Fallback: clean up and use as-is
                    method_clean = method_lower.replace("-", "").replace("_", "").replace("xtb", "")
                    return f"tblite-{method_clean}" if method_clean else "tblite"
            else:
                return f"{base}-{model}"
        # Handle plain "tblite" (defaults to GFN2-xTB)
        if backend.lower() == "tblite":
            return "tblite-gfn2"
        return backend

    def merge_results(self, result_files: dict[str, Path | None]) -> dict[str, Any]:
        """Merge results from multiple environments into unified structure.

        Parameters
        ----------
        result_files : dict[str, Path | None]
            Mapping of backend name to results file path

        Returns
        -------
        dict
            Unified results structure with backend as top-level keys
        """
        unified_results: dict[str, Any] = {}

        for backend, results_file in result_files.items():
            if results_file is None or not results_file.exists():
                logger.warning(f"Skipping {backend}: results file not found")
                continue

            try:
                with open(results_file) as f:
                    backend_results = json.load(f)

                # Normalize backend name for storage
                normalized_backend = self._normalize_backend_name(backend)

                # Results file has backend as key, extract it
                # Structure is: {backend: {reaction1: {...}, reaction2: {...}, ...}}
                # First try exact match
                if backend in backend_results:
                    unified_results[normalized_backend] = backend_results[backend]
                # Then try base backend (in case model wasn't in result key)
                elif backend.split(":")[0] in backend_results:
                    base_backend = backend.split(":")[0]
                    unified_results[normalized_backend] = backend_results[base_backend]
                elif len(backend_results) == 1:
                    # Only one backend in results, use it
                    # Get the key (backend name) and value (reactions dict)
                    backend_key = list(backend_results.keys())[0]
                    unified_results[normalized_backend] = backend_results[backend_key]
                else:
                    # Multiple backends or unexpected structure
                    # Try to find our backend in the results
                    if backend in backend_results:
                        unified_results[backend] = backend_results[backend]
                    else:
                        logger.warning(
                            f"Unexpected structure in {results_file}. "
                            f"Expected backend '{backend}' but found keys: {list(backend_results.keys())}"
                        )
                        # Include all results but with a note
                        unified_results[backend] = backend_results

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {results_file}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to load results from {results_file}: {e}", exc_info=True)
                continue

        return unified_results

    def run_all(self, requested_backends: list[str] | None = None) -> dict[str, Any]:
        """Run benchmark for all backends, each in its own environment.

        Parameters
        ----------
        requested_backends : list[str] | None
            If provided, only run these backends

        Returns
        -------
        dict
            Unified results structure
        """
        # Get all backends to run
        all_backends = self.get_all_backends()

        # Filter if specific backends requested (match full name or base backend)
        if requested_backends:
            requested_set = {b.lower() for b in requested_backends}

            def _matches_requested(backend_name: str) -> bool:
                backend_lower = backend_name.lower()
                base_backend = backend_lower.split(":")[0]
                return backend_lower in requested_set or base_backend in requested_set

            all_backends = [b for b in all_backends if _matches_requested(b)]

        result_files: dict[str, Path | None] = {}

        # Setup environments if needed
        if not self.skip_env_setup:
            for backend in all_backends:
                logger.info(f"Setting up environment for {backend}...")
                success = self.setup_conda_environment_for_backend(backend)
                if not success:
                    logger.warning(f"Failed to setup environment for {backend}, skipping")
                    continue

        # Run benchmarks - each backend in its own environment
        for backend in all_backends:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing backend: {backend}")
            logger.info(f"{'=' * 60}\n")

            results_file = self.run_benchmark_in_env(backend)
            result_files[backend] = results_file

        # Merge results from all individual result files
        logger.info("\nMerging results...")
        unified_results = self.merge_results(result_files)

        # Save unified results
        unified_file = self.results_dir / "unified_bh28_results.json"
        with open(unified_file, "w") as f:
            json.dump(unified_results, f, indent=2)

        logger.info(f"\n{'=' * 60}")
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"{'=' * 60}")
        successful_count = sum(1 for f in result_files.values() if f is not None)
        logger.info(f"Successfully ran {successful_count} backend(s)")
        logger.info(f"Results saved to {unified_file}")

        return unified_results


def main() -> int:
    """Run the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run BH28 benchmark across all available backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_backends.py                    # Run all backends
  python run_all_backends.py --quick            # Quick test subset
  python run_all_backends.py --skip-env-setup   # Use existing environments
  python run_all_backends.py --backends aimnet2,uma  # Specific backends only
        """,
    )

    parser.add_argument(
        "--skip-env-setup",
        action="store_true",
        help="Skip conda environment setup (use existing environments)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with representative subset",
    )
    parser.add_argument(
        "--quicker",
        action="store_true",
        help="Run quicker benchmark with single reaction",
    )
    parser.add_argument(
        "--backends",
        type=str,
        help="Comma-separated list of backends to run (e.g., aimnet2,uma,mace)",
    )

    args = parser.parse_args()

    # Get benchmark directory (assume script is in examples/bh28_benchmark)
    benchmark_dir = Path(__file__).parent.resolve()

    runner = MultiBackendBenchmarkRunner(
        benchmark_dir=benchmark_dir,
        skip_env_setup=args.skip_env_setup,
        quick=args.quick,
        quicker=args.quicker,
    )

    # Parse requested backends
    requested_backends = None
    if args.backends:
        requested_backends = [b.strip() for b in args.backends.split(",")]

    try:
        unified_results = runner.run_all(requested_backends=requested_backends)
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Successfully ran {len(unified_results)} backend(s)")
        logger.info(f"Results saved to {runner.results_dir / 'unified_bh28_results.json'}")
        return 0
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
