#!/usr/bin/env python3
"""Kaggle kernel entry point for FAMEX GPU CI (rendered by kaggle-gpu.yml)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

import tomllib

GIT_REF = "__GIT_REF__"
PYTEST_MARKER = "__PYTEST_MARKER__"
MLIP_EXTRA = "__MLIP_EXTRA__"
HF_TOKEN = "__HF_TOKEN__"
CONDA_ENV = "famex-gpu"
# Use /tmp so pytest/pip artifacts are not saved as Kaggle kernel output.
WORKDIR = Path("/tmp/famex")
# Kaggle mounts datasets either at /kaggle/input/<slug> or the newer
# /kaggle/input/datasets/<owner>/<slug>; _resolve_dataset_dir() probes for
# whichever layout this kernel actually got.
DATASET_OWNER = "rlaplaza"
DATASET_SLUG = "famexcisrc"
DATASET_INPUT = Path("/kaggle/input") / DATASET_SLUG
SOURCE_ARCHIVE = "famex-src.tar.gz"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
PYPI_INDEX = "https://pypi.org/simple"

# Suites that install via a famex optional-dependency extra.
_EXTRA_SUITES = frozenset({"pet", "uma"})
# Suites that need an extra pip package outside famex extras.
_PIP_EXTRA_PACKAGES: dict[str, list[str]] = {
    "mace": ["mace-torch"],
}


def log(message: str) -> None:
    print(message, flush=True)


def run(
    cmd: list[str], *, cwd: str | Path | None = None, env: dict[str, str] | None = None
) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _log_kaggle_inputs() -> None:
    inputs_root = Path("/kaggle/input")
    if not inputs_root.is_dir():
        log("No /kaggle/input directory mounted")
        return
    log("Kaggle input mounts:")
    for path in sorted(inputs_root.rglob("*")):
        if path.is_file():
            log(f"  {path} ({path.stat().st_size} bytes)")


def _conda_exe() -> str | None:
    """Return a usable conda executable, or ``None`` when conda is unavailable.

    Kaggle's default GPU image ships a plain CPython at
    ``/usr/local/lib/python3.12`` with no conda; the newer GPU runtimes in
    particular omit conda entirely. Callers must fall back to the system
    interpreter when this returns ``None``.
    """
    for candidate in (
        os.environ.get("CONDA_EXE", ""),
        "/opt/conda/bin/conda",
        shutil.which("conda") or "",
    ):
        if candidate and (candidate == "conda" or os.path.isfile(candidate)):
            return candidate
    return None


def _conda_python() -> list[str]:
    conda = _conda_exe()
    if conda is None:
        raise RuntimeError("conda required for conda Python path but not found")
    conda_env = os.environ.copy()
    conda_env["CONDA_PLUGINS_AUTO_ACCEPT_TOS"] = "yes"
    for tos_cmd in (
        [
            conda,
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/main",
        ],
        [
            conda,
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/r",
        ],
    ):
        subprocess.run(tos_cmd, env=conda_env, check=False)
    run([conda, "create", "-y", "-n", CONDA_ENV, "python=3.12"], env=conda_env)
    return [conda, "run", "--no-capture-output", "-n", CONDA_ENV, "python"]


def _resolve_python() -> list[str]:
    """Prefer the conda env interpreter; fall back to the system interpreter when conda is absent."""
    if _conda_exe() is not None:
        return _conda_python()
    log(f"conda not found; using system interpreter {sys.executable}")
    return [sys.executable]


def _safe_extractall(tar: tarfile.TarFile, path: Path) -> None:
    # Python 3.12+ (project minimum is 3.10; Kaggle GPU images are 3.12) provides
    # tarfile.data_filter.
    tar.extractall(path=path, filter="data")


def _extract_dataset_archive(archive: Path) -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    log(f"Extracting bundled source from {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extractall(tar, WORKDIR)


def _resolve_dataset_dir() -> Path | None:
    """Locate the mounted CI source dataset regardless of Kaggle's mount layout.

    Kaggle exposes dataset inputs at either ``/kaggle/input/<slug>`` or the
    newer ``/kaggle/input/datasets/<owner>/<slug>``; probe both, then fall back
    to a recursive search so a path-layout change can't silently break source
    discovery (the dataset is published and polled to 'complete' by the
    kaggle-gpu.yml workflow before the kernel launches).
    """
    candidates = [
        DATASET_INPUT,
        Path("/kaggle/input/datasets") / DATASET_OWNER / DATASET_SLUG,
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "pyproject.toml").is_file():
            return cand
    root = Path("/kaggle/input")
    if root.is_dir():
        for match in sorted(root.rglob(DATASET_SLUG)):
            if match.is_dir() and (match / "pyproject.toml").is_file():
                return match
    return None


def _find_dataset_archive() -> Path | None:
    dataset_dir = _resolve_dataset_dir()
    if dataset_dir is None:
        return None
    direct = dataset_dir / SOURCE_ARCHIVE
    if direct.is_file():
        return direct
    matches = sorted(dataset_dir.rglob(SOURCE_ARCHIVE))
    return matches[0] if matches else None


def _dataset_tree_ready() -> bool:
    dataset_dir = _resolve_dataset_dir()
    return dataset_dir is not None and (dataset_dir / "pyproject.toml").is_file()


def _copy_dataset_tree() -> None:
    dataset_dir = _resolve_dataset_dir()
    assert dataset_dir is not None
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    log(f"Copying bundled source tree from {dataset_dir}")
    shutil.copytree(
        dataset_dir,
        WORKDIR,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        dirs_exist_ok=True,
    )


def _fetch_repo_from_dataset() -> bool:
    archive = _find_dataset_archive()
    if archive is not None:
        _extract_dataset_archive(archive)
        return True
    if _dataset_tree_ready():
        _copy_dataset_tree()
        return True
    return False


def _fetch_repo() -> None:
    if not _fetch_repo_from_dataset():
        raise FileNotFoundError(
            "Kaggle dataset bundle 'rlaplaza/famexcisrc' not found in the kernel "
            "input; the kaggle-gpu.yml workflow publishes and polls it to "
            "'complete' before launching the kernel."
        )
    log("Using CI source bundle from Kaggle dataset input")


def _numpy_requirement() -> str:
    """Match pyproject.toml so Kaggle uses the same NumPy pin as CI tests."""
    data = tomllib.loads((WORKDIR / "pyproject.toml").read_text(encoding="utf-8"))
    for dep in data["project"]["dependencies"]:
        if dep.startswith("numpy"):
            return dep
    raise RuntimeError("numpy requirement missing from pyproject.toml")


def _install_numpy(py: list[str], pip: list[str]) -> None:
    """Install project NumPy before torch/torchvision (Kaggle base image ships 2.0.x)."""
    spec = _numpy_requirement()
    run([*pip, "install", "--no-cache-dir", spec])


def _install_torch_stack(py: list[str], pip: list[str]) -> None:
    """Install CUDA torch on Kaggle where cu124 wheels may be 2.4–2.6 only."""
    attempts = (
        [
            *pip,
            "install",
            "--no-cache-dir",
            "torch>=2.12.0,<2.13",
            "torchvision",
            "--index-url",
            PYTORCH_CUDA_INDEX,
            "--extra-index-url",
            PYPI_INDEX,
        ],
        [
            *pip,
            "install",
            "--no-cache-dir",
            # Last-resort Kaggle workaround: unpinned torch when cu124 index lacks 2.12.x.
            "torch",
            "torchvision",
            "--index-url",
            PYTORCH_CUDA_INDEX,
        ],
    )
    for cmd in attempts:
        log("+ " + " ".join(cmd))
        completed = subprocess.run(cmd)
        if completed.returncode == 0:
            return
        log(f"Torch install failed (exit {completed.returncode}); trying fallback")
    raise subprocess.CalledProcessError(1, attempts[-1])


def _should_skip_dep(dep: str) -> bool:
    """Skip lint/type tooling and torch (already installed from the CUDA index)."""
    skip_prefixes = ("ruff", "mypy", "types-", "pre-commit")
    if dep.startswith(skip_prefixes):
        return True
    return bool(dep.startswith(("torch>", "torch=")))


def _install_famex_suite(py: list[str], pip: list[str], *, mlip_extra: str) -> None:
    """Install famex + one GPU suite (``aimnet2``, ``mace``, ``pet``, or ``uma``)."""
    if mlip_extra not in ("aimnet2", "mace", "pet", "uma"):
        raise SystemExit(
            f"Unsupported MLIP_EXTRA={mlip_extra!r}; expected 'aimnet2', 'mace', 'pet', or 'uma'."
        )

    data = tomllib.loads((WORKDIR / "pyproject.toml").read_text(encoding="utf-8"))
    deps = list(data["project"]["dependencies"])
    optional = data["project"]["optional-dependencies"]
    deps.extend(optional["dev"])
    if mlip_extra in _EXTRA_SUITES:
        deps.extend(optional[mlip_extra])

    install_deps = [dep for dep in deps if not _should_skip_dep(dep)]
    # pytest-timeout is used by the kernel runner but not in famex[dev].
    install_deps.append("pytest-timeout>=2.3.0")

    if mlip_extra in _EXTRA_SUITES:
        editable = f".[{mlip_extra},dev]"
    else:
        editable = ".[dev]"

    run([*pip, "install", "--no-cache-dir", "-e", editable, "--no-deps"])
    run([*pip, "install", "--no-cache-dir", *install_deps])

    for pkg in _PIP_EXTRA_PACKAGES.get(mlip_extra, []):
        run([*pip, "install", "--no-cache-dir", pkg])

    if mlip_extra == "pet":
        # metatomic-torchsim (pulled by upet) may need vesin skin= from 0.6.0
        run(
            [
                *pip,
                "install",
                "--no-cache-dir",
                "vesin==0.6.0",
                "--force-reinstall",
                "--no-deps",
            ]
        )


def _assert_numpy_version(py: list[str]) -> None:
    spec = _numpy_requirement()
    run(
        [
            *py,
            "-c",
            (
                "import re\n"
                "import numpy as np\n"
                f"spec = {spec!r}\n"
                "match = re.fullmatch(r'numpy>=(\\d+)\\.(\\d+)(?:,<(\\d+)\\.(\\d+))?', spec)\n"
                "if match is None:\n"
                "    raise SystemExit(f'Unsupported numpy spec: {spec!r}')\n"
                "lo_major, lo_minor, hi_major, hi_minor = match.groups()\n"
                "lo = (int(lo_major), int(lo_minor))\n"
                "hi = (int(hi_major), int(hi_minor)) if hi_major else None\n"
                "parts = [int(part) for part in np.__version__.split('.')[:2]]\n"
                "version = (parts[0], parts[1])\n"
                "if version < lo or (hi is not None and version >= hi):\n"
                "    raise SystemExit(\n"
                "        f'NumPy {np.__version__} does not satisfy {spec!r}'\n"
                "    )\n"
                "print(f'NumPy {np.__version__} satisfies {spec!r}')\n"
            ),
        ]
    )


def _assert_cuda_usable(py: list[str]) -> None:
    run(
        [
            *py,
            "-c",
            (
                "import torch\n"
                "if not torch.cuda.is_available():\n"
                "    raise SystemExit('CUDA required')\n"
                "name = torch.cuda.get_device_name()\n"
                "cap = torch.cuda.get_device_capability()\n"
                "print(f'GPU: {name}, capability sm_{cap[0]}{cap[1]}')\n"
                "if cap[0] < 7:\n"
                "    raise SystemExit(\n"
                "        f'GPU {name} (sm_{cap[0]}{cap[1]}) is incompatible with the '\n"
                "        'installed PyTorch CUDA build; use machine_shape NvidiaTeslaT4'\n"
                "    )\n"
                "torch.ones(1, device='cuda')\n"
                "print('CUDA smoke test passed')\n"
            ),
        ]
    )


def _apply_hf_token(env: dict[str, str]) -> None:
    """Inject HuggingFace token for the uma suite when the workflow provided one."""
    token = HF_TOKEN.strip()
    if not token or token.startswith("__"):
        return
    env["HF_TOKEN"] = token
    env["HUGGING_FACE_HUB_TOKEN"] = token
    log("HuggingFace token configured for UMA weight download")


def main() -> int:
    try:
        _log_kaggle_inputs()
        _fetch_repo()
        os.chdir(WORKDIR)

        py = _resolve_python()
        pip = [*py, "-m", "pip"]
        run([*pip, "install", "--upgrade", "pip"])
        _install_numpy(py, pip)
        _install_torch_stack(py, pip)
        log(f"Installing famex suite={MLIP_EXTRA} for GPU CI")
        _install_famex_suite(py, pip, mlip_extra=MLIP_EXTRA)
        _assert_numpy_version(py)
        _assert_cuda_usable(py)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # e3nn (pulled by mace-torch) unpickles constants.pt via torch.load;
        # PyTorch >=2.6 defaults weights_only=True and rejects it.
        env.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
        if MLIP_EXTRA == "uma":
            _apply_hf_token(env)

        pytest_timeout = "1800" if PYTEST_MARKER.strip() else "3600"
        if "not slow" in PYTEST_MARKER:
            pytest_timeout = "1800"

        pytest_cmd = [
            *py,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            f"--timeout={pytest_timeout}",
            "--capture=tee-sys",
            "--log-cli-level=INFO",
            "--log-cli-format=%(asctime)s %(levelname)s %(name)s: %(message)s",
            "-rA",
            "--durations=25",
        ]
        marker = PYTEST_MARKER.strip()
        if marker:
            pytest_cmd.extend(["-m", marker])

        log("+ " + " ".join(pytest_cmd))
        completed = subprocess.run(pytest_cmd, env=env)
        return int(completed.returncode)
    except Exception:
        log("FAMEX Kaggle runner failed:")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
