#!/usr/bin/env python3
"""Regenerate figures and patch manuscript tables/prose from benchmark_runs/.

Local-only: figures/ and manuscript/ are gitignored. Never git-add them.
Barrier energies are reported in kcal/mol (JSON stores eV).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "benchmark_runs"
MANUSCRIPT = REPO / "manuscript" / "famex_paper.tex"
FIGURES = REPO / "figures"
sys.path.insert(0, str(FIGURES))
from plot_common import extract_zimmermann_data, load_zimmermann_results  # noqa: E402

BACKENDS = ["aimnet2", "uma", "mace"]
LABELS = {"aimnet2": "AIMNet2", "uma": "UMA", "mace": "MACE"}
EV_TO_KCAL = 23.0609


def bh28_row(backend: str) -> dict:
    data = json.loads((BENCHMARK / backend / "bh28_benchmark_results.json").read_text())
    rr = data[backend]
    n = len(rr)
    ts_conv = sum(1 for v in rr.values() if v.get("optimization_results", {}).get("ts_converged"))
    errs = [
        v["optimization_results"]["absolute_error"] * EV_TO_KCAL
        for v in rr.values()
        if v.get("optimization_results", {}).get("barrier_success")
    ]
    arr = np.asarray(errs, dtype=float)
    return {
        "label": LABELS[backend],
        "conv": f"{ts_conv}/{n}",
        "succ": f"{len(errs)}/{n}",
        "mae": float(np.abs(arr).mean()),
        "rmse": float(np.sqrt((arr**2).mean())),
    }


def z93_row(backend: str) -> dict:
    """Load Zimmermann stats from canonical JSON + shards (success-only RMSD)."""
    data = extract_zimmermann_data(load_zimmermann_results())
    if backend not in data:
        return {
            "label": LABELS[backend],
            "succ": "0/0",
            "median": float("nan"),
            "avg": float("nan"),
            "n": 0,
            "success": 0,
        }
    d = data[backend]
    use = d["rmsds"]
    med = float(np.median(use)) if len(use) else float("nan")
    avg = float(np.mean(use)) if len(use) else float("nan")
    return {
        "label": LABELS[backend],
        "succ": f"{d['success']}/{d['total']}",
        "median": med,
        "avg": avg,
        "n": d["total"],
        "success": d["success"],
    }


def timing_row(backend: str) -> dict:
    path = BENCHMARK / backend / "timing_benchmark_results.json"
    data = json.loads(path.read_text())
    if isinstance(data, list):
        item = next(x for x in data if x.get("backend") == backend)
    else:
        item = data.get(backend, data)
    timings = item.get("timings", {})
    return {
        "label": LABELS[backend],
        "opt": float(timings.get("optimization", 0)),
        "energy_ms": float(timings.get("single_energy", 0)) * 1e3,
        "forces_ms": float(timings.get("single_forces", 0)) * 1e3,
    }


def uma_ablation() -> tuple[float, float]:
    path = BENCHMARK / "model_ablation_uma" / "bh28_benchmark_results.json"
    data = json.loads(path.read_text())
    key = next(k for k in data if "1p1" in k)
    errs = [
        v["optimization_results"]["absolute_error"] * EV_TO_KCAL
        for v in data[key].values()
        if v.get("optimization_results", {}).get("barrier_success")
    ]
    arr = np.asarray(errs, dtype=float)
    return float(np.abs(arr).mean()), float(np.sqrt((arr**2).mean()))


def mace_off_row() -> dict | None:
    path = BENCHMARK / "mace_off_medium" / "bh28_benchmark_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    key = next(iter(data))
    rr = data[key]
    n = len(rr)
    ts_conv = sum(1 for v in rr.values() if v.get("optimization_results", {}).get("ts_converged"))
    errs = [
        v["optimization_results"]["absolute_error"] * EV_TO_KCAL
        for v in rr.values()
        if v.get("optimization_results", {}).get("barrier_success")
    ]
    if not errs:
        return None
    arr = np.asarray(errs, dtype=float)
    return {
        "label": "MACE-OFF",
        "conv": f"{ts_conv}/{n}",
        "succ": f"{len(errs)}/{n}",
        "mae": float(np.abs(arr).mean()),
        "rmse": float(np.sqrt((arr**2).mean())),
    }


# Plain Sella finite-differences its Hessian. The paper reports only the
# analytical-Hessian searches, on the 28 named BH28 reactions.
_TS_OPTIMIZERS = (
    ("rfo", "RFO"),
    ("sella-analytical", "SELLA"),
    ("trust-exact", "trust-exact"),
    ("trust-krylov", "trust-krylov"),
    ("trust-ncg", "trust-ncg"),
)


def ts_optimizer_table_body() -> str:
    """Build LaTeX rows for the analytical-Hessian TS optimizer table."""
    mapping = {"aimnet2": "aimnet2", "uma": "bh28_uma_s1p2", "mace": "mace"}
    reactions = set(
        json.loads(
            (
                REPO / "examples/bh28_benchmark/bh28_dataset/reference_barrier_heights.json"
            ).read_text()
        )
    )
    lines = []
    for b in BACKENDS:
        path = BENCHMARK / mapping[b] / "ts_benchmark_suite_full_bh28.json"
        if not path.exists():
            continue
        suite = json.loads(path.read_text())
        local = suite.get("local") or suite.get("local_bh28", [])
        for key, label in _TS_OPTIMIZERS:
            rows = [
                r for r in local if r.get("optimizer") == key and r.get("reaction") in reactions
            ]
            conv = []
            for r in rows:
                if "optimization_results" in r:
                    ok = r["optimization_results"].get("converged")
                    steps = r["optimization_results"].get("steps_taken", 0)
                    t = r.get("timings", {}).get("optimization", 0) or 0
                else:
                    ok = r.get("converged")
                    steps = r.get("steps", 0)
                    t = r.get("time_s", r.get("time", 0)) or 0
                if ok:
                    conv.append((steps, t))
            n = len(rows)
            nc = len(conv)
            if nc:
                mean_steps = float(np.mean([c[0] for c in conv]))
                mean_time = float(np.mean([c[1] for c in conv]))
            else:
                mean_steps = float("nan")
                mean_time = float("nan")
            lines.append(
                f"{LABELS[b]} & {label} & {nc}/{n} & {mean_steps:.1f} & {mean_time:.1f} \\\\"
            )
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return "\n".join(lines)


def replace_tabular(tex: str, label: str, new_body_rows: str) -> str:
    r"""Replace tabular body after \label{label} between midrule and bottomrule."""
    lab = rf"\label{{{label}}}"
    idx = tex.find(lab)
    if idx < 0:
        print(f"Warning: label {label} not found")
        return tex
    tab_start = tex.find(r"\begin{tabular}", idx)
    if tab_start < 0:
        print(f"Warning: tabular after {label} not found")
        return tex
    mid = tex.find(r"\midrule", tab_start)
    bottom = tex.find(r"\bottomrule", mid)
    if mid < 0 or bottom < 0:
        print(f"Warning: midrule/bottomrule missing for {label}")
        return tex
    mid_end = tex.find("\n", mid) + 1
    return tex[:mid_end] + new_body_rows + "\n" + tex[bottom:]


def replace_caption_units_bh28(tex: str) -> str:
    """Ensure BH28 table header uses kcal/mol."""
    tex = tex.replace(
        r"\textbf{MAE (eV)} & \textbf{RMSE (eV)}",
        r"\textbf{MAE (kcal\,mol$^{-1}$)} & \textbf{RMSE (kcal\,mol$^{-1}$)}",
    )
    # Also handle already-converted or alternate spacing
    tex = re.sub(
        r"\\textbf\{MAE \([^)]+\)\} & \\textbf\{RMSE \([^)]+\)\}",
        r"\\textbf{MAE (kcal\\,mol$^{-1}$)} & \\textbf{RMSE (kcal\\,mol$^{-1}$)}",
        tex,
        count=1,
    )
    return tex


def ensure_figure(tex: str, marker: str, block: str, anchor: str, before: bool = False) -> str:
    if marker in tex:
        return tex
    if before:
        return tex.replace(anchor, block + "\n" + anchor, 1)
    return tex.replace(anchor, anchor + "\n" + block, 1)


def patch_manuscript(bh28, z93, timing, mace_off, uma_1p1_mae, uma_1p1_rmse) -> None:
    tex = MANUSCRIPT.read_text()

    if r"\graphicspath" not in tex:
        tex = tex.replace(
            r"\usepackage{graphicx}",
            "\\usepackage{graphicx}\n\\graphicspath{{figures/}} % Sets the folder for your images",
        )

    tex = replace_caption_units_bh28(tex)

    bh28_rows = "\n".join(
        f"{r['label']} & {r['conv']} & {r['succ']} & {r['mae']:.2f} & {r['rmse']:.2f} \\\\"
        for r in bh28
    )
    if mace_off:
        bh28_rows += (
            f"\n{mace_off['label']} & {mace_off['conv']} & {mace_off['succ']} "
            f"& {mace_off['mae']:.2f} & {mace_off['rmse']:.2f} \\\\"
        )
    tex = replace_tabular(tex, "tab:bh28", bh28_rows)

    z_n = z93[0]["n"]
    z93_rows = "\n".join(
        f"{r['label']} & {r['succ']} & {r['median']:.2f} & {r['avg']:.2f} \\\\" for r in z93
    )
    tex = replace_tabular(tex, "tab:zimmermann93", z93_rows)

    timing_rows = "\n".join(
        f"{r['label']} & {r['opt']:.1f} & {r['energy_ms']:.2f} & {r['forces_ms']:.2f} \\\\"
        for r in timing
    )
    tex = replace_tabular(tex, "tab:timing", timing_rows)
    tex = replace_tabular(tex, "tab:ts_optimizer", ts_optimizer_table_body())

    # Captions / subset language
    tex = re.sub(r"\(20 reactions, GPU\)", f"({z_n} reactions, GPU)", tex)
    tex = re.sub(
        r"on a 20-reaction subset",
        f"on the full {z_n}-reaction Zimmermann-93 set",
        tex,
    )
    tex = re.sub(
        r"with 11 images on a 20-reaction subset",
        f"with 11 images on the full {z_n}-reaction set",
        tex,
    )

    # Narrative, merged figures, and the conclusions live in the manuscript.
    # Re-running this script refreshes tabular numbers and does not rewrite them.
    off = "n/a" if mace_off is None else f"{mace_off['mae']:.2f} kcal/mol"
    print(
        f"Ablation context (not written into prose): "
        f"UMA 1p1 MAE/RMSE {uma_1p1_mae:.2f}/{uma_1p1_rmse:.2f} kcal/mol; "
        f"MACE-OFF MAE {off}."
    )
    MANUSCRIPT.write_text(tex)
    print(f"Patched {MANUSCRIPT}")


def regenerate_figures() -> None:
    cmd = [sys.executable, str(FIGURES / "generate_all_figures.py")]
    print("Running", " ".join(cmd))
    subprocess.run(cmd, cwd=str(FIGURES), check=False)


def main() -> int:
    bh28 = [bh28_row(b) for b in BACKENDS]
    z93 = [z93_row(b) for b in BACKENDS]
    timing = [timing_row(b) for b in BACKENDS]
    mace_off = mace_off_row()
    uma_1p1_mae, uma_1p1_rmse = uma_ablation()

    print("BH28 (kcal/mol):", bh28)
    print("Z93:", z93)
    print("Timing:", timing)
    print("MACE-OFF:", mace_off)
    print(f"UMA 1p1 MAE/RMSE kcal/mol: {uma_1p1_mae:.2f}/{uma_1p1_rmse:.2f}")

    regenerate_figures()
    patch_manuscript(bh28, z93, timing, mace_off, uma_1p1_mae, uma_1p1_rmse)
    print("Done. figures/ and manuscript/ remain gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
