#!/usr/bin/env python
"""Simple diagnostic to capture the exact error from tblite TS optimization."""

import os

# Disable CUDA/GPU usage - tblite doesn't need it and avoids memory issues
# MUST be set before importing torch or any ML libraries
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Empty string disables all GPUs
os.environ["TORCH_USE_CUDA_DSA"] = "0"

import traceback
from pathlib import Path

import qme

# Disable PyTorch CUDA if available (after import)
try:
    import torch

    # Set device to CPU before any operations
    torch.set_default_device("cpu")
    if torch.cuda.is_available():
        print("WARNING: CUDA is available but we're forcing CPU mode")
    else:
        print("✓ CUDA disabled - using CPU only")
except ImportError:
    pass


def diagnose_ts_optimization():
    """Run a diagnostic on one of the failing TS optimizations."""
    input_file = Path("dft_reference/M0018-TSopt_OaC_S_1.xyz")

    print("=" * 80)
    print("TBLITE TS Optimization Error Diagnostic")
    print("=" * 80)
    print(f"Structure: {input_file.name}")
    print(
        f"CUDA disabled (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})"
    )
    print()

    print("Creating QME Explorer with verbose output...")
    explorer = qme.Explorer.from_file(
        str(input_file),
        backend="tblite",
        target="ts",
        strategy="local",
        default_charge=0,
        default_spin=1,
        model_name=None,
        local_optimizer="sella",
        verbose=2,  # High verbosity
    )

    explorer.atoms_list[0].info["charge"] = 0
    explorer.atoms_list[0].info["spin"] = 1

    # Store summaries for printing at the end
    summaries = []

    print("\n" + "=" * 80)
    print("Step 1: Running TS optimization (without frequencies)...")
    print("=" * 80)
    print("(Watch for Sella optimizer output below)")
    print()

    try:
        result_opt = explorer.run(
            fmax=0.05,
            steps=1000,
            calculate_frequencies=False,  # Skip frequencies for now
            temperature=298.15,
        )
        # Collect summary instead of printing immediately
        summaries.append(
            {
                "step": 1,
                "success": True,
                "message": "Optimization completed successfully!",
                "converged": result_opt.get("converged", False),
                "steps": result_opt.get("steps_taken", 0),
                "energy": result_opt["optimized_atoms"].get_potential_energy(),
                "frequencies_count": None,
            }
        )
    except Exception as e:
        summaries.append(
            {"step": 1, "success": False, "error": str(e), "traceback": traceback.format_exc()}
        )
        print(f"\n✗ Optimization failed: {e}")
        traceback.print_exc()
        # Print summaries collected so far before returning
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for summary in summaries:
            print(f"\nStep {summary['step']}:")
            if summary["success"]:
                print(f"  ✓ {summary['message']}")
                print(f"    Converged: {summary['converged']}")
                print(f"    Steps: {summary['steps']}")
                print(f"    Final energy: {summary['energy']:.6f} eV")
            else:
                print(f"  ✗ Failed: {summary['error']}")
        return

    print("\n" + "=" * 80)
    print("Step 2: Running TS optimization (with frequencies)...")
    print("=" * 80)
    print()

    try:
        result_with_freq = explorer.run(
            fmax=0.05,
            steps=1000,
            calculate_frequencies=True,  # THIS triggers the Sella Hessian update error
            temperature=298.15,
        )
        freq_result = result_with_freq.get("frequency_analysis", {})
        # Collect summary instead of printing immediately
        summaries.append(
            {
                "step": 2,
                "success": True,
                "message": "Frequency calculation via QME completed successfully!",
                "converged": result_with_freq.get("converged", False),
                "steps": result_with_freq.get("steps_taken", 0),
                "energy": result_with_freq["optimized_atoms"].get_potential_energy(),
                "frequencies_count": len(freq_result.get("frequencies", [])),
            }
        )
    except Exception as e:
        summaries.append(
            {"step": 2, "success": False, "error": str(e), "traceback": traceback.format_exc()}
        )
        print(f"✗ Opt+freq {e}")
        traceback.print_exc()

    # Print all summaries at the end
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for summary in summaries:
        print(f"\nStep {summary['step']}:")
        if summary["success"]:
            print(f"  ✓ {summary['message']}")
            print(f"    Converged: {summary['converged']}")
            print(f"    Steps: {summary['steps']}")
            print(f"    Final energy: {summary['energy']:.6f} eV")
            if summary["frequencies_count"] is not None:
                print(f"    Frequencies computed: {summary['frequencies_count']}")
        else:
            print(f"  ✗ Failed: {summary['error']}")

    return


if __name__ == "__main__":
    diagnose_ts_optimization()
