"""
Enhanced QME demonstration showing new features and improvements.

This example demonstrates:
1. Configuration management
2. Unified mock calculator system
3. Enhanced error handling
4. Multiple optimization backends
5. Reaction pathway analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from ase.build import molecule
from ase.io import write

import qme
from qme import Geometry, QMEOptimizer, Reaction, config, set_defaults


def setup_demo():
    """Setup demo with custom configuration."""
    print("QME Enhanced Demo")
    print("=" * 50)

    # Show current configuration
    print(f"Current default backend: {config.default_backend}")
    print(f"Current default optimizer: {config.default_optimizer}")
    print(f"Available backends: {list(QMEOptimizer.AVAILABLE_BACKENDS.keys())}")

    # Optionally modify defaults for this demo
    print("\nSetting custom defaults for demo...")
    set_defaults(
        default_fmax=0.005,  # Tighter convergence
        default_steps=300,  # More steps
        verbose_fallbacks=True,
    )
    print(f"Updated fmax default: {config.default_fmax}")


def test_multiple_backends():
    """Test multiple backends with unified mock system."""
    print("\n" + "=" * 50)
    print("Testing Multiple Backends")
    print("=" * 50)

    backends = ["uma", "so3lr", "aimnet2"]
    atoms = molecule("H2O")

    results = {}

    for backend in backends:
        print(f"\nTesting {backend.upper()} backend...")

        try:
            # This will automatically fall back to mock if dependencies missing
            qme_opt = QMEOptimizer(backend=backend, use_mock=True)

            # Load structure and optimize
            atoms_copy = atoms.copy()
            atoms_copy.calc = qme_opt.calculator

            print(f"  Calculator: {type(qme_opt.calculator).__name__}")
            print(f"  Backend config: {getattr(qme_opt.calculator, 'backend', 'N/A')}")

            # Quick optimization test
            result = qme_opt.optimize_minimum(
                atoms=atoms_copy, fmax=config.default_fmax, steps=50
            )

            results[backend] = {
                "converged": result["converged"],
                "steps": result["steps_taken"],
                "energy_change": result["energy_change"],
            }

            print(f"  Converged: {result['converged']}")
            print(f"  Steps: {result['steps_taken']}")
            print(f"  Energy change: {result['energy_change']:.6f} eV")

        except Exception as e:
            print(f"  Error with {backend}: {e}")
            results[backend] = {"error": str(e)}

    return results


def demonstrate_error_handling():
    """Demonstrate enhanced error handling."""
    print("\n" + "=" * 50)
    print("Testing Error Handling")
    print("=" * 50)

    qme_opt = QMEOptimizer(use_mock=True)

    # Test file not found
    print("\n1. Testing file not found...")
    try:
        qme_opt.load_structure("nonexistent_file.xyz")
    except FileNotFoundError as e:
        print(f"   ✓ Caught FileNotFoundError: {e}")

    # Test invalid backend
    print("\n2. Testing invalid backend...")
    try:
        QMEOptimizer(backend="invalid_backend")
    except ValueError as e:
        print(f"   ✓ Caught ValueError: {e}")

    # Test empty atoms for optimization
    print("\n3. Testing optimization with no atoms...")
    try:
        qme_opt.optimize_minimum()  # No atoms loaded
    except ValueError as e:
        print(f"   ✓ Caught ValueError: {e}")


def reaction_pathway_demo():
    """Demonstrate enhanced reaction pathway features."""
    print("\n" + "=" * 50)
    print("Reaction Pathway Analysis")
    print("=" * 50)

    # Create simple H2 dissociation reaction
    h2_reactant = Geometry(
        atoms=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        charge=0,
        mult=1,
    )

    h2_product = Geometry(
        atoms=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        charge=0,
        mult=3,  # Triplet dissociated state
    )

    reaction = Reaction(h2_reactant, h2_product, name="H2_dissociation")
    print(f"Created reaction: {reaction}")

    # Generate pathway with both methods
    print(f"\nGenerating reaction pathways...")
    linear_path = reaction.interpolate(npoints=11, method="linear")
    geodesic_path = reaction.interpolate(npoints=11, method="geodesic")

    print(f"Linear interpolation: {len(linear_path)} points")
    print(f"Geodesic interpolation: {len(geodesic_path)} points")

    # Calculate distances along path
    print(f"\nH-H distances along linear path:")
    for i, geom in enumerate(linear_path[::2]):  # Every other point
        distance = geom.get_distance(0, 1)
        print(f"  Point {i*2}: {distance:.3f} Å")

    # Get RMSD profile
    rmsd_from_r, rmsd_from_p = reaction.get_rmsd_profile(linear_path)
    print(f"\nRMSD analysis:")
    print(f"  Max RMSD from reactant: {max(rmsd_from_r):.3f} Å")
    print(f"  Max RMSD from product: {max(rmsd_from_p):.3f} Å")

    # Find transition state guess
    ts_guess = reaction.find_transition_state_guess(method="geodesic")
    ts_distance = ts_guess.get_distance(0, 1)
    print(f"\nTS guess H-H distance: {ts_distance:.3f} Å")

    # Export trajectory
    xyz_traj = reaction.to_xyz_trajectory(linear_path[:5])  # First 5 points
    with open("h2_pathway_demo.xyz", "w") as f:
        f.write(xyz_traj)
    print(f"\nSaved pathway to h2_pathway_demo.xyz")

    return linear_path, geodesic_path


def create_summary_plot(backend_results, linear_path, geodesic_path):
    """Create a summary plot of results."""
    print("\n" + "=" * 50)
    print("Creating Summary Plots")
    print("=" * 50)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Backend comparison
    backends = [b for b in backend_results.keys() if "error" not in backend_results[b]]
    if backends:
        steps = [backend_results[b]["steps"] for b in backends]
        ax1.bar(backends, steps)
        ax1.set_ylabel("Optimization Steps")
        ax1.set_title("Backend Performance Comparison")
        ax1.set_ylim(0, max(steps) * 1.1 if steps else 1)

    # H-H distances for both paths
    linear_distances = [geom.get_distance(0, 1) for geom in linear_path]
    geodesic_distances = [geom.get_distance(0, 1) for geom in geodesic_path]

    points = range(len(linear_distances))
    ax2.plot(points, linear_distances, "b-o", label="Linear", markersize=4)
    ax2.plot(points, geodesic_distances, "r-s", label="Geodesic", markersize=4)
    ax2.set_xlabel("Pathway Point")
    ax2.set_ylabel("H-H Distance (Å)")
    ax2.set_title("Interpolation Method Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Configuration overview
    config_info = [
        f"Default Backend: {config.default_backend}",
        f"Default Optimizer: {config.default_optimizer}",
        f"Default fmax: {config.default_fmax}",
        f"Default steps: {config.default_steps}",
        f"Mock force constant: {config.mock_force_constant}",
    ]

    ax3.text(
        0.1,
        0.9,
        "Current Configuration:",
        transform=ax3.transAxes,
        fontsize=12,
        fontweight="bold",
    )
    for i, info in enumerate(config_info):
        ax3.text(0.1, 0.8 - i * 0.12, f"• {info}", transform=ax3.transAxes, fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    # Dependency status
    ax4.text(
        0.1,
        0.9,
        "Dependency Status:",
        transform=ax4.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    dep_status = [
        f"PyTorch: {'✓' if qme.deps.has('torch') else '✗'}",
        f"SELLA: {'✓' if qme.deps.has('sella') else '✗'}",
        f"AIMNET2: {'✓' if qme.deps.has('aimnet2') else '✗'}",
        f"FairChem: {'✓' if qme.deps.has('fairchem') else '✗'}",
        f"SO3LR: {'✓' if qme.deps.has('so3lr') else '✗'}",
    ]

    for i, status in enumerate(dep_status):
        color = "green" if "✓" in status else "red"
        ax4.text(
            0.1,
            0.8 - i * 0.12,
            f"• {status}",
            transform=ax4.transAxes,
            fontsize=10,
            color=color,
        )
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig("qme_enhanced_demo.png", dpi=150, bbox_inches="tight")
    print("Saved summary plot to qme_enhanced_demo.png")


def main():
    """Run the enhanced QME demonstration."""
    try:
        # Setup
        setup_demo()

        # Test backends
        backend_results = test_multiple_backends()

        # Test error handling
        demonstrate_error_handling()

        # Test reaction pathways
        linear_path, geodesic_path = reaction_pathway_demo()

        # Create summary
        create_summary_plot(backend_results, linear_path, geodesic_path)

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Files created:")
        print("• h2_pathway_demo.xyz - Reaction pathway trajectory")
        print("• qme_enhanced_demo.png - Summary plots")
        print("=" * 50)

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
