#!/usr/bin/env python3
"""
Trajectory Saving Example - Demonstrating NEB/CI-NEB Path Optimization

This example demonstrates QME's new trajectory saving functionality for
complete reaction pathways from NEB and CI-NEB calculations.
"""

import os
from pathlib import Path

from ase import Atoms

# Disable GUI popups
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("MPLBACKEND", "Agg")


def create_test_reaction():
    """Create a simple test reaction: H2 dissociation."""
    # H2 molecule (bond length ~0.74 Å)
    h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    h2.set_cell([10, 10, 10])
    h2.center()

    # Two separate H atoms (dissociated state)
    h_atoms = Atoms("H2", positions=[[0, 0, 0], [3.0, 0, 0]])
    h_atoms.set_cell([10, 10, 10])
    h_atoms.center()

    return h2, h_atoms


def demonstrate_trajectory_saving():
    """Demonstrate trajectory saving with NEB and CI-NEB."""
    print("=" * 60)
    print("QME Trajectory Saving Example")
    print("=" * 60)
    print("This example demonstrates saving complete reaction pathways")
    print("from NEB and CI-NEB calculations.\n")

    # Create test reaction
    reactant, product = create_test_reaction()
    print(f"Reactant: {len(reactant)} atoms")
    print(f"Product: {len(product)} atoms")

    try:
        from qme import Explorer

        # Test NEB with trajectory saving
        print("\n1. Testing NEB path optimization...")
        neb_explorer = Explorer(
            atoms=[reactant, product], backend="mock", strategy="two-ended", target="path"
        )

        neb_result = neb_explorer.run(mode="neb", npoints=5, steps=3, fmax=0.1, spring_constant=1.0)

        print(f"   NEB returned {len(neb_result['trajectory'])} images")
        neb_explorer.save_trajectory(neb_result["trajectory"], "neb_reaction_path.xyz")
        print("   ✅ NEB trajectory saved to 'neb_reaction_path.xyz'")

        # Test CI-NEB with trajectory saving
        print("\n2. Testing CI-NEB path optimization...")
        cineb_explorer = Explorer(
            atoms=[reactant, product], backend="mock", strategy="two-ended", target="path"
        )

        cineb_result = cineb_explorer.run(
            mode="cineb", npoints=5, steps=3, fmax=0.1, climb=True, spring_constant=1.0
        )

        print(f"   CI-NEB returned {len(cineb_result['trajectory'])} images")
        cineb_explorer.save_trajectory(cineb_result["trajectory"], "cineb_reaction_path.xyz")
        print("   ✅ CI-NEB trajectory saved to 'cineb_reaction_path.xyz'")

        # Verify output files
        print("\n3. Verifying output files...")
        for filename in ["neb_reaction_path.xyz", "cineb_reaction_path.xyz"]:
            if Path(filename).exists():
                with open(filename, "r") as f:
                    content = f.read()
                    frame_count = content.count("2")  # Count frames
                    file_size = Path(filename).stat().st_size
                print(f"   {filename}: {frame_count} frames, {file_size} bytes")
            else:
                print(f"   ❌ {filename} not found")

        print("\n🎉 Trajectory saving demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Complete reaction pathway saving")
        print("✅ NEB and CI-NEB trajectory support")
        print("✅ New target='path' functionality")
        print("✅ Explorer.save_trajectory() method")

    except ImportError as e:
        print(f"❌ Error importing QME: {e}")
        print("Please ensure QME is properly installed.")
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")


if __name__ == "__main__":
    demonstrate_trajectory_saving()
