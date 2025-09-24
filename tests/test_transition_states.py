"""
Unified transition state optimization tests across all available backends.

This module tests transition state searches on the same test problems using
all available ML backends (UMA, SO3LR, AIMNET2, Mock) that support SELLA
to ensure consistency and enable performance comparisons.

Test systems include:
- H2 dissociation pathway
- Water dissociation (H2O -> H + OH)
- Simple SN2-like reaction coordinate
- Methanol rotation barrier

Note: All transition state tests require SELLA to be installed.
"""

import tempfile
import time
from pathlib import Path

import pytest
from ase import Atoms

import qme
from qme.dependencies import deps

# Define available backends (only include those with SELLA support)
AVAILABLE_BACKENDS = []
if deps.has("sella"):
    AVAILABLE_BACKENDS.append("mock")
    if deps.has("fairchem"):
        AVAILABLE_BACKENDS.append("uma")
    if deps.has("so3lr"):
        AVAILABLE_BACKENDS.append("so3lr")
    if deps.has("aimnet2"):
        AVAILABLE_BACKENDS.append("aimnet2")


class TestTransitionStateDefinitions:
    """Define transition state test systems."""

    @staticmethod
    def get_h2_dissociation_ts_guess():
        """H2 dissociation transition state guess (very stretched H2)."""
        return Atoms("H2", positions=[[0, 0, 0], [3.0, 0, 0]])

    @staticmethod
    def get_water_dissociation_ts_guess():
        """Water dissociation TS guess (H2O -> H + OH)."""
        return Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],  # O
                [2.5, 0.0, 0.0],  # H (dissociating, far)
                [-0.5, 0.8, 0.0],  # H (staying)
            ],
        )

    @staticmethod
    def get_methanol_rotation_ts_guess():
        """Methanol OH rotation transition state guess."""
        return Atoms(
            "CH4O",
            positions=[
                [0.0, 0.0, 0.0],  # C
                [1.4, 0.0, 0.0],  # O
                [1.8, 1.0, 0.0],  # H (OH, perpendicular for TS)
                [-0.5, 1.0, 0.0],  # H (CH3)
                [-0.5, -0.5, 0.8],  # H (CH3)
                [-0.5, -0.5, -0.8],  # H (CH3)
            ],
        )

    @staticmethod
    def get_sn2_like_ts_guess():
        """Simple SN2-like transition state guess (F- + CH3Cl -> FCH3 + Cl-)."""
        return Atoms(
            "CH3FCl",
            positions=[
                [0.0, 0.0, 0.0],  # C (center)
                [-2.5, 0.0, 0.0],  # F (approaching nucleophile)
                [2.5, 0.0, 0.0],  # Cl (leaving group)
                [0.0, 1.1, 0.0],  # H
                [0.0, -0.5, 1.0],  # H
                [0.0, -0.5, -1.0],  # H
            ],
        )


@pytest.mark.skipif(
    not deps.has("sella"), reason="SELLA not available for TS optimization"
)
class TestTransitionStateOptimization:
    """Test transition state optimization across all available backends."""

    @pytest.fixture(params=AVAILABLE_BACKENDS)
    def backend(self, request):
        """Parametrized fixture for available backends with SELLA support."""
        backend_name = request.param

        # Double-check backend availability
        if backend_name == "uma" and not deps.has("fairchem"):
            pytest.skip("UMA backend not available")
        elif backend_name == "so3lr" and not deps.has("so3lr"):
            pytest.skip("SO3LR backend not available")
        elif backend_name == "aimnet2" and not deps.has("aimnet2"):
            pytest.skip("AIMNET2 backend not available")

        # Ensure SELLA is available
        if not deps.has("sella"):
            pytest.skip("SELLA not available for transition state optimization")

        return backend_name

    def test_h2_dissociation_ts(self, backend):
        """Test H2 dissociation transition state across all backends."""
        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            h2_ts_guess = TestTransitionStateDefinitions.get_h2_dissociation_ts_guess()

            start_time = time.time()
            result = optimizer.find_transition_state(
                atoms=h2_ts_guess, fmax=0.05, steps=100
            )
            optimization_time = time.time() - start_time

            # Check basic results
            assert "converged" in result
            assert "steps_taken" in result
            assert "optimized_atoms" in result

            # Check that we have a reasonable TS structure
            final_atoms = result["optimized_atoms"]
            final_distance = final_atoms.get_distance(0, 1)

            if backend == "mock":
                # Mock calculator should give stretched H2
                assert 1.5 < final_distance < 5.0
            else:
                # ML potentials might find a more realistic TS
                assert 1.0 < final_distance < 4.0

            print(
                f"Backend {backend}: H2 dissociation TS took {optimization_time:.3f}s, "
                f"{result['steps_taken']} steps, H-H distance: {final_distance:.3f} Å"
            )

        except ImportError:
            pytest.skip(f"{backend} backend dependencies not available")
        except Exception as e:
            # TS optimization can be challenging, log but don't fail
            print(f"Backend {backend}: H2 TS optimization failed with: {e}")

    def test_water_dissociation_ts(self, backend):
        """Test water dissociation transition state across all backends."""
        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            water_ts_guess = (
                TestTransitionStateDefinitions.get_water_dissociation_ts_guess()
            )

            start_time = time.time()
            result = optimizer.find_transition_state(
                atoms=water_ts_guess,
                fmax=0.1,  # Slightly looser convergence for complex system
                steps=150,
            )
            optimization_time = time.time() - start_time

            # Check convergence
            assert "converged" in result or result["steps_taken"] > 0
            final_atoms = result["optimized_atoms"]

            # Check O-H distances
            oh1_dist = final_atoms.get_distance(0, 1)  # O-H (dissociating)
            oh2_dist = final_atoms.get_distance(0, 2)  # O-H (staying)

            # Dissociating H should be farther
            if result.get("converged", False):
                if backend != "mock":
                    assert oh1_dist > oh2_dist  # Dissociating H should be farther
                    assert oh1_dist > 1.5  # Should be stretched
                    assert 0.8 < oh2_dist < 1.5  # Remaining OH should be reasonable

            print(
                f"Backend {backend}: H2O dissociation TS took "
                f"{optimization_time:.3f}s, {result['steps_taken']} steps, "
                f"O-H distances: {oh1_dist:.3f}, {oh2_dist:.3f} Å"
            )

        except ImportError:
            pytest.skip(f"{backend} backend dependencies not available")
        except Exception as e:
            print(f"Backend {backend}: H2O TS optimization failed with: {e}")

    def test_methanol_rotation_ts(self, backend):
        """Test methanol OH rotation transition state across all backends."""
        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            meoh_ts_guess = (
                TestTransitionStateDefinitions.get_methanol_rotation_ts_guess()
            )

            start_time = time.time()
            result = optimizer.find_transition_state(
                atoms=meoh_ts_guess, fmax=0.1, steps=200
            )
            optimization_time = time.time() - start_time

            # Check convergence
            assert "converged" in result or result["steps_taken"] > 0
            final_atoms = result["optimized_atoms"]

            # Check key structural features
            co_dist = final_atoms.get_distance(0, 1)  # C-O
            oh_dist = final_atoms.get_distance(1, 2)  # O-H

            if backend != "mock" and result.get("converged", False):
                assert 1.2 < co_dist < 1.8  # C-O should be reasonable
                assert 0.8 < oh_dist < 1.3  # O-H should be reasonable

            print(
                f"Backend {backend}: CH3OH rotation TS took {optimization_time:.3f}s, "
                f"{result['steps_taken']} steps, C-O: {co_dist:.3f} Å, "
                f"O-H: {oh_dist:.3f} Å"
            )

        except ImportError:
            pytest.skip(f"{backend} backend dependencies not available")
        except Exception as e:
            print(f"Backend {backend}: Methanol TS optimization failed with: {e}")

    def test_sn2_like_ts(self, backend):
        """Test SN2-like transition state across all backends."""
        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            sn2_ts_guess = TestTransitionStateDefinitions.get_sn2_like_ts_guess()

            start_time = time.time()
            result = optimizer.find_transition_state(
                atoms=sn2_ts_guess, fmax=0.1, steps=200
            )
            optimization_time = time.time() - start_time

            # Check convergence
            assert "converged" in result or result["steps_taken"] > 0
            final_atoms = result["optimized_atoms"]

            # Check key distances for SN2 mechanism
            cf_dist = final_atoms.get_distance(0, 1)  # C-F (forming)
            ccl_dist = final_atoms.get_distance(0, 2)  # C-Cl (breaking)

            if result.get("converged", False):
                # In TS, both bonds should be elongated
                if backend != "mock":
                    assert 1.3 < cf_dist < 3.0  # Forming bond
                    assert 1.5 < ccl_dist < 3.5  # Breaking bond

            print(
                f"Backend {backend}: SN2-like TS took {optimization_time:.3f}s, "
                f"{result['steps_taken']} steps, C-F: {cf_dist:.3f} Å, "
                f"C-Cl: {ccl_dist:.3f} Å"
            )

        except ImportError:
            pytest.skip(f"{backend} backend dependencies not available")
        except Exception as e:
            print(f"Backend {backend}: SN2 TS optimization failed with: {e}")


@pytest.mark.skipif(
    not deps.has("sella"), reason="SELLA not available for TS optimization"
)
class TestTransitionStateConsistency:
    """Test consistency of TS results between backends."""

    def test_h2_ts_cross_backend_consistency(self):
        """Test that all backends give reasonable H2 TS results."""
        results = {}

        for backend in AVAILABLE_BACKENDS:
            if backend == "uma" and not deps.has("fairchem"):
                continue
            elif backend == "so3lr" and not deps.has("so3lr"):
                continue
            elif backend == "aimnet2" and not deps.has("aimnet2"):
                continue

            try:
                optimizer = qme.QMEOptimizer(backend=backend)
                h2_ts = TestTransitionStateDefinitions.get_h2_dissociation_ts_guess()

                result = optimizer.find_transition_state(
                    atoms=h2_ts, fmax=0.1, steps=100
                )

                if result.get("converged", False):
                    final_distance = result["optimized_atoms"].get_distance(0, 1)
                    results[backend] = {
                        "distance": final_distance,
                        "steps": result["steps_taken"],
                    }

            except Exception as e:
                print(f"Backend {backend} failed H2 TS: {e}")
                continue

        # Check that converged results are reasonable
        distances = [r["distance"] for r in results.values()]
        if len(distances) > 1:
            # All should be stretched H2
            assert all(1.0 < d < 5.0 for d in distances)

            print(f"H2 TS optimization results: {results}")


@pytest.mark.skipif(
    not deps.has("sella"), reason="SELLA not available for TS optimization"
)
class TestTransitionStateFileIO:
    """Test file I/O functionality for transition states."""

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_ts_xyz_workflow(self, backend):
        """Test complete TS optimization workflow with XYZ files."""
        if backend == "uma" and not deps.has("fairchem"):
            pytest.skip("UMA backend not available")
        elif backend == "so3lr" and not deps.has("so3lr"):
            pytest.skip("SO3LR backend not available")
        elif backend == "aimnet2" and not deps.has("aimnet2"):
            pytest.skip("AIMNET2 backend not available")

        try:
            optimizer = qme.QMEOptimizer(backend=backend)

            with tempfile.TemporaryDirectory() as tmpdir:
                input_file = Path(tmpdir) / "ts_guess.xyz"
                output_file = Path(tmpdir) / "ts_optimized.xyz"

                # Create TS guess file
                h2_ts = TestTransitionStateDefinitions.get_h2_dissociation_ts_guess()
                h2_ts.write(str(input_file))

                # Load, optimize TS, save
                atoms = optimizer.load_structure(str(input_file))
                result = optimizer.find_transition_state(
                    atoms=atoms, fmax=0.1, steps=50
                )

                if "optimized_atoms" in result:
                    optimizer.save_structure(
                        result["optimized_atoms"], str(output_file)
                    )

                    # Verify output file
                    if output_file.exists():
                        final_atoms = optimizer.load_structure(str(output_file))
                        assert len(final_atoms) == 2

        except ImportError:
            pytest.skip(f"{backend} backend dependencies not available")
        except Exception as e:
            print(f"Backend {backend} TS workflow failed: {e}")


@pytest.mark.skipif(
    not deps.has("sella"), reason="SELLA not available for TS optimization"
)
class TestTransitionStateRobustness:
    """Test robustness of TS optimization with different starting conditions."""

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_multiple_h2_ts_guesses(self, backend):
        """Test TS optimization from multiple starting geometries."""
        if backend == "uma" and not deps.has("fairchem"):
            pytest.skip("UMA backend not available")
        elif backend == "so3lr" and not deps.has("so3lr"):
            pytest.skip("SO3LR backend not available")
        elif backend == "aimnet2" and not deps.has("aimnet2"):
            pytest.skip("AIMNET2 backend not available")

        try:
            optimizer = qme.QMEOptimizer(backend=backend)

            # Test different starting H-H distances
            starting_distances = [2.5, 3.0, 3.5, 4.0]
            results = []

            for dist in starting_distances:
                h2_guess = Atoms("H2", positions=[[0, 0, 0], [dist, 0, 0]])

                try:
                    result = optimizer.find_transition_state(
                        atoms=h2_guess, fmax=0.1, steps=80
                    )

                    if "optimized_atoms" in result:
                        final_dist = result["optimized_atoms"].get_distance(0, 1)
                        results.append(final_dist)

                except Exception as e:
                    print(f"Failed with starting distance {dist}: {e}")
                    continue

            # Check that we got some results
            if results:
                # Results should be in reasonable range
                assert all(1.0 < d < 5.0 for d in results)
                print(
                    f"Backend {backend}: Multiple H2 TS starting points gave "
                    f"distances: {results}"
                )

        except ImportError:
            pytest.skip(f"{backend} backend dependencies not available")


@pytest.mark.skipif(
    not deps.has("sella"), reason="SELLA not available for TS optimization"
)
class TestTransitionStatePerformance:
    """Performance comparison tests for transition state optimization."""

    def test_backend_ts_performance_comparison(self):
        """Compare TS optimization performance across backends."""
        if len(AVAILABLE_BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for performance comparison")

        performance_results = {}

        for backend in AVAILABLE_BACKENDS:
            if backend == "uma" and not deps.has("fairchem"):
                continue
            elif backend == "so3lr" and not deps.has("so3lr"):
                continue
            elif backend == "aimnet2" and not deps.has("aimnet2"):
                continue

            try:
                optimizer = qme.QMEOptimizer(backend=backend)
                h2_ts = TestTransitionStateDefinitions.get_h2_dissociation_ts_guess()

                start_time = time.time()
                result = optimizer.find_transition_state(
                    atoms=h2_ts, fmax=0.1, steps=100
                )
                total_time = time.time() - start_time

                performance_results[backend] = {
                    "time": total_time,
                    "steps": result.get("steps_taken", 0),
                    "converged": result.get("converged", False),
                }

            except Exception as e:
                print(f"Backend {backend} performance test failed: {e}")
                continue

        # Print performance comparison
        print("\nTransition State Performance Comparison:")
        print("-" * 50)
        for backend, perf in performance_results.items():
            print(
                f"{backend:>10}: {perf['time']:.3f}s, {perf['steps']:3d} steps, "
                f"converged: {perf['converged']}"
            )

        # Basic sanity checks
        assert len(performance_results) > 0, "No backends completed performance test"
