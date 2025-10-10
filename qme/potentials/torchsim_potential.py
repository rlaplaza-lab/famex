"""
TorchSim Machine Learning Potential integration for ASE.

This module implements TorchSim calculator integration for supported ML potentials,
providing significant speedup over traditional ASE-based implementations through
automatic batching and efficient GPU memory management.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes

from qme.dependencies import deps
from qme.potentials.base_potential import BasePotential


class TorchSimPotential(BasePotential):
    """
    TorchSim potential calculator for supported ML models.

    This calculator provides access to TorchSim's accelerated ML potential
    implementations, offering significant speedup over ASE-based approaches
    through automatic batching and efficient GPU memory management.

    Supported models:
    - MACE (via TorchSim's MACE integration)
    - Fairchem models (UMA, etc.)
    - Other TorchSim-supported models
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        backend: str = "mace",  # Default to MACE
        default_charge: int = 0,
        default_spin: int = 1,
        **kwargs,
    ):
        """
        Initialize TorchSim potential calculator.

        Parameters:
        -----------
        model_name : str, optional
            Model name to use. Defaults to backend-specific default
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
        backend : str
            Backend to use ('mace', 'fairchem', etc.)
        default_charge : int
            Default charge to use if not specified in atoms.info (default: 0)
        default_spin : int
            Default spin multiplicity to use if not specified in atoms.info (default: 1)
        **kwargs : dict
            Additional arguments passed to BasePotential
        """
        if not deps.has("torch_sim"):
            raise ImportError(
                "TorchSim is required for TorchSimPotential. "
                "Install with: pip install torch-sim-atomistic"
            )

        if not deps.has("torch"):
            raise ImportError(
                "PyTorch is required for TorchSimPotential. " "Install with: pip install torch"
            )

        # Set device if not provided
        if device is None:
            from qme.utils.device import get_optimal_device

            device = get_optimal_device()

        # Set default model name based on backend if not provided
        if model_name is None:
            if backend.lower() == "mace":
                model_name = "mace-omol-0"  # Good default for molecules
            elif backend.lower() in ["fairchem", "uma"]:
                model_name = "uma-s-1p1"  # Good default for molecules

        # Initialize base class
        super().__init__(model_name=model_name, device=device, **kwargs)

        # TorchSim-specific attributes
        self.backend = backend
        self.default_charge = default_charge
        self.default_spin = default_spin
        self._torch_sim = None
        self._model = None
        self._state = None

        # Enable batch evaluation for TorchSim
        self._supports_batch_evaluation = True

    def _load_calculator(self):
        """Load the TorchSim model and setup."""
        from qme.logging_utils import quiet_backend_loading

        if self._torch_sim is not None:
            return  # Already loaded

        try:
            self._torch_sim = deps.get("torch_sim")
            torch = deps.get("torch")

            # Don't show model info - let the outer context handle it
            with quiet_backend_loading(
                "torchsim",
                self.model_name or self.backend,
                None,
                self.device,
                show_model_info=False,
            ):
                if self.backend.lower() == "mace":
                    self._load_mace_model()
                elif self.backend.lower() in ["fairchem", "uma"]:
                    self._load_fairchem_model()
                else:
                    raise ValueError(f"Unsupported TorchSim backend: {self.backend}")

        except ValueError as e:
            if "too many values to unpack" in str(e):
                raise ImportError(
                    f"TorchSim MACE compatibility issue with e3nn. "
                    f"MACE 0.3.14 requires e3nn==0.4.4, but a newer e3nn version is installed. "
                    f"This affects both regular MACE and TorchSim MACE backends. "
                    f"Error: {e}"
                )
            else:
                raise ImportError(
                    f"TorchSim not available ({e}). Install with: pip install torch-sim-atomistic"
                )
        except Exception as e:
            raise ImportError(
                f"TorchSim not available ({e}). Install with: pip install torch-sim-atomistic"
            )

    def _load_mace_model(self):
        """Load MACE model through TorchSim."""

        try:
            # Load MACE model through TorchSim
            from mace.calculators.foundations_models import mace_mp, mace_off, mace_omol
            from torch_sim.models.mace import MaceModel

            if self.model_name == "mace-omol-0":
                mace_model = mace_omol(return_raw_model=True)
            elif self.model_name.startswith("mace-mp"):
                model_size = self.model_name.replace("mace-mp-", "") or "medium"
                mace_model = mace_mp(model=model_size, return_raw_model=True)
            elif self.model_name.startswith("mace-off"):
                model_size = self.model_name.replace("mace-off-", "") or "medium"
                mace_model = mace_off(model=model_size, return_raw_model=True)
            else:
                # Default to MACE-OMOL
                mace_model = mace_omol(return_raw_model=True)

            # Create TorchSim MACE model
            torch_device = deps.get("torch").device(self.device)
            mace_model = mace_model.to(torch_device)
            self._mace_model = MaceModel(model=mace_model, device=torch_device)

        except (ValueError, AttributeError, RuntimeError) as e:
            if any(
                phrase in str(e)
                for phrase in [
                    "too many values to unpack",
                    "_compiled_main",
                    "tensor size",
                ]
            ):
                raise ImportError(
                    f"TorchSim MACE compatibility issue with e3nn versions. "
                    f"MACE models require e3nn==0.4.4 but e3nn 0.5+ is installed. "
                    f"This affects both regular MACE and TorchSim MACE backends. "
                    f"Use UMA backend or separate environment with e3nn==0.4.4. "
                    f"Error: {e}"
                )
            else:
                raise

        # Monkey-patch both the TorchSim wrapper and the underlying MACE model
        original_macemodel_forward = self._mace_model.forward
        original_mace_forward = self._mace_model.model.forward

        def patched_macemodel_forward(state, *args, **kwargs):
            import torch

            # Enable gradients on positions before processing
            if hasattr(state, "positions") and not state.positions.requires_grad:
                state.positions.requires_grad_(True)
            return original_macemodel_forward(state, *args, **kwargs)

        def patched_mace_forward(data, *args, **kwargs):
            import torch

            # Get atoms object from the calculator instance
            atoms = getattr(self, "_current_atoms", None)

            # Add total_spin and total_charge if not present
            # Use the same dtype as positions to ensure consistency
            dtype = data["positions"].dtype if "positions" in data else torch.float64

            if "total_spin" not in data:
                # Get spin from atoms.info or use default
                if atoms is not None and "spin" in atoms.info:
                    spin_value = float(atoms.info["spin"])
                else:
                    spin_value = float(self.default_spin)
                data["total_spin"] = torch.tensor([spin_value], device=torch_device, dtype=dtype)

            if "total_charge" not in data:
                # Get charge from atoms.info or use default
                if atoms is not None and "charge" in atoms.info:
                    charge_value = float(atoms.info["charge"])
                else:
                    charge_value = float(self.default_charge)
                data["total_charge"] = torch.tensor(
                    [charge_value], device=torch_device, dtype=dtype
                )

            # Enable gradients for positions if not already enabled
            if "positions" in data and not data["positions"].requires_grad:
                data["positions"].requires_grad_(True)
            return original_mace_forward(data, *args, **kwargs)

        self._mace_model.forward = patched_macemodel_forward
        self._mace_model.model.forward = patched_mace_forward
        self._model = self._mace_model

    def _load_fairchem_model(self):
        """Load Fairchem model through TorchSim."""

        try:
            # Try to load Fairchem model through TorchSim
            from torch_sim.models.fairchem import FairchemModel

            # For now, we'll use the regular Fairchem approach since TorchSim Fairchem
            # has compatibility issues with the current fairchem-core version
            raise ImportError("TorchSim Fairchem not compatible with current fairchem-core version")

        except ImportError as e:
            raise ImportError(
                f"TorchSim Fairchem not available ({e}). "
                f"Install with: pip install torch-sim-atomistic"
            )

    def _atoms_to_state(self, atoms: Atoms):
        """Convert ASE Atoms to TorchSim state."""
        torch = deps.get("torch")
        device = torch.device(self.device)
        # Use float64 for MACE models (they expect float64)
        dtype = torch.float64 if self.backend.lower() == "mace" else torch.float32

        # For non-periodic systems with zero cell, set a large cell to avoid issues
        atoms_copy = atoms.copy()
        if not any(atoms.get_pbc()) and torch.allclose(
            torch.tensor(atoms.get_cell().array, dtype=dtype),
            torch.zeros(3, 3, dtype=dtype),
        ):
            # Center the molecule with vacuum
            atoms_copy.center(vacuum=10.0)

        if self._state is None:
            self._state = self._torch_sim.io.atoms_to_state(atoms_copy, device=device, dtype=dtype)
        else:
            # Update existing state
            self._state = self._torch_sim.io.atoms_to_state(atoms_copy, device=device, dtype=dtype)

        # Enable gradients for positions (needed for force calculations)
        self._state.positions.requires_grad_(True)

        # Return the SimState directly - the MACE model will convert it if needed
        return self._state

    def _state_to_atoms(self, state):
        """Convert TorchSim state back to ASE Atoms."""
        return self._torch_sim.io.state_to_atoms(state)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """Calculate properties using TorchSim."""
        # Common setup
        super().calculate(atoms, properties, system_changes)

        # Set default charge and spin if not already set to avoid warnings
        if "charge" not in self.atoms.info:
            self.atoms.info["charge"] = self.default_charge
        if "spin" not in self.atoms.info:
            self.atoms.info["spin"] = self.default_spin

        # Ensure calculator is loaded
        if self._torch_sim is None:
            self._load_calculator()

        if self._model is None:
            raise RuntimeError("Failed to load TorchSim model")

        # Check if we're using a TorchSim model or regular calculator
        if hasattr(self._model, "forward"):
            # TorchSim model - use the state-based approach
            state = self._atoms_to_state(self.atoms)
            self._current_atoms = self.atoms

            # Calculate properties using TorchSim model
            torch = deps.get("torch")
            # Enable gradients for force calculations
            results = self._model(state)
        else:
            # Regular calculator (e.g., Fairchem fallback) - use direct calculation
            self._model.calculate(self.atoms, properties, system_changes)

            # Extract results from the calculator
            if "energy" in properties:
                try:
                    self.results["energy"] = self._model.results["energy"]
                except Exception:
                    self.results["energy"] = self.results.get("energy")

            if "forces" in properties:
                try:
                    self.results["forces"] = self._model.results["forces"]
                except Exception:
                    self.results["forces"] = self.results.get("forces")
            return

        if "energy" in properties and "energy" in results:
            energy = results["energy"]
            # Handle both single system and batched results
            if energy.dim() == 0:
                self.results["energy"] = float(energy)
            else:
                self.results["energy"] = float(energy[0])  # Take first system if batched

        if "forces" in properties and "forces" in results:
            forces = results["forces"]
            # Convert to numpy and ensure correct shape
            forces_np = forces.detach().cpu().numpy()
            self.results["forces"] = forces_np

    def get_potential_energy(self, atoms=None, force_consistent: bool = False):
        """Get potential energy."""
        if atoms is not None:
            self.atoms = atoms
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Get forces on atoms."""
        if atoms is not None:
            self.atoms = atoms
        return super().get_forces(atoms)

    def _get_backend_name(self) -> str:
        """Get the backend name for this calculator."""
        return f"torchsim_{self.backend}"

    def calculate_batch(self, atoms_list, properties=None):
        """Calculate properties for a batch of structures using TorchSim.

        This method leverages TorchSim's automatic batching capabilities to
        calculate properties for multiple structures simultaneously, providing
        significant speedup over individual calculations.

        Parameters:
        -----------
        atoms_list : List[Atoms]
            List of ASE Atoms objects to calculate properties for
        properties : List[str], optional
            Properties to calculate (default: ["energy", "forces"])

        Returns:
        --------
        List[dict]
            List of result dictionaries, one for each structure
        """
        if properties is None:
            properties = ["energy", "forces"]

        # Ensure calculator is loaded
        if self._torch_sim is None:
            self._load_calculator()

        if self._model is None:
            raise RuntimeError("Failed to load TorchSim model")

        # Convert all atoms to TorchSim states
        states = []
        for atoms in atoms_list:
            # Set default charge and spin if not already set
            if "charge" not in atoms.info:
                atoms.info["charge"] = self.default_charge
            if "spin" not in atoms.info:
                atoms.info["spin"] = self.default_spin

            state = self._atoms_to_state(atoms)
            states.append(state)

        # Check if we're using a TorchSim model or regular calculator
        if hasattr(self._model, "forward"):
            # TorchSim model - use the state-based approach
            batch_state = self._batch_states(states)
            self._current_atoms = atoms_list[0]  # Use first atoms for spin/charge info

            # Calculate properties for the entire batch
            torch = deps.get("torch")
            # Don't use no_grad() as MACE needs gradients for force calculations
            batch_results = self._model(batch_state)

            # Split results back to individual structures
            return self._split_batch_results(self._batch_results, len(atoms_list), properties)
        else:
            # Regular calculator - use individual calculations
            batch_results = []
            for atoms in atoms_list:
                # Set default charge and spin if not already set
                if "charge" not in atoms.info:
                    atoms.info["charge"] = self.default_charge
                if "spin" not in atoms.info:
                    atoms.info["spin"] = self.default_spin

                atoms.calc = self._model

                # Calculate properties
                self._model.calculate(atoms, properties=properties, system_changes=all_changes)

                # Extract results
                result = {}
                for prop in properties:
                    if prop in self._model.results:
                        result[prop] = self._model.results[prop]
                batch_results.append(result)

            return batch_results

    def _batch_states(self, states):
        """Batch multiple TorchSim states using proper TorchSim batching."""
        if not states:
            return None

        # Check if we're using a TorchSim model
        if hasattr(self._model, "forward"):
            try:
                # Try TorchSim's native batching (works on both CPU and GPU)
                if hasattr(self._torch_sim, "batch_states"):
                    # Use TorchSim's built-in batching
                    batch_state = self._torch_sim.batch_states(states)
                    batch_results = self._model(batch_state)
                    self._batch_results = self._split_batch_results(batch_results, len(states))
                    return states[0]

                # Fallback: Manual batching for TorchSim models
                elif len(states) > 1:
                    # Create a proper batch by concatenating states
                    batch_state = self._create_manual_batch(states)
                    batch_results = self._model(batch_state)
                    self._batch_results = self._split_batch_results(batch_results, len(states))
                    return states[0]

                else:
                    # Single state - process normally
                    result = self._model(states[0])
                    self._batch_results = [result]
                    return states[0]

            except Exception as e:
                # Fallback to individual processing if batching fails
                print(
                    "Warning: TorchSim batching failed ("
                    f"{e}), "
                    "falling back to individual processing"
                )
                batch_results = []
                for state in states:
                    result = self._model(state)
                    batch_results.append(result)
                self._batch_results = batch_results
                return states[0]

        else:
            # Regular calculator fallback (CPU compatible)
            self._batch_results = self._fallback_individual_calculations(states)
            return states[0]

    def _create_manual_batch(self, states):
        """Create a manual batch for TorchSim when native batching isn't available."""
        torch = deps.get("torch")

        # Extract common properties
        batch_positions = []
        batch_numbers = []
        batch_charges = []
        batch_spins = []

        for state in states:
            batch_positions.append(state.positions)
            batch_numbers.append(state.numbers)
            batch_charges.append(getattr(state, "charge", self.default_charge))
            batch_spins.append(getattr(state, "spin", self.default_spin))

        # Create batch tensors
        device = torch.device(self.device)
        dtype = torch.float64 if self.backend.lower() == "mace" else torch.float32

        # Pad to same size and stack
        max_atoms = max(len(pos) for pos in batch_positions)

        padded_positions = []
        padded_numbers = []

        for pos, num in zip(batch_positions, batch_numbers):
            if len(pos) < max_atoms:
                # Pad with zeros
                pad_size = max_atoms - len(pos)
                pos_padded = torch.cat(
                    [
                        torch.tensor(pos, dtype=dtype, device=device),
                        torch.zeros(pad_size, 3, dtype=dtype, device=device),
                    ]
                )
                num_padded = torch.cat(
                    [
                        torch.tensor(num, dtype=torch.long, device=device),
                        torch.zeros(pad_size, dtype=torch.long, device=device),
                    ]
                )
            else:
                pos_padded = torch.tensor(pos, dtype=dtype, device=device)
                num_padded = torch.tensor(num, dtype=torch.long, device=device)

            padded_positions.append(pos_padded)
            padded_numbers.append(num_padded)

        # Stack into batch
        batch_positions = torch.stack(padded_positions)
        batch_numbers = torch.stack(padded_numbers)
        batch_charges = torch.tensor(batch_charges, dtype=dtype, device=device)
        batch_spins = torch.tensor(batch_spins, dtype=dtype, device=device)

        # Create batch state (this depends on TorchSim's state format)
        # For now, we'll create a simple batch state that mimics the individual state structure
        batch_state = type(states[0])(
            positions=batch_positions,
            numbers=batch_numbers,
            charge=batch_charges,
            spin=batch_spins,
        )

        return batch_state

    def _fallback_individual_calculations(self, states):
        """Fallback for regular calculators (CPU compatible)."""
        batch_results = []
        for state in states:
            atoms = self._state_to_atoms(state)
            atoms.calc = self._model

            # Calculate properties
            self._model.calculate(
                atoms, properties=["energy", "forces"], system_changes=all_changes
            )

            result = {
                "energy": self._model.results.get("energy", 0.0),
                "forces": self._model.results.get("forces", np.zeros((len(atoms), 3))),
            }
            batch_results.append(result)

        return batch_results

    def _split_batch_results(self, batch_results, n_structures, properties=None):
        """Split batch results back to individual structure results."""
        if properties is None:
            properties = ["energy", "forces"]

        results = []

        # Handle different batch result formats
        if isinstance(batch_results, dict):
            # Single batch result - split by structure
            if "energy" in batch_results:
                energies = batch_results["energy"]
                if hasattr(energies, "shape") and len(energies.shape) > 0:
                    # Batched energies
                    for i in range(n_structures):
                        structure_results = {}
                        if i < len(energies):
                            structure_results["energy"] = float(energies[i])
                        results.append(structure_results)
                else:
                    # Single energy for all structures
                    for i in range(n_structures):
                        results.append({"energy": float(energies)})

            if "forces" in batch_results:
                forces = batch_results["forces"]
                if hasattr(forces, "shape") and len(forces.shape) > 2:
                    # Batched forces [batch, atoms, 3]
                    for i in range(n_structures):
                        if i < len(results):
                            if i < len(forces):
                                results[i]["forces"] = forces[i].detach().cpu().numpy()
                        else:
                            results.append({"forces": forces[i].detach().cpu().numpy()})
                else:
                    # Single force array for all structures
                    forces_np = forces.detach().cpu().numpy()
                    for i in range(n_structures):
                        if i < len(results):
                            results[i]["forces"] = forces_np
                        else:
                            results.append({"forces": forces_np})

        elif isinstance(batch_results, list):
            # List of individual results
            for i in range(n_structures):
                structure_results = {}
                if i < len(batch_results):
                    result = batch_results[i]

                    if "energy" in properties and "energy" in result:
                        energy = result["energy"]
                        if hasattr(energy, "dim") and energy.dim() == 0:
                            structure_results["energy"] = float(energy)
                        else:
                            structure_results["energy"] = float(energy)

                    if "forces" in properties and "forces" in result:
                        forces = result["forces"]
                        if hasattr(forces, "detach"):
                            structure_results["forces"] = forces.detach().cpu().numpy()
                        else:
                            structure_results["forces"] = forces

                results.append(structure_results)

        else:
            # Fallback: create empty results
            for i in range(n_structures):
                results.append({})

        return results


def get_torchsim_mace_calculator(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> TorchSimPotential:
    """Convenience function for TorchSim MACE calculator."""
    return TorchSimPotential(model_name=model_name, device=device, backend="mace", **kwargs)


def get_torchsim_uma_calculator(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> TorchSimPotential:
    """Convenience function for TorchSim UMA calculator."""
    return TorchSimPotential(model_name=model_name, device=device, backend="fairchem", **kwargs)
