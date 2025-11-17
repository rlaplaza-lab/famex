"""PathManager class for handling reaction pathways and path operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms

from qme.interpolation.strategies import get_interpolation_strategy
from qme.io.geometry import Geometry
from qme.utils.logging import get_qme_logger
from qme.utils.validation import validate_atoms_compatibility

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_qme_logger(__name__)


@dataclass
class StructureSimilarity:
    """Result of structure similarity comparison.

    Attributes
    ----------
    is_similar : bool
        Whether structures are considered similar
    rmsd : float
        RMSD value in Angstroms
    energy_diff : float | None
        Energy difference in eV, or None if energy unavailable
    """

    is_similar: bool
    rmsd: float
    energy_diff: float | None


class PathManager:
    def __init__(
        self,
        structures: Sequence[Atoms] | Atoms,
        calculator: Any = None,
        name: str | None = None,
    ) -> None:
        self.name = name or "PathManager"

        if isinstance(structures, Atoms):
            structures_list = [structures]
        else:
            structures_list = list(structures)

        if len(structures_list) < 2:
            msg = "PathManager requires at least 2 structures"
            raise ValueError(msg)

        self.structures: list[Geometry] = []
        for struct in structures_list:
            if isinstance(struct, Atoms) and not isinstance(struct, Geometry):
                geom = Geometry(ase_atoms=struct)
            else:
                geom = struct
            self.structures.append(geom)

        for i in range(len(self.structures) - 1):
            validate_atoms_compatibility(
                self.structures[i],
                self.structures[i + 1],
                f"path segment {i}",
            )

        self.calculator = calculator

        if calculator is not None:
            for struct in self.structures:
                struct.calc = calculator

    @property
    def reactant(self) -> Geometry:
        return self.structures[0]

    @property
    def product(self) -> Geometry:
        return self.structures[-1]

    @property
    def reaction_energy(self) -> float | None:
        if self.reactant.energy is not None and self.product.energy is not None:
            return self.product.energy - self.reactant.energy
        return None

    def interpolate(
        self,
        npoints: int = 10,
        method: str = "linear",
        optimize_path: bool = False,
        calculator: Any | None = None,
        explorer: Any | None = None,
    ) -> list[Geometry]:
        if npoints < 2:
            msg = "Need at least 2 points for interpolation"
            raise ValueError(msg)

        if len(self.structures) == 2:
            return self._interpolate_segment(
                self.reactant,
                self.product,
                npoints,
                method,
                optimize_path,
                calculator,
            )

        segments = len(self.structures) - 1
        total_intervals = npoints - 1
        base_intervals = total_intervals // segments
        remainder = total_intervals % segments

        per_segment_npoints = []
        for i in range(segments):
            intervals = base_intervals + (1 if i < remainder else 0)
            per_segment_npoints.append(intervals + 1)

        stitched_path = []
        for i in range(segments):
            start_struct = self.structures[i]
            end_struct = self.structures[i + 1]
            nseg = per_segment_npoints[i]

            seg_path = self._interpolate_segment(
                start_struct,
                end_struct,
                nseg,
                method,
                optimize_path,
                calculator,
            )

            if i == 0:
                stitched_path.extend(seg_path)
            else:
                stitched_path.extend(seg_path[1:])

        return stitched_path

    def _interpolate_segment(
        self,
        start: Geometry,
        end: Geometry,
        npoints: int,
        method: str,
        optimize_path: bool,
        calculator: Any | None,
    ) -> list[Geometry]:
        start_coords = start.coords3d
        end_coords = end.coords3d

        interpolator = get_interpolation_strategy(method)
        path_coords = interpolator.interpolate(start_coords, end_coords, npoints)

        path_geometries = []
        for coords in path_coords:
            geom = Geometry(
                atoms=list(start.get_chemical_symbols()),
                positions=coords,
                charge=start.charge,
                mult=start.mult,
            )

            if hasattr(geom, "info") and geom.info is not None:
                geom.info.setdefault("charge", start.charge)
                geom.info.setdefault("spin", start.mult)

            if self.calculator is not None:
                geom.calc = self.calculator
            path_geometries.append(geom)

        if optimize_path:
            calc = calculator or self.calculator
            if calc is None:
                msg = "Need calculator for path optimization"
                raise ValueError(msg)
            path_geometries = self._optimize_path(path_geometries, calc)

        return path_geometries

    def _optimize_path(
        self, path_geometries: list[Geometry], calculator: Any | None
    ) -> list[Geometry]:
        logger.info("Note: Using simplified NEB-like algorithm for path optimization")

        for geom in path_geometries:
            geom.calc = calculator
            energy = geom.get_potential_energy()
            geom.energy = energy

        return path_geometries

    @staticmethod
    def find_ts_guess(path: Sequence[Atoms]) -> tuple[Atoms, int]:
        energies = []
        for atoms in path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("-inf"))  # Invalid energy

        if not energies or all(e == float("-inf") for e in energies):
            ts_index = len(path) // 2
        else:
            ts_index = energies.index(max(energies))

        return path[ts_index], ts_index

    @staticmethod
    def find_local_minima(path: Sequence[Atoms]) -> list[int]:
        import math

        energies = []
        for atoms in path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("nan"))

        minima_idxs = []
        for i, e in enumerate(energies):
            if math.isnan(e):
                continue
            left = energies[i - 1] if i - 1 >= 0 else float("inf")
            right = energies[i + 1] if i + 1 < len(energies) else float("inf")
            if (not math.isnan(left) and e < left) and (not math.isnan(right) and e < right):
                minima_idxs.append(i)

        if not minima_idxs:
            valid = [(i, e) for i, e in enumerate(energies) if not math.isnan(e)]
            if not valid:
                msg = "No valid energies found along path to select minima"
                raise RuntimeError(msg)
            min_idx = min(valid, key=lambda ie: ie[1])[0]
            minima_idxs = [min_idx]

        return minima_idxs

    @staticmethod
    def find_local_maxima(path: Sequence[Atoms]) -> list[int]:
        import math

        energies = []
        for atoms in path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("nan"))

        maxima_idxs = []
        for i, e in enumerate(energies):
            if math.isnan(e):
                continue
            left = energies[i - 1] if i - 1 >= 0 else float("-inf")
            right = energies[i + 1] if i + 1 < len(energies) else float("-inf")
            if (not math.isnan(left) and e > left) and (not math.isnan(right) and e > right):
                maxima_idxs.append(i)

        if not maxima_idxs:
            valid = [(i, e) for i, e in enumerate(energies) if not math.isnan(e)]
            if not valid:
                msg = "No valid energies found along path to select maxima"
                raise RuntimeError(msg)
            max_idx = max(valid, key=lambda ie: ie[1])[0]
            maxima_idxs = [max_idx]

        return maxima_idxs

    @staticmethod
    def calculate_rmsd(
        atoms1: Atoms,
        atoms2: Atoms,
        align: bool = True,
    ) -> float:
        if len(atoms1) != len(atoms2):
            return float("inf")

        pos1 = atoms1.get_positions()
        pos2 = atoms2.get_positions()

        if not align:
            diff = pos1 - pos2
            return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

        ref_cent = pos1.mean(axis=0)
        tar_cent = pos2.mean(axis=0)
        ref = pos1 - ref_cent
        tar = pos2 - tar_cent

        if len(pos1) == 1:
            diff = ref - tar
            return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

        C = np.dot(tar.T, ref)
        try:
            V, _S, Wt = np.linalg.svd(C)
            d = np.sign(np.linalg.det(np.dot(V, Wt)))
            D = np.diag([1.0, 1.0, d])
            U = np.dot(np.dot(V, D), Wt)
            aligned = np.dot(tar, U.T)
            rmsd = np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1)))
            return float(rmsd)
        except np.linalg.LinAlgError:
            diff = ref - tar
            return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    @staticmethod
    def calculate_structure_similarity(
        atoms1: Atoms,
        atoms2: Atoms,
        rmsd_threshold: float,
        energy_threshold: float,
        energy1: float | None = None,
        energy2: float | None = None,
        align: bool = True,
    ) -> StructureSimilarity:
        rmsd = PathManager.calculate_rmsd(atoms1, atoms2, align=align)

        if energy1 is None:
            try:
                energy1 = atoms1.get_potential_energy()
            except (RuntimeError, ValueError, AttributeError):
                energy1 = None

        if energy2 is None:
            try:
                energy2 = atoms2.get_potential_energy()
            except (RuntimeError, ValueError, AttributeError):
                energy2 = None

        if energy1 is not None and energy2 is not None:
            energy_diff = abs(energy1 - energy2)
        else:
            energy_diff = None

        is_similar = rmsd < rmsd_threshold
        if energy_diff is not None:
            is_similar = is_similar and energy_diff < energy_threshold
        elif is_similar:
            logger.debug(
                f"Structure similarity check: RMSD={rmsd:.3f} Å < threshold, "
                "but energy unavailable for comparison",
            )

        return StructureSimilarity(
            is_similar=is_similar,
            rmsd=rmsd,
            energy_diff=energy_diff,
        )

    @staticmethod
    def filter_redundant_structures(
        structures: list[Atoms],
        input_structures: list[Atoms] | None = None,
        rmsd_threshold: float = 0.1,
        energy_threshold: float = 0.001,
        strategy_name: str = "strategy",
        preserve_path_continuity: bool = False,
        min_spacing: int = 1,
        align_rmsd: bool = True,
    ) -> tuple[list[Atoms], list[int], list[str]]:
        if not structures:
            return [], [], []

        filtered_structures: list[Atoms] = []
        removed_indices = []
        warning_messages = []

        energies: list[float | None] = []
        for i, atoms in enumerate(structures):
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except (RuntimeError, ValueError, AttributeError):
                energies.append(None)
                logger.debug(f"Energy unavailable for structure {i + 1}")

        if input_structures is not None and input_structures:
            input_only = True
            key_indices = [0, len(structures) - 1] if len(structures) > 1 else [0]
            for struct_idx in key_indices:
                struct = structures[struct_idx]
                for input_struct in input_structures:
                    rmsd = PathManager.calculate_rmsd(struct, input_struct, align=align_rmsd)
                    if rmsd > rmsd_threshold:
                        input_only = False
                        break
                if not input_only:
                    break

            if input_only:
                for struct in structures:
                    for input_struct in input_structures:
                        rmsd = PathManager.calculate_rmsd(struct, input_struct, align=align_rmsd)
                        if rmsd > rmsd_threshold:
                            input_only = False
                            break
                    if not input_only:
                        break

            if input_only:
                warning_messages.append(
                    f"Warning: {strategy_name} only found input structures. "
                    f"No new optimized structures were discovered.",
                )

        rmsd_cache: dict[tuple[int, int], float] = {}

        def get_cached_rmsd(idx1: int, idx2: int) -> float:
            key = (min(idx1, idx2), max(idx1, idx2))
            if key not in rmsd_cache:
                rmsd_cache[key] = PathManager.calculate_rmsd(
                    structures[idx1],
                    structures[idx2],
                    align=align_rmsd,
                )
            return rmsd_cache[key]

        kept_indices: list[int] = []

        for i, atoms in enumerate(structures):
            is_redundant = False
            best_match_idx: int | None = None
            best_rmsd: float = float("inf")
            best_energy_diff: float | None = None

            for _j, filtered_idx in enumerate(kept_indices):
                rmsd = get_cached_rmsd(i, filtered_idx)
                if rmsd >= rmsd_threshold:
                    continue

                energy_i = energies[i]
                energy_j = energies[filtered_idx]

                similarity = PathManager.calculate_structure_similarity(
                    atoms,
                    structures[filtered_idx],
                    rmsd_threshold,
                    energy_threshold,
                    energy1=energy_i,
                    energy2=energy_j,
                    align=align_rmsd,
                )

                if similarity.is_similar:
                    is_redundant = True
                    best_match_idx = filtered_idx
                    best_rmsd = similarity.rmsd
                    best_energy_diff = similarity.energy_diff
                    break

            if preserve_path_continuity and not is_redundant and kept_indices:
                last_kept = kept_indices[-1]
                spacing = i - last_kept
                if spacing < min_spacing:
                    is_redundant = True
                    removed_indices.append(i)
                    warning_messages.append(
                        f"Warning: Removed structure {i + 1} to maintain path continuity "
                        f"(spacing: {spacing}, minimum: {min_spacing})",
                    )
                    continue

            if is_redundant and best_match_idx is not None:
                removed_indices.append(i)
                energy_str = (
                    f"ΔE: {best_energy_diff:.3f} eV"
                    if best_energy_diff is not None
                    else "energy unavailable"
                )
                warning_messages.append(
                    f"Warning: Removed redundant structure {i + 1} "
                    f"(RMSD: {best_rmsd:.3f} Å, {energy_str})",
                )
            elif not is_redundant:
                filtered_structures.append(atoms)
                kept_indices.append(i)

        if removed_indices:
            warning_messages.insert(
                0,
                f"Warning: {strategy_name} removed {len(removed_indices)} redundant structures "
                f"out of {len(structures)} total structures.",
            )

        return filtered_structures, removed_indices, warning_messages

    @staticmethod
    def attach_calculators(explorer: Any, structures: Sequence[Atoms] | Atoms) -> Any | None:
        struct_list = [structures] if isinstance(structures, Atoms) else list(structures)

        if not struct_list:
            return None

        for struct in struct_list:
            if hasattr(struct, "info") and struct.info is not None:
                struct.info.setdefault("charge", explorer.default_charge)
                struct.info.setdefault("spin", explorer.default_spin)

        for struct in struct_list:
            explorer._create_and_attach_calculator(struct)
            explorer._apply_constraints(struct)

        return struct_list[0].calc if struct_list else None

    def calculate_path_energies(self, path: list[Geometry] | list[Atoms]) -> list[float]:
        if not path:
            return []

        energies = []
        for geom in path:
            if geom.calc is None:
                geom.calc = self.calculator
            energies.append(geom.get_potential_energy())
        return energies

    def get_rmsd_profile(
        self,
        path_geometries: list[Geometry] | list[Atoms],
    ) -> tuple[list[float], list[float]]:
        rmsd_from_reactant = []
        rmsd_from_product = []

        reactant_coords = self.reactant.coords3d
        product_coords = self.product.coords3d

        for geom in path_geometries:
            coords = geom.get_positions() if isinstance(geom, Atoms) else geom.coords3d
            rmsd_r = np.sqrt(np.mean((coords - reactant_coords) ** 2))
            rmsd_from_reactant.append(rmsd_r)
            rmsd_p = np.sqrt(np.mean((coords - product_coords) ** 2))
            rmsd_from_product.append(rmsd_p)

        return rmsd_from_reactant, rmsd_from_product

    def to_xyz_trajectory(self, path_geometries: list[Geometry] | list[Atoms]) -> str:
        xyz_lines = []

        for i, geom in enumerate(path_geometries):
            xyz_lines.append(str(len(geom)))
            try:
                energy = geom.get_potential_energy()
                comment = f"Frame {i}, Energy = {energy:.6f} eV"
            except Exception:
                comment = f"Frame {i}"
            xyz_lines.append(comment)
            symbols = geom.get_chemical_symbols()
            coords = geom.get_positions()

            for _j, (symbol, pos) in enumerate(zip(symbols, coords, strict=False)):
                line = f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}"
                xyz_lines.append(line)

        return "\n".join(xyz_lines)

    def get_path_statistics(self, path: list[Atoms]) -> dict:
        energies = []
        for atoms in path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("nan"))

        import math

        valid_energies = [e for e in energies if not math.isnan(e)]

        stats = {
            "num_structures": len(path),
            "energies": energies,
            "min_energy": min(valid_energies) if valid_energies else None,
            "max_energy": max(valid_energies) if valid_energies else None,
            "energy_range": (max(valid_energies) - min(valid_energies) if valid_energies else None),
        }

        try:
            _, ts_idx = PathManager.find_ts_guess(path)
            stats["ts_index"] = ts_idx
        except Exception:
            stats["ts_index"] = None

        try:
            stats["minima_indices"] = PathManager.find_local_minima(path)
        except Exception:
            stats["minima_indices"] = []

        try:
            stats["maxima_indices"] = PathManager.find_local_maxima(path)
        except Exception:
            stats["maxima_indices"] = []

        return stats

    def __str__(self) -> str:
        return f"PathManager({len(self.reactant)} atoms, {len(self.structures)} structures)"

    def __repr__(self) -> str:
        return self.__str__()
