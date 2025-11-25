from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Tuple


FamilyId = str


@dataclass
class FamilyGrouping:
    """Family grouping for modules by index.

    Provides bi-directional maps between module indices and family IDs and
    convenience helpers to aggregate per-family statistics from η vectors.
    """

    index_to_family: Dict[int, FamilyId]
    family_to_indices: Dict[FamilyId, List[int]]

    @staticmethod
    def from_mapping(index_to_family: Mapping[int, FamilyId]) -> "FamilyGrouping":
        fam_to_idx: Dict[FamilyId, List[int]] = {}
        for idx, fam in index_to_family.items():
            fam_to_idx.setdefault(str(fam), []).append(int(idx))
        # Ensure stable order per family
        for k in fam_to_idx:
            fam_to_idx[k] = sorted(fam_to_idx[k])
        return FamilyGrouping(index_to_family=dict(index_to_family), family_to_indices=fam_to_idx)

    @staticmethod
    def from_discriminator(
        num_modules: int,
        family_of_index: Callable[[int], FamilyId],
    ) -> "FamilyGrouping":
        mapping: Dict[int, FamilyId] = {}
        for i in range(num_modules):
            mapping[i] = str(family_of_index(i))
        return FamilyGrouping.from_mapping(mapping)

    def families(self) -> List[FamilyId]:
        return sorted(self.family_to_indices.keys())

    def indices_for(self, family: FamilyId) -> List[int]:
        return list(self.family_to_indices.get(str(family), []))

    def masses(
        self,
        etas: List[float],
        *,
        mode: str = "mean",
    ) -> Dict[FamilyId, float]:
        """Compute per-family mass from η values.

        mode:
          - "mean": average of member η
          - "sum": sum of member η
        """
        assert mode in ("mean", "sum"), f"unsupported mass mode: {mode}"
        masses: Dict[FamilyId, float] = {}
        for fam, idxs in self.family_to_indices.items():
            if not idxs:
                masses[fam] = 0.0
                continue
            vals = [float(etas[i]) for i in idxs]
            if mode == "sum":
                masses[fam] = float(sum(vals))
            else:
                masses[fam] = float(sum(vals) / float(len(vals)))
        return masses


@dataclass
class CoarseEnergySpec:
    """Specification for coarse family-level energy.

    mass_mode: how to compute family mass from η ('mean' or 'sum')
    mass_weight: global scale for mass quadratic energy
    per_family_weight: optional overrides by family
    cross_family_consistency: optional penalty weight for dispersion across families
    """

    mass_mode: str = "mean"
    mass_weight: float = 1.0
    per_family_weight: Optional[Mapping[FamilyId, float]] = None
    cross_family_consistency: float = 0.0


def compute_coarse_energy(
    etas: List[float],
    grouping: FamilyGrouping,
    spec: Optional[CoarseEnergySpec] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute coarse family-level energy and a breakdown dictionary.

    E_coarse = sum_f w_f * mass_f^2
               + λ_consistency * var_f(mass_f)
    """
    if spec is None:
        spec = CoarseEnergySpec()
    masses = grouping.masses(etas, mode=spec.mass_mode)
    breakdown: Dict[str, float] = {}
    total = 0.0
    # Mass quadratic terms
    for fam, mass in masses.items():
        w = float(spec.mass_weight)
        if spec.per_family_weight is not None:
            w = float(spec.per_family_weight.get(fam, w))
        e = w * float(mass * mass)
        breakdown[f"family:{fam}:mass_energy"] = e
        total += e
    # Cross-family consistency (variance penalty)
    if spec.cross_family_consistency > 1e-12 and len(masses) >= 2:
        vals = list(masses.values())
        mean_val = float(sum(vals) / float(len(vals)))
        var = float(sum((v - mean_val) ** 2 for v in vals) / float(len(vals)))
        cons = float(spec.cross_family_consistency) * var
        breakdown["consistency:variance"] = cons
        total += cons
    return total, breakdown


def select_modules_by_families(
    etas: List[float],
    grouping: FamilyGrouping,
    *,
    mass_mode: str = "mean",
    family_mass_threshold: Optional[float] = None,
    per_family_top_k: Optional[int] = None,
) -> List[int]:
    """Select module indices based on family masses and optional within-family top-k.

    - If family_mass_threshold is provided, only families with mass >= threshold are active.
    - If per_family_top_k is provided, pick top-k η within each active family; otherwise pick all.
    """
    masses = grouping.masses(etas, mode=mass_mode)
    # Determine active families
    if family_mass_threshold is None:
        active_families = set(masses.keys())
    else:
        active_families = {f for f, m in masses.items() if float(m) >= float(family_mass_threshold)}
    selected: List[int] = []
    for fam in sorted(active_families):
        idxs = grouping.indices_for(fam)
        if not idxs:
            continue
        if per_family_top_k is None or per_family_top_k >= len(idxs):
            selected.extend(idxs)
        else:
            # Pick top-k by η value
            scored = sorted(((i, float(etas[i])) for i in idxs), key=lambda t: t[1], reverse=True)
            selected.extend([i for i, _ in scored[: per_family_top_k]])
    # Stable order
    return sorted(selected)
