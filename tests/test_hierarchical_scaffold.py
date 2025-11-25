from __future__ import annotations

from core.hierarchy import (
    CoarseEnergySpec,
    FamilyGrouping,
    compute_coarse_energy,
    select_modules_by_families,
)


def test_family_grouping_from_mapping():
    mapping = {0: "A", 1: "A", 2: "B", 3: "B", 4: "C"}
    grouping = FamilyGrouping.from_mapping(mapping)
    assert grouping.indices_for("A") == [0, 1]
    assert grouping.indices_for("B") == [2, 3]
    assert grouping.indices_for("C") == [4]
    assert sorted(grouping.families()) == ["A", "B", "C"]


def test_family_masses_mean_and_sum():
    mapping = {0: "A", 1: "A", 2: "B"}
    grouping = FamilyGrouping.from_mapping(mapping)
    etas = [0.2, 0.6, 0.9]
    masses_mean = grouping.masses(etas, mode="mean")
    masses_sum = grouping.masses(etas, mode="sum")
    assert masses_mean["A"] == (0.2 + 0.6) / 2.0
    assert masses_sum["A"] == (0.2 + 0.6)
    assert masses_mean["B"] == 0.9


def test_coarse_energy_quadratic_and_consistency():
    mapping = {0: "A", 1: "A", 2: "B", 3: "C"}
    grouping = FamilyGrouping.from_mapping(mapping)
    etas = [0.5, 0.5, 0.1, 0.9]
    spec = CoarseEnergySpec(mass_mode="mean", mass_weight=2.0, cross_family_consistency=1.0)
    total, breakdown = compute_coarse_energy(etas, grouping, spec)
    # Mass energies
    # fam A mass = mean(0.5, 0.5) = 0.5 -> 2 * (0.5^2) = 0.5
    # fam B mass = 0.1 -> 2 * 0.01 = 0.02
    # fam C mass = 0.9 -> 2 * 0.81 = 1.62
    mass_energy = 0.5 + 0.02 + 1.62
    assert "family:A:mass_energy" in breakdown
    assert abs((breakdown["family:A:mass_energy"] + breakdown["family:B:mass_energy"] + breakdown["family:C:mass_energy"]) - mass_energy) < 1e-9
    # Consistency uses variance of masses
    assert "consistency:variance" in breakdown
    assert total >= mass_energy


def test_select_modules_by_families_threshold_and_topk():
    mapping = {0: "A", 1: "A", 2: "B", 3: "B", 4: "C"}
    grouping = FamilyGrouping.from_mapping(mapping)
    etas = [0.9, 0.1, 0.8, 0.2, 0.05]
    # Threshold on family mass (mean) >= 0.5
    selected = select_modules_by_families(etas, grouping, mass_mode="mean", family_mass_threshold=0.5)
    # fam A mass = 0.5 -> selected, fam B mass = 0.5 -> selected, fam C mass = 0.05 -> not selected
    assert set(selected) == {0, 1, 2, 3}
    # Top-1 per active family
    selected_top1 = select_modules_by_families(etas, grouping, mass_mode="mean", family_mass_threshold=0.5, per_family_top_k=1)
    # top in fam A is idx 0 (0.9), in fam B is idx 2 (0.8)
    assert set(selected_top1) == {0, 2}
