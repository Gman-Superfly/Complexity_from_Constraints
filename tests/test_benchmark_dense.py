from __future__ import annotations

from experiments.benchmark_delta_f90 import make_dense_modules_and_couplings


def test_make_dense_modules_and_couplings_sizes() -> None:
    mods, coups, constraints, inputs = make_dense_modules_and_couplings(10)
    assert len(mods) == 10
    assert len(inputs) == 10
    # Each node adds three couplings (forward, benefit, skip)
    assert len(coups) == 30
    assert "delta_dense" in constraints


def test_dense_inputs_are_unique_lists() -> None:
    _, _, _, inputs = make_dense_modules_and_couplings(5)
    # ensure lists are not shared references
    ids = {id(inp) for inp in inputs}
    assert len(ids) == len(inputs)

