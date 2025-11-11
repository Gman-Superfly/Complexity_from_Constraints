from __future__ import annotations

from modules.connectivity.nl_threshold_shift import build_grid_bond_graph, largest_component_fraction


def test_connectivity_eta_behaves_with_p():
    n = 20
    G_low = build_grid_bond_graph(n=n, p=0.2, add_shortcuts=False, seed=1)
    G_high = build_grid_bond_graph(n=n, p=0.9, add_shortcuts=False, seed=1)
    eta_low = largest_component_fraction(G_low)
    eta_high = largest_component_fraction(G_high)
    assert 0.0 <= eta_low <= 1.0
    assert 0.0 <= eta_high <= 1.0
    assert eta_high > eta_low


def test_shortcuts_increase_connectivity_at_mid_p():
    n = 20
    p = 0.55
    G_no = build_grid_bond_graph(n=n, p=p, add_shortcuts=False, seed=2)
    G_yes = build_grid_bond_graph(n=n, p=p, add_shortcuts=True, shortcut_frac=0.05, seed=2)
    eta_no = largest_component_fraction(G_no)
    eta_yes = largest_component_fraction(G_yes)
    assert eta_yes >= eta_no - 1e-9


