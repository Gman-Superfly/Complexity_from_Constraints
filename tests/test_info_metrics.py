"""Tests for Information Structure Metrics."""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.info_metrics import InformationMetrics


def test_compute_alignment():
    # Perfectly aligned
    a1 = InformationMetrics.compute_alignment([1, 0], [1, 0])
    assert a1 == pytest.approx(1.0)
    
    # Orthogonal
    a2 = InformationMetrics.compute_alignment([1, 0], [0, 1])
    assert a2 == pytest.approx(0.0)
    
    # Opposed
    a3 = InformationMetrics.compute_alignment([1, 0], [-1, 0])
    assert a3 == pytest.approx(-1.0)
    
    # Zero vector handling
    a4 = InformationMetrics.compute_alignment([0, 0], [1, 0])
    assert a4 == 0.0


def test_compute_drift():
    # No drift
    d1 = InformationMetrics.compute_drift([1, 2], [1, 2])
    assert d1 == 0.0
    
    # Euclidean distance
    d2 = InformationMetrics.compute_drift([0, 0], [3, 4])
    assert d2 == pytest.approx(5.0)


def test_compute_redundancy():
    # Normal case
    r1 = InformationMetrics.compute_redundancy(gain=0.5, uncertainty=0.5)
    assert r1 == pytest.approx(1.0)
    
    # High gain, low uncertainty -> high redundancy
    r2 = InformationMetrics.compute_redundancy(gain=1.0, uncertainty=0.1)
    assert r2 == pytest.approx(10.0)
    
    # Zero uncertainty (safe division)
    r3 = InformationMetrics.compute_redundancy(gain=1.0, uncertainty=0.0)
    assert r3 > 1e6  # Large value


def test_compute_hallucination_rate():
    # 50% rate
    h1 = InformationMetrics.compute_hallucination_rate(violations=5, total_constraints=10)
    assert h1 == 0.5
    
    # Zero constraints
    h2 = InformationMetrics.compute_hallucination_rate(violations=0, total_constraints=0)
    assert h2 == 0.0

