"""Tests for Great Lakes list-price job cost estimates."""

from __future__ import annotations

import pytest

from reportseff.great_lakes_pricing import PRICING_DATE, estimate_cost


@pytest.mark.parametrize(
    ("partition", "elapsed", "alloc_tres", "expected"),
    [
        ("standard", "60", "cpu=2,mem=7G", "$0.00"),
        ("debug", "120", "cpu=1,mem=28G", "$0.00"),
        ("viz", "30", "cpu=2,mem=14G", "$0.00"),
        ("largemem", "60", "cpu=1,mem=83.5G", "$0.00"),
        ("gpu", "90", "cpu=20,mem=90G,gres/gpu=2", "$0.01"),
        ("spgpu", "60", "cpu=8,mem=48G,gres/gpu:a40=1", "$0.00"),
        ("gpu-rtx6000", "60", "cpu=32,mem=192G,gres/gpu=1", "$0.01"),
        ("gpu_mig40", "60", "cpu=8,mem=250G,gres/gpu:a100_3g.20gb=1", "$0.01"),
    ],
)
def test_estimate_cost_for_each_profile(
    partition: str,
    elapsed: str,
    alloc_tres: str,
    expected: str,
) -> None:
    """CPU-, memory-, and GPU-dominant jobs use each supported profile."""
    assert (
        estimate_cost("greatlakes", "COMPLETED", partition, elapsed, alloc_tres)
        == expected
    )


@pytest.mark.parametrize(
    ("partition", "cpu_unit", "memory_unit", "one_unit", "two_units"),
    [
        ("standard", "1", "7", "$0.00", "$0.00"),
        ("debug", "1", "7", "$0.00", "$0.00"),
        ("viz", "1", "7", "$0.00", "$0.00"),
        ("largemem", "1", "41.75", "$0.00", "$0.00"),
        ("gpu", "20", "90", "$0.00", "$0.01"),
        ("spgpu", "4", "48", "$0.00", "$0.00"),
        ("gpu-rtx6000", "16", "192", "$0.00", "$0.01"),
        ("gpu_mig40", "8", "125", "$0.00", "$0.01"),
    ],
)
def test_cpu_memory_and_tied_costs(
    partition: str,
    cpu_unit: str,
    memory_unit: str,
    one_unit: str,
    two_units: str,
) -> None:
    """Every profile charges the maximum CPU or memory weight, including ties."""
    prefix = "greatlakes", "COMPLETED", partition, "60"
    assert estimate_cost(*prefix, f"cpu={cpu_unit},mem={memory_unit}G") == one_unit
    assert (
        estimate_cost(*prefix, f"cpu={float(cpu_unit) * 2:g},mem={memory_unit}G")
        == two_units
    )
    assert (
        estimate_cost(*prefix, f"cpu={cpu_unit},mem={float(memory_unit) * 2:g}G")
        == two_units
    )


@pytest.mark.parametrize(
    ("partition", "cpu_unit", "memory_unit", "expected"),
    [
        ("gpu", "20", "90", "$0.01"),
        ("spgpu", "4", "48", "$0.00"),
        ("gpu-rtx6000", "16", "192", "$0.01"),
        ("gpu_mig40", "8", "125", "$0.01"),
    ],
)
def test_gpu_dominant_costs(
    partition: str,
    cpu_unit: str,
    memory_unit: str,
    expected: str,
) -> None:
    """Every GPU profile can be dominated by the allocated GPU count."""
    alloc_tres = f"cpu={cpu_unit},mem={memory_unit}G,gres/gpu=2"
    assert (
        estimate_cost("greatlakes", "COMPLETED", partition, "60", alloc_tres)
        == expected
    )


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"])
def test_estimate_cost_for_finished_states(state: str) -> None:
    """All finished job states are eligible for estimates."""
    assert (
        estimate_cost("greatlakes", state, "standard", "60", "cpu=1,mem=7G")
        == "$0.00"
    )


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "running", "pending"])
def test_estimate_cost_omits_active_jobs(state: str) -> None:
    """Running and pending jobs do not display a changing cost estimate."""
    assert estimate_cost("greatlakes", state, "standard", "60", "cpu=1,mem=7G") == "---"


def test_estimate_cost_zero_runtime() -> None:
    """A finished zero-runtime job has a zero-dollar estimate."""
    assert (
        estimate_cost("greatlakes", "FAILED", "standard", "0", "cpu=1,mem=7G")
        == "$0.00"
    )


def test_estimate_cost_prefers_generic_gpu_count() -> None:
    """Typed GPUs are not double-counted when a generic GPU total is present."""
    alloc_tres = "cpu=20,mem=90G,gres/gpu=2,gres/gpu:v100=2"
    assert (
        estimate_cost("greatlakes", "COMPLETED", "gpu", "60", alloc_tres)
        == "$0.01"
    )


def test_estimate_cost_sums_typed_gpu_counts() -> None:
    """Typed GPU counts are summed when the generic GPU total is absent."""
    alloc_tres = "cpu=4,mem=48G,gres/gpu:a40=1,gres/gpu:a100=2"
    assert (
        estimate_cost("greatlakes", "COMPLETED", "spgpu", "60", alloc_tres)
        == "$0.01"
    )


@pytest.mark.parametrize(
    ("cluster", "partition"),
    [
        ("armis2", "standard"),
        ("greatlakes", "standard-oc"),
        ("greatlakes", "viz-long"),
        ("greatlakes", "faculty-owned"),
    ],
)
def test_estimate_cost_rejects_unsupported_targets(
    cluster: str,
    partition: str,
) -> None:
    """Only explicitly published Great Lakes partition rates are supported."""
    assert estimate_cost(cluster, "COMPLETED", partition, "60", "cpu=1,mem=7G") == "---"


@pytest.mark.parametrize(
    ("elapsed", "alloc_tres"),
    [
        ("", "cpu=1,mem=7G"),
        ("not-a-number", "cpu=1,mem=7G"),
        ("-1", "cpu=1,mem=7G"),
        ("60", ""),
        ("60", "cpu=1"),
        ("60", "mem=7G"),
        ("60", "cpu=x,mem=7G"),
        ("60", "cpu=1,mem=seven"),
        ("60", "cpu=1,mem=7G,gres/gpu=x"),
        ("60", "cpu=1,mem=7G,bad"),
        ("60", "cpu=1,mem="),
        ("60", "cpu=1,mem=7G,=1"),
        ("60", "cpu=1,mem=7G,cpu=2"),
    ],
)
def test_estimate_cost_rejects_incomplete_or_malformed_resources(
    elapsed: str,
    alloc_tres: str,
) -> None:
    """Incomplete and malformed runtime or resource data is unavailable."""
    assert estimate_cost("greatlakes", "COMPLETED", "gpu", elapsed, alloc_tres) == "---"


def test_pricing_profile_date() -> None:
    """The built-in rates have an explicit snapshot date."""
    assert PRICING_DATE == "2026-08-07"
