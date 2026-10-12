"""Tests for Great Lakes and Armis2 list-price job cost estimates."""

from __future__ import annotations

from decimal import Decimal

import pytest

from reportseff.pricing import (
    CLUSTER_PRICING,
    calculate_cost,
    estimate_cost,
    format_cost,
)


@pytest.mark.parametrize(
    ("partition", "elapsed", "alloc_tres", "expected"),
    [
        ("standard", "60", "cpu=2,mem=7G", "$0.00050093"),
        ("debug", "120", "cpu=1,mem=28G", "$0.00200370"),
        ("viz", "30", "cpu=2,mem=14G", "$0.00025046"),
        ("largemem", "60", "cpu=1,mem=83.5G", "$0.00154074"),
        ("gpu", "90", "cpu=20,mem=90G,gres/gpu=2", "$0.00821736"),
        ("spgpu", "60", "cpu=8,mem=48G,gres/gpu:a40=1", "$0.00361574"),
        ("gpu-rtx6000", "60", "cpu=32,mem=192G,gres/gpu=1", "$0.00772203"),
        ("gpu_mig40", "60", "cpu=8,mem=250G,gres/gpu:a100_3g.20gb=1", "$0.00547824"),
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
        ("standard", "1", "7", "$0.00025046", "$0.00050093"),
        ("debug", "1", "7", "$0.00025046", "$0.00050093"),
        ("viz", "1", "7", "$0.00025046", "$0.00050093"),
        ("largemem", "1", "41.75", "$0.00077037", "$0.00154074"),
        ("gpu", "20", "90", "$0.00273912", "$0.00547824"),
        ("spgpu", "4", "48", "$0.00180787", "$0.00361574"),
        ("gpu-rtx6000", "16", "192", "$0.00386102", "$0.00772203"),
        ("gpu_mig40", "8", "125", "$0.00273912", "$0.00547824"),
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
        ("gpu", "20", "90", "$0.00547824"),
        ("spgpu", "4", "48", "$0.00361574"),
        ("gpu-rtx6000", "16", "192", "$0.00772203"),
        ("gpu_mig40", "8", "125", "$0.00547824"),
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
        == "$0.00025046"
    )


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "running", "pending"])
def test_estimate_cost_omits_active_jobs(state: str) -> None:
    """Running and pending jobs do not display a changing cost estimate."""
    assert estimate_cost("greatlakes", state, "standard", "60", "cpu=1,mem=7G") == "---"


def test_estimate_cost_zero_runtime() -> None:
    """A finished zero-runtime job has a zero-dollar estimate."""
    assert (
        estimate_cost("greatlakes", "FAILED", "standard", "0", "cpu=1,mem=7G")
        == "$0.00000000"
    )


def test_estimate_cost_prefers_generic_gpu_count() -> None:
    """Typed GPUs are not double-counted when a generic GPU total is present."""
    alloc_tres = "cpu=20,mem=90G,gres/gpu=2,gres/gpu:v100=2"
    assert (
        estimate_cost("greatlakes", "COMPLETED", "gpu", "60", alloc_tres)
        == "$0.00547824"
    )


def test_estimate_cost_sums_typed_gpu_counts() -> None:
    """Typed GPU counts are summed when the generic GPU total is absent."""
    alloc_tres = "cpu=4,mem=48G,gres/gpu:a40=1,gres/gpu:a100=2"
    assert (
        estimate_cost("greatlakes", "COMPLETED", "spgpu", "60", alloc_tres)
        == "$0.00542361"
    )


@pytest.mark.parametrize(
    ("cluster", "partition"),
    [
        ("somecluster", "standard"),
        ("greatlakes", "standard-oc"),
        ("greatlakes", "viz-long"),
        ("greatlakes", "faculty-owned"),
    ],
)
def test_estimate_cost_rejects_unsupported_targets(
    cluster: str,
    partition: str,
) -> None:
    """Only explicitly published cluster and partition rates are supported."""
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


def test_pricing_snapshot_dates() -> None:
    """Each cluster's built-in rates have an explicit snapshot date."""
    assert CLUSTER_PRICING["greatlakes"].date == "2026-08-07"
    assert CLUSTER_PRICING["armis2"].date == "2026-10-12"


def test_unrounded_costs_are_summed_before_formatting() -> None:
    """The total is calculated from raw costs rather than displayed values."""
    cost = calculate_cost("greatlakes", "COMPLETED", "standard", "960", "cpu=1,mem=7G")
    assert cost is not None
    assert format_cost(cost) == "$0.00400741"
    assert format_cost(cost + cost) == "$0.00801482"


@pytest.mark.parametrize(
    ("partition", "elapsed", "alloc_tres", "expected"),
    [
        ("standard", "3600", "cpu=4,mem=12G,node=1", Decimal("0.06961104")),
        ("debug", "3600", "cpu=4,mem=12G,node=1", Decimal("0.06961104")),
        ("standard", "60", "cpu=1,mem=14G", Decimal("0.000580092")),
        ("largemem", "60", "cpu=2,mem=10G", Decimal("0.001607408")),
        ("gpu", "3600", "cpu=5,mem=15G,gres/gpu=2", Decimal("0.33788892")),
        (
            "gpu-rtx6000",
            "600",
            "cpu=16,mem=192G,gres/gpu=1",
            Decimal("0.03973804"),
        ),
    ],
)
def test_armis2_costs(
    partition: str,
    elapsed: str,
    alloc_tres: str,
    expected: Decimal,
) -> None:
    """Armis2 partitions use their own rates and max-of-resources weighting."""
    cost = calculate_cost("armis2", "COMPLETED", partition, elapsed, alloc_tres)
    assert cost == expected


@pytest.mark.parametrize(
    "partition",
    [
        "miumbrella",
        "nnallasa",
        "tocho",
        "tocho-rtx6000",
        "sigbio-a40",
        "drjieliu-h200",
        "build",
        "unknown",
    ],
)
def test_armis2_unpriced_partitions(partition: str) -> None:
    """Private and unknown Armis2 partitions are not charged per job."""
    assert (
        estimate_cost("armis2", "COMPLETED", partition, "60", "cpu=1,mem=7G") == "---"
    )


@pytest.mark.parametrize("cluster", ["armis2", "ARMIS2", "GreatLakes", "GREATLAKES"])
def test_cluster_matching_is_case_insensitive(cluster: str) -> None:
    """Cluster names are matched without regard to case."""
    assert estimate_cost(cluster, "COMPLETED", "standard", "60", "cpu=1,mem=7G") != (
        "---"
    )


def test_cluster_rates_are_independent() -> None:
    """The same partition name is priced per cluster."""
    args = ("COMPLETED", "standard", "60", "cpu=1,mem=7G")
    assert estimate_cost("greatlakes", *args) == "$0.00025046"
    assert estimate_cost("armis2", *args) == "$0.00029005"


def test_armis2_cost_matches_slurm_billing() -> None:
    """The estimate agrees with Slurm's billing TRES (weight * 1e-7 dollars)."""
    cost = calculate_cost(
        "armis2", "COMPLETED", "debug", "60", "billing=11601,cpu=4,mem=12G,node=1"
    )
    assert cost is not None
    billing_dollars = Decimal(11601) * Decimal("1e-7")
    assert abs(cost - billing_dollars) / billing_dollars < Decimal("1e-4")
