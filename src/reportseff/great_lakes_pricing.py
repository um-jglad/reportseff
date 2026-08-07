"""Estimate Great Lakes job costs from allocated Slurm resources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

PRICING_DATE = "2026-08-07"
UNAVAILABLE = "---"
GIB = Decimal(1024) ** 3
MONEY_QUANTUM = Decimal("0.01")
MEMORY_RE = re.compile(r"(?P<amount>\d+(?:\.\d+)?)(?P<unit>[KMGTPE]?)")
MEMORY_MULTIPLIERS = {
    "": Decimal(1),
    "K": Decimal(1024),
    "M": Decimal(1024) ** 2,
    "G": GIB,
    "T": Decimal(1024) ** 4,
    "P": Decimal(1024) ** 5,
    "E": Decimal(1024) ** 6,
}


@dataclass(frozen=True)
class PricingProfile:
    """Per-minute price and the resources represented by one billing unit."""

    rate: Decimal
    cpu_unit: Decimal
    memory_unit_gib: Decimal
    gpu_unit: Decimal | None = None


STANDARD = PricingProfile(Decimal("0.000250463"), Decimal(1), Decimal(7))
PRICING_PROFILES = {
    "standard": STANDARD,
    "debug": STANDARD,
    "viz": STANDARD,
    "largemem": PricingProfile(Decimal("0.000770370"), Decimal(1), Decimal("41.75")),
    "gpu": PricingProfile(Decimal("0.002739120"), Decimal(20), Decimal(90), Decimal(1)),
    "spgpu": PricingProfile(
        Decimal("0.001807870"), Decimal(4), Decimal(48), Decimal(1)
    ),
    "gpu-rtx6000": PricingProfile(
        Decimal("0.003861015"), Decimal(16), Decimal(192), Decimal(1)
    ),
    "gpu_mig40": PricingProfile(
        Decimal("0.002739120"), Decimal(8), Decimal(125), Decimal(1)
    ),
}


def estimate_cost(
    cluster: str,
    state: str,
    partition: str,
    elapsed_seconds: str,
    alloc_tres: str,
) -> str:
    """Return the estimated list-price cost of a completed Great Lakes job.

    Args:
        cluster: Slurm cluster name.
        state: Base Slurm job state.
        partition: Slurm partition name.
        elapsed_seconds: Runtime in seconds from ``ElapsedRaw``.
        alloc_tres: Allocated resources from ``AllocTRES``.

    Returns:
        A dollar value rounded to the nearest cent, or ``---`` when an
        estimate is unavailable.
    """
    return format_cost(
        calculate_cost(cluster, state, partition, elapsed_seconds, alloc_tres)
    )


def calculate_cost(
    cluster: str,
    state: str,
    partition: str,
    elapsed_seconds: str,
    alloc_tres: str,
) -> Decimal | None:
    """Return an unrounded Great Lakes list-price cost when available."""
    if state.casefold() in {"running", "pending"}:
        return None
    if cluster.casefold() != "greatlakes":
        return None

    profile = PRICING_PROFILES.get(partition.casefold())
    if profile is None:
        return None

    try:
        elapsed_minutes = _parse_nonnegative_decimal(elapsed_seconds) / Decimal(60)
        cpus, memory_gib, gpus = _parse_alloc_tres(alloc_tres)
    except ValueError:
        return None

    weighted_resources = [
        cpus / profile.cpu_unit,
        memory_gib / profile.memory_unit_gib,
    ]
    if profile.gpu_unit is not None:
        weighted_resources.append(gpus / profile.gpu_unit)

    return elapsed_minutes * profile.rate * max(weighted_resources)


def format_cost(cost: Decimal | None) -> str:
    """Format a cost to the nearest cent, or return the unavailable marker."""
    if cost is None:
        return UNAVAILABLE

    rendered = cost.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    return f"${rendered:.2f}"


def _parse_alloc_tres(alloc_tres: str) -> tuple[Decimal, Decimal, Decimal]:
    """Parse CPU, memory, and GPU allocations from an AllocTRES value."""
    values: dict[str, str] = {}
    typed_gpu_values: list[str] = []

    if not alloc_tres:
        msg = "AllocTRES is empty"
        raise ValueError(msg)

    for item in alloc_tres.split(","):
        if item.count("=") != 1:
            msg = f"Malformed AllocTRES item: {item!r}"
            raise ValueError(msg)
        key, value = item.split("=", 1)
        if not key or not value:
            msg = f"Malformed AllocTRES item: {item!r}"
            raise ValueError(msg)
        if key in values:
            msg = f"Duplicate AllocTRES item: {key!r}"
            raise ValueError(msg)
        values[key] = value
        if key.startswith("gres/gpu:"):
            typed_gpu_values.append(value)

    if "cpu" not in values or "mem" not in values:
        msg = "AllocTRES must contain cpu and mem"
        raise ValueError(msg)

    cpus = _parse_nonnegative_decimal(values["cpu"])
    memory_gib = _parse_memory_gib(values["mem"])
    if "gres/gpu" in values:
        gpus = _parse_nonnegative_decimal(values["gres/gpu"])
    else:
        gpus = sum(
            (_parse_nonnegative_decimal(value) for value in typed_gpu_values),
            start=Decimal(0),
        )
    return cpus, memory_gib, gpus


def _parse_nonnegative_decimal(value: str) -> Decimal:
    """Parse a finite, nonnegative decimal value."""
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        msg = f"Invalid decimal value: {value!r}"
        raise ValueError(msg) from error
    if not result.is_finite() or result < 0:
        msg = f"Invalid decimal value: {value!r}"
        raise ValueError(msg)
    return result


def _parse_memory_gib(value: str) -> Decimal:
    """Parse a Slurm memory value and return GiB."""
    match = re.fullmatch(MEMORY_RE, value)
    if match is None:
        msg = f"Invalid memory value: {value!r}"
        raise ValueError(msg)
    amount = _parse_nonnegative_decimal(match.group("amount"))
    return amount * MEMORY_MULTIPLIERS[match.group("unit")] / GIB
