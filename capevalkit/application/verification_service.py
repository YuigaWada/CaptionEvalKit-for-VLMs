from __future__ import annotations

from dataclasses import dataclass
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NumericComparison:
    key: str
    actual: float
    expected: float
    actual_display: str
    expected_display: str
    ok: bool


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text())


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, dict):
        flat: dict[str, float] = {}
        for key, item in value.items():
            next_prefix = str(key) if not prefix else f"{prefix}.{key}"
            flat.update(_flatten_numbers(item, next_prefix))
        return flat
    return {}


def _round_decimal(value: float, decimals: int) -> Decimal:
    exponent = Decimal("1").scaleb(-decimals)
    return Decimal(str(value)).quantize(exponent, rounding=ROUND_HALF_UP)


def _display_raw(value: float) -> str:
    return f"{value:.12g}"


def compare_results(
    results_path: str,
    expected_path: str,
    *,
    tolerance: float = 1e-4,
    round_decimals: int | None = None,
) -> list[NumericComparison]:
    results = _flatten_numbers(_load(results_path))
    expected = _flatten_numbers(_load(expected_path))
    if not expected:
        raise ValueError(f"{expected_path} has no numeric paper values")

    missing = sorted(key for key in expected if key not in results)
    if missing:
        raise AssertionError(f"missing result values: {', '.join(missing)}")

    comparisons = []
    tolerance_decimal = Decimal(str(tolerance))
    for key, expected_value in expected.items():
        actual_value = results[key]
        if round_decimals is not None:
            actual_rounded = _round_decimal(actual_value, round_decimals)
            expected_rounded = _round_decimal(expected_value, round_decimals)
            comparisons.append(
                NumericComparison(
                    key=key,
                    actual=actual_value,
                    expected=expected_value,
                    actual_display=str(actual_rounded),
                    expected_display=str(expected_rounded),
                    ok=abs(actual_rounded - expected_rounded) <= tolerance_decimal,
                )
            )
            continue
        comparisons.append(
            NumericComparison(
                key=key,
                actual=actual_value,
                expected=expected_value,
                actual_display=_display_raw(actual_value),
                expected_display=_display_raw(expected_value),
                ok=abs(actual_value - expected_value) <= tolerance,
            )
        )
    return comparisons


def verify_results(
    results_path: str,
    expected_path: str,
    *,
    tolerance: float = 1e-4,
    round_decimals: int | None = None,
) -> None:
    mismatches = []
    for comparison in compare_results(
        results_path,
        expected_path,
        tolerance=tolerance,
        round_decimals=round_decimals,
    ):
        if comparison.ok:
            continue
        if round_decimals is not None:
            mismatches.append(
                f"{comparison.key}: actual={comparison.actual} rounded={comparison.actual_display} "
                f"expected={comparison.expected} rounded_expected={comparison.expected_display}"
            )
            continue
        mismatches.append(f"{comparison.key}: actual={comparison.actual} expected={comparison.expected}")
    if mismatches:
        raise AssertionError("; ".join(mismatches))
