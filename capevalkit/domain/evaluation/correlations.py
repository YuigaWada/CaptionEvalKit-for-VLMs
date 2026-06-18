from __future__ import annotations

from collections import Counter
from math import isnan, sqrt
from typing import Iterable

from capevalkit.shared.compat import zip_strict


class _Fenwick:
    def __init__(self, size: int) -> None:
        self.tree = [0] * (size + 1)

    def add(self, index: int, value: int = 1) -> None:
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def sum(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total


def _pair_count(count: int) -> int:
    return count * (count - 1) // 2


def kendall_correlations(values: Iterable[float], targets: Iterable[float]) -> dict[str, float]:
    pairs_list = [(float(value), float(target)) for value, target in zip_strict(values, targets)]
    n = len(pairs_list)
    if n < 2:
        return {"kendall_tau_b": 0.0, "kendall_tau_c": 0.0}

    values_list = [value for value, _ in pairs_list]
    targets_list = [target for _, target in pairs_list]
    if any(isnan(value) for value in values_list) or any(isnan(target) for target in targets_list):
        raise ValueError("Kendall correlation cannot be computed with NaN values")

    pair_count = n * (n - 1) // 2
    tie_x = sum(_pair_count(count) for count in Counter(values_list).values())
    tie_y = sum(_pair_count(count) for count in Counter(targets_list).values())
    tie_both = sum(_pair_count(count) for count in Counter(pairs_list).values())
    comparable = pair_count - tie_x - tie_y + tie_both

    y_ranks = {value: index + 1 for index, value in enumerate(sorted(set(targets_list)))}
    fenwick = _Fenwick(len(y_ranks))
    discordant = 0
    seen = 0
    sorted_pairs = sorted(pairs_list)
    index = 0
    while index < n:
        end = index + 1
        while end < n and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        for _, target in sorted_pairs[index:end]:
            rank = y_ranks[target]
            discordant += seen - fenwick.sum(rank)
        for _, target in sorted_pairs[index:end]:
            fenwick.add(y_ranks[target])
            seen += 1
        index = end

    numerator = comparable - 2 * discordant
    denom_b = sqrt((pair_count - tie_x) * (pair_count - tie_y))
    distinct = min(len(set(values_list)), len(set(targets_list)))
    denom_c = (n * n * (distinct - 1) / (2 * distinct)) if distinct > 1 else 0
    return {
        "kendall_tau_b": 100 * numerator / denom_b if denom_b else 0.0,
        "kendall_tau_c": 100 * numerator / denom_c if denom_c else 0.0,
    }
