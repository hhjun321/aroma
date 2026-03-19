import os
import pytest
from utils.parallel import resolve_workers, run_parallel


def _double(x):
    return x * 2


def test_resolve_workers_zero_returns_zero():
    assert resolve_workers(0) == 0


def test_resolve_workers_negative_returns_positive():
    result = resolve_workers(-1)
    assert result >= 1


def test_resolve_workers_positive_unchanged():
    assert resolve_workers(4) == 4


def test_run_parallel_sequential_correctness():
    items = list(range(5))
    results = run_parallel(_double, items, num_workers=0)
    assert sorted(results) == [0, 2, 4, 6, 8]


def test_run_parallel_small_list_uses_sequential():
    """10 or fewer items always run sequentially regardless of workers."""
    items = list(range(5))
    results = run_parallel(_double, items, num_workers=4)
    assert sorted(results) == [0, 2, 4, 6, 8]


def test_run_parallel_large_list_parallel_correct():
    items = list(range(20))
    results = run_parallel(_double, items, num_workers=2)
    assert sorted(results) == sorted(x * 2 for x in items)


def test_run_parallel_filters_none():
    def maybe_none(x):
        return None if x % 2 == 0 else x
    items = list(range(10))
    results = run_parallel(maybe_none, items, num_workers=0)
    assert all(r is not None for r in results)
