"""
Parallel processing utility — CASDA pattern.

resolve_workers(workers):
  0  → sequential (no parallelism)
  -1 → auto (cpu_count - 1, minimum 1)
  N  → N workers

run_parallel(fn, tasks, num_workers, desc):
  Sequential when num_workers <= 1 or len(tasks) <= 10.
  ProcessPoolExecutor otherwise.
  fn must be a module-level callable (pickle-safe).
  None results are filtered out.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Any
from tqdm import tqdm
import os


def resolve_workers(workers: int) -> int:
    """Resolve --workers CLI value to actual worker count."""
    if workers < 0:
        cpu_count = os.cpu_count() or 4
        return max(1, cpu_count - 1)
    return workers


def run_parallel(
    fn: Callable,
    tasks: List[Any],
    num_workers: int,
    desc: str = "",
) -> List[Any]:
    """
    Run fn over tasks, sequentially or in parallel.

    Args:
        fn: Module-level callable. Receives one item from tasks.
        tasks: List of arguments, one per fn call.
        num_workers: Worker count from resolve_workers(). 0 = sequential.
        desc: tqdm progress bar label (parallel mode only).

    Returns:
        List of non-None results. Order is not guaranteed in parallel mode.
    """
    use_parallel = num_workers > 1 and len(tasks) >= 2

    if not use_parallel:
        return [r for r in (fn(t) for t in tasks) if r is not None]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fn, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result is not None:
                results.append(result)
    return results
