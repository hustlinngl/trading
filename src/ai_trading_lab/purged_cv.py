from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PurgedSplit:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge_bars: int


def walk_forward_splits(n_rows: int, train_bars: int, test_bars: int, step_bars: int,
                        purge_bars: int, min_train_bars: int = 1, independent_test: bool = False):
    n_rows = int(n_rows); train_bars = int(train_bars); test_bars = int(test_bars)
    step_bars = max(1, int(step_bars)); purge_bars = max(0, int(purge_bars))
    if independent_test:
        step_bars = max(step_bars, test_bars)
    min_train_bars = max(1, int(min_train_bars))
    cursor = train_bars + purge_bars
    fold = 0
    while cursor + test_bars <= n_rows:
        train_end = cursor - purge_bars
        train_start = max(0, train_end - train_bars)
        if train_end - train_start >= min_train_bars:
            yield PurgedSplit(fold, train_start, train_end, cursor, cursor + test_bars, purge_bars)
            fold += 1
        cursor += step_bars
