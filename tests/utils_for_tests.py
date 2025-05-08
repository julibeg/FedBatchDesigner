import numpy as np
import pandas as pd


def compare_series(series_1, series_2, name=None, rtol=1e-4, atol=1e-8):
    """Compare two series with detailed error message. Logic re. `rtol` and `atol` is
    similar to `np.isclose()` but using the mean instead of `b` for the relative
    difference.

    :param series_1: first series to compare
    :param series_2: second series to compare
    :param name: name of the variable being compared (e.g., 'V', 'X', 'P')
    :raises AssertionError: if the series are not close
    """
    # make sure we got two `pd.Series`
    assert isinstance(series_1, pd.Series)
    assert isinstance(series_2, pd.Series)
    assert series_1.index.equals(series_2.index)

    # find if / where they differe
    means = np.asarray((series_1 + series_2) / 2)
    diffs = np.asarray(series_1 - series_2)
    diff_mask = np.abs(diffs) > atol + rtol * np.abs(means)
    if diff_mask.any():
        # get index of first difference
        diff_idx = series_1.index[np.where(diff_mask)[0][0]]
        name = name or series_1.name or series_2.name
        raise AssertionError(
            "difference "
            + (f"in {name} " if name is not None else "")
            + f"at t={diff_idx}: {series_1[diff_idx]} != {series_2[diff_idx]} "
            + f"(diff = {series_1[diff_idx] - series_2[diff_idx]})"
        )
