import numpy as np

EPSILON = np.finfo(float).eps


def get_df_row_with_index(df, idx):
    """
    Get a row from a `pd.DataFrame` as Series including the index and index name(s).

    Looks like there is no proper idiomatic way to do this (when accessing a row with
    `.loc[]`, the index name is lost). Therefore, we use this somewhat hacky approach
    with a dummy numerical index column.
    """
    # find a column name that is not in `df` (if already present, prepend another `_`)
    num_idx_col = "_num_idx"
    while num_idx_col in df.columns:
        num_idx_col = f"_{num_idx_col}"
    # get the numerical index of the row with the given index
    num_idx = df.assign(**{num_idx_col: range(len(df))}).loc[idx, num_idx_col]
    return df.reset_index().iloc[num_idx].rename(None)


def is_iterable(x):
    """Return `True` if `x` is an iterable."""
    try:
        iter(x)
        return True
    except TypeError:
        return False


def get_increasingly_smaller_steps(smaller_than=10):
    """
    Yield increasingly smaller numbers following the pattern 10, 5, 2, 0.1, 0.05, 0.02,
    0.01, 0.005, 0.002, 0.001... that are smaller than `smaller_than`.
    """
    factor = 1
    while True:
        for i in [10, 5, 2]:
            i *= factor
            if i <= smaller_than:
                yield i
        factor *= 0.1


def get_first_nice_value_range_with_at_least_N_values(min_val, max_val, min_n_values):
    """
    Find a value range with nice values and the largest-possible step size (from
    `steps`) that has at least `min_n_values` values.
    """
    for step in get_increasingly_smaller_steps(min_val):
        vals = nice_value_range(min_val, max_val, step)
        if len(vals) >= min_n_values:
            return vals
    # if no range with at least `min_n_values` values was found, return the smallest
    return vals


def nice_value_range(min_val, max_val, step):
    """
    Get a range of values between `min_val` and `max_val` that are multiples of `step`.

    Example:
    >>> nice_value_range(0.12345, 2.8, 0.5)
    [0.5, 1.0, 1.5, 2.0, 2.5]
    """
    high = max_val
    low = min_val - step
    vals = np.arange(high, low, -step)
    return vals[(vals > min_val - EPSILON) & (vals < max_val + EPSILON)][::-1]
