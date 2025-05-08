import numpy as np

EPSILON = np.finfo(float).eps
ROUND_DIGITS = 5


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
    Yield increasingly smaller numbers following the pattern 10, 5, 2, 1, 0.5, 0.2, 0.1,
    0.05, 0.02, 0.01, 0.005, 0.002, 0.001... that are smaller than `smaller_than`.
    """
    factor = 1
    while True:
        for i in [10, 5, 2]:
            i *= factor
            if i <= smaller_than:
                yield i
        factor *= 0.1


def get_range_with_at_least_N_nice_values(
    min_val,
    max_val,
    min_n_values,
    always_include_min=False,
    always_include_max=False,
):
    """
    Find a value range with "nice" values and the largest-possible step size (from
    `steps`) that has at least `min_n_values` values.
    """
    for step in get_increasingly_smaller_steps(max_val - min_val):
        vals = nice_value_range(min_val, max_val, step)
        if len(vals) >= min_n_values:
            break
    # if no range with at least `min_n_values` values was found, the for loop finished
    # and we return the range with the smallest steps
    if always_include_min:
        if np.isclose(vals[0], min_val):
            # replace in case there are tiny differences due to rounding etc.
            vals[0] = min_val
        else:
            vals = np.insert(vals, 0, min_val)
    if always_include_max:
        if np.isclose(vals[-1], max_val):
            # replace in case there are tiny differences due to rounding etc.
            vals[-1] = max_val
        else:
            vals = np.append(vals, max_val)
    return vals


def nice_value_range(min_val, max_val, step):
    """
    Get a range of values between `min_val` and `max_val` that are multiples of `step`.

    Example:
    >>> nice_value_range(0.12345, 2.8, 0.5)
    [0.5, 1.0, 1.5, 2.0, 2.5]
    """
    high = max_val
    low = min_val - (min_val % step)
    vals = np.arange(low, high, step)
    return vals[(vals > min_val - EPSILON) & (vals < max_val + EPSILON)]


def quadratic_formula(a, b, c, plus_only=False):
    if abs(a) < EPSILON:
        # linear case: a ~ 0 -> b * t + c = 0 -> t = -c / b
        if abs(b) < EPSILON:
            raise ValueError(
                "No solution or infinite solutions in quadratic formula "
                "(`a` and `b` are both close to zero)."
            )
        t = -c / b
        return t if plus_only else (t, t)

    # convert into array if not already
    disc = np.array(b**2 - 4 * a * c)
    if (disc < 0).any():
        raise ValueError("Complex roots in quadratic formula (discriminant < 0).")

    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    if plus_only:
        return t1
    t2 = (-b - sqrt_disc) / (2 * a)
    return (t1, t2)


def expand_grid_search_results(df_comb, stage_1, input_params):
    """
    Calculate a few extra metrics (final biomass and product concentration,
    productivity, space-time yield, etc.) for the resutls df of a grid search.
    """
    df_comb["x2"] = df_comb["X2"] / df_comb["V2"]
    df_comb["p2"] = df_comb["P2"] / df_comb["V2"]
    df_comb["productivity"] = df_comb["P2"] / df_comb["t_end"]
    df_comb["space_time_yield"] = df_comb["productivity"] / df_comb["V2"]
    # calculate total amount of substrate added and per-substrate yield
    V_add_s1 = df_comb["V1"] - input_params["common"]["V_batch"]
    V_add_s2 = df_comb["V2"] - df_comb["V1"]
    S_add_s1 = V_add_s1 * input_params["s1"]["s_f"]
    S_add_s2 = V_add_s2 * input_params["s2"]["s_f"]
    df_comb.insert(1, "S1", S_add_s1)
    df_comb.insert(6, "S2", S_add_s1 + S_add_s2)
    df_comb["substrate_yield"] = df_comb["P2"] / df_comb["S2"]

    # if exponential, add substrate start volume; if constant, add mu in first instance
    # of feed (as this will be the largest mu encountered in the feed phase)
    if isinstance(stage_1, ExpS1):
        # exponential feed --> add substrate start volume and initial feed rate for each
        # `mu` to the results
        for mu, df in df_comb.groupby("mu"):
            df_comb.loc[(mu, slice(None)), "substrate_start_volume"] = (
                stage_1.substrate_start_volume(mu=mu)
            )
            df_comb.loc[(mu, slice(None)), "F0"] = stage_1.F0(mu=mu)
            df_comb.loc[(mu, slice(None)), "F_end"] = stage_1.dV(
                mu=mu, t=df["t_switch"].iloc[-1]
            ).max()
    elif isinstance(stage_1, ConstS1):
        # constant feed --> add maximum growth rate (at first instance of feed)
        for F, _ in df_comb.groupby("F"):
            df_comb.loc[(F, slice(None)), "mu_max"] = (
                stage_1.calculate_initial_mu_from_F(F)
            )
    elif isinstance(stage_1, LinS1):
        # linear feed with constant absolute growth --> add maximum specific growth rate
        # and feed rate
        for G, df in df_comb.groupby("G"):
            F0, dF = stage_1.F0_and_dF_for_constant_growth(G=G)
            df_comb.loc[(G, slice(None)), "F0"] = F0
            df_comb.loc[(G, slice(None)), "dF"] = dF
            df_comb.loc[(G, slice(None)), "mu_max"] = (
                stage_1.calculate_initial_mu_from_F(F0)
            )
            df_comb.loc[(G, slice(None)), "F_end"] = stage_1.dV(
                G=G, t=df["t_switch"].iloc[-1]
            )

    return df_comb
