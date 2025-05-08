import numpy as np
import pandas as pd

from process_stages import (
    ExponentialStageAnalytical as ExpS1,
    ConstantStageAnalytical as ConstS1,
    LinearStageConstantGrowthAnalytical as LinS1,
    NoGrowthConstantStage as NoGrowthS2,
)
import util

# define the feed types for the first stage and add some attributes to the corresponding
# classes
STAGE_1_TYPES = [ConstS1, LinS1, ExpS1]

ExpS1.feed_type = "exponential"
ExpS1.growth_param = "mu"
ExpS1.extra_columns = ["substrate_start_volume", "F0", "F_end"]

ConstS1.feed_type = "constant"
ConstS1.growth_param = "F"
ConstS1.extra_columns = ["mu_max"]

LinS1.feed_type = "linear"
LinS1.growth_param = "G"
LinS1.extra_columns = ["F0", "dF", "mu_max", "F_end"]

N_MINIMUM_LEVELS_FOR_GROWTH_PARAM = 15
MU_MIN_FACTOR = 0.05
V_FRAC_STEP = 0.02
ROUND_DIGITS = util.ROUND_DIGITS


def run(stage_1, input_params):
    """
    Perform grid search over `mu` or `F` and `V_frac` using `PARAMS`.

    For each `mu` (or `F`), this first calculates the trajectory of the feed as if it
    was all in the first stage (`V_frac=1`). Then, for each `V_frac` it calculates the
    output of the corresponding two-stage process (with `V_frac` being the fraction of
    the feed volume used in the first stage (exponential with `mu` or constant with `F`)
    and the rest used in the constant phase with `mu=0`). The `pd.DataFrame` with the
    combined results (V, X, P, for stage 1 and stage 2 as well as `t_switch` and `t_end`
    for each `mu|F`--`V_frac` combination) is returned alongside the corresponding
    user parameters.
    """

    mu_max_phys = input_params["s1"]["mu_max_phys"]
    F_max = input_params["common"]["F_max"]
    V_max = input_params["common"]["V_max"]

    # for each feed type we need to make sure we never exceed `F_max` nor `mu`
    if isinstance(stage_1, ExpS1):
        # calculate the `mu_max` that ensures that the feed rate never exceeds
        # `F_max`
        mu_max_F_max = stage_1.calculate_mu_for_F_max(
            F_max=F_max,
            V_end=V_max,
        )
        mu_max_exp_feed = min(input_params["common"]["mu_max_feed"], mu_max_F_max)
        mu_min = mu_max_exp_feed * MU_MIN_FACTOR
        growth_val_range = util.get_range_with_at_least_N_nice_values(
            min_val=mu_min,
            max_val=mu_max_exp_feed,
            min_n_values=N_MINIMUM_LEVELS_FOR_GROWTH_PARAM,
            always_include_max=True,
        )
    elif isinstance(stage_1, LinS1):
        # we test a range of constant absolute (not specific) growth rates `G` from
        # 0 g/h to `G_max`, where `G_max` is such that neither `mu_max` nor `F_max`
        # are exceeded. For a constant absolute growth rate (e.g. 2 g/h) `mu` is
        # largest at `t=0` and `F` is largest at the end of the feed phase
        G_max_mu = stage_1.X0 * mu_max_phys
        G_max_F = stage_1.get_G_max_from_F_max(V_max, F_max)
        # get a range of values of the constant absolute growth rate and round to
        # avoid floating point issues
        growth_val_range = util.get_range_with_at_least_N_nice_values(
            min_val=0,
            max_val=min(G_max_mu, G_max_F),
            min_n_values=N_MINIMUM_LEVELS_FOR_GROWTH_PARAM,
        )
    elif isinstance(stage_1, ConstS1):
        F_max_mu_max = stage_1.calculate_F_from_initial_mu(mu_max_phys)
        growth_val_range = util.get_range_with_at_least_N_nice_values(
            min_val=stage_1.F_min,
            max_val=min(F_max, F_max_mu_max),
            min_n_values=N_MINIMUM_LEVELS_FOR_GROWTH_PARAM,
            always_include_max=True,
        )
    else:
        raise ValueError(f"`stage_1` has unexpected type: {type(stage_1)}")
    growth_val_range = growth_val_range.round(ROUND_DIGITS)
    # get `V_frac` range between 0 and 1
    V_frac_range = np.arange(0, 1 + V_FRAC_STEP, V_FRAC_STEP).round(ROUND_DIGITS)
    V_interval_range = (
        input_params["common"]["V_batch"]
        + (V_max - input_params["common"]["V_batch"]) * V_frac_range
    )

    # initialise empty results df
    df_comb = pd.DataFrame(
        np.nan,
        index=pd.MultiIndex.from_product(
            (growth_val_range, V_frac_range), names=[stage_1.growth_param, "V_frac"]
        ),
        columns=["V1", "X1", "P1", "t_switch", "V2", "X2", "P2", "F2", "t_end"],
    ).sort_index()

    # for each `F`, get the points in the exponential feed at all `V_frac`; then "fill
    # up" until `V_max` with a constant feed rate and calculate productivity based on
    # the end time
    for growth_val in growth_val_range:
        df_s1 = stage_1.evaluate_at_V(
            V=V_interval_range, **{stage_1.growth_param: growth_val}
        )
        df_s1["V_frac"] = V_frac_range
        for t_switch, row_s1 in df_s1.iterrows():
            stage_2 = NoGrowthS2(
                *row_s1[["V", "X", "P"]],
                **input_params["s2"],
            )
            row_s2 = stage_2.evaluate_at_V(V_max).squeeze()
            # get constant feed rate in stage 2 and the end time
            F2 = stage_2.F
            t_end = row_s2.name + t_switch
            df_comb.loc[(growth_val, row_s1["V_frac"])] = [
                *row_s1[["V", "X", "P"]],
                t_switch,
                *row_s2[["V", "X", "P"]],
                F2,
                t_end,
            ]
    # add some extra metrics to the results df
    df_comb = expand_grid_search_results(df_comb, stage_1, input_params)
    return df_comb.round(ROUND_DIGITS)


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
