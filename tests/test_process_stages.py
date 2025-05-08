import numpy as np
import scipy
import pytest

import params
import process_stages

# utils for tests
import utils_for_tests as util

DEFAULTS = params.defaults["valine_two_stage"]["values"]
V_BATCH = DEFAULTS["V_batch"]
X_BATCH = V_BATCH * DEFAULTS["x_batch"]

N_POINTS = 101
TIME_POINTS = np.linspace(0, 20, N_POINTS)


def compare_results(cls_1, cls_2, init_kwargs, eval_kwargs_1, eval_kwargs_2=None):
    """Compare results from two `FedBatchStage` classes.

    :param cls_1: first class to compare (e.g., `ConstantStageIntegrate`)
    :param cls_2: second class to compare (e.g., `ConstantStageAnalytical`)
    :param init_kwargs: dictionary of kwargs for stage initialization
    :param eval_kwargs: dictionary of kwargs for `evaluate_at_t` (e.g., {'F': 0.05})
    :param t_max: maximum time for simulation (default: 20)
    """
    # create both stages with same parameters
    stage_1 = cls_1(**init_kwargs)
    stage_2 = cls_2(**init_kwargs)

    # if `eval_kwargs_s2` is not provided, use `eval_kwargs_s1`
    if eval_kwargs_2 is None:
        eval_kwargs_2 = eval_kwargs_1

    # get solutions
    df_1 = stage_1.evaluate_at_t(TIME_POINTS, **eval_kwargs_1)
    df_2 = stage_2.evaluate_at_t(TIME_POINTS, **eval_kwargs_2)

    # compare results
    util.compare_series(
        df_1["V"], df_2["V"], f"{cls_1.__name__} vs {cls_2.__name__}: V"
    )
    util.compare_series(
        df_1["X"], df_2["X"], f"{cls_1.__name__} vs {cls_2.__name__}: X"
    )
    util.compare_series(
        df_1["P"], df_2["P"], f"{cls_1.__name__} vs {cls_2.__name__}: P"
    )


@pytest.fixture
def default_init_kwargs():
    # set up initial conditions from defaults
    init_kwargs = {
        "V0": V_BATCH,
        "X0": X_BATCH,
        "P0": 10,
        **{k.split("s1_")[-1]: v for k, v in DEFAULTS.items() if k.startswith("s1_")},
    }
    return init_kwargs


def test_defaults(default_init_kwargs):
    # test constant feed
    compare_results(
        process_stages.ConstantStageIntegrate,
        process_stages.ConstantStageAnalytical,
        default_init_kwargs,
        {"F": 0.05},
    )

    # test exponential feed
    compare_results(
        process_stages.ExponentialStageIntegrate,
        process_stages.ExponentialStageAnalytical,
        default_init_kwargs,
        {"mu": 0.2},
    )

    # test logistic feed
    compare_results(
        process_stages.LogisticStageIntegrate,
        process_stages.LogisticStageAnalytical,
        default_init_kwargs,
        {"mu": 0.2, "F_inf": 0.2},
    )

    # test linear feed
    compare_results(
        process_stages.LinearStageIntegrate,
        process_stages.LinearStageAnalytical,
        default_init_kwargs,
        {"dF": 0.05, "F0": 0.1},
    )

    # test linear feed with constant absolute growth
    compare_results(
        process_stages.LinearStageConstantGrowthIntegrate,
        process_stages.LinearStageConstantGrowthAnalytical,
        default_init_kwargs,
        {"G": 1},
    )


def test_exp_feed_F0(default_init_kwargs):
    """
    Test that the initial feed rate `F0` calculated by `ExponentialFeed` indeed ensures
    `dX/dt = mu * X` for the whole feed duration.
    """
    stage = process_stages.ExponentialStageIntegrate(**default_init_kwargs)
    # get `F0` and assert that this indeed leads to `dX/dt = mu * X`
    mu = np.random.uniform(0, 1)
    V_end = np.random.uniform(V_BATCH + 0.5, V_BATCH + 20)
    df = stage.evaluate_at_V(
        V=np.linspace(V_BATCH, V_end, N_POINTS),
        mu=mu,
    )
    util.compare_series(df.eval("X * @mu"), df["dX"])


def test_exp_feed_mu_for_F_max(default_init_kwargs):
    stage = process_stages.ExponentialStageAnalytical(**default_init_kwargs)
    # get random values for `V_end` (between 3.5 and 23 L) and `F_max` (between 0.01 and
    # 1 L/h)
    V_end = np.random.uniform(V_BATCH + 0.5, V_BATCH + 20)
    F_max = 10 ** np.random.uniform(-2, 0)
    # get `mu` using the analytical expression and check that `F` at `V=V_end` is indeed
    # `F_max`
    mu = stage.calculate_mu_for_F_max(F_max, V_end)
    t_end = stage.t_until_V(V=V_end, mu=mu)
    F_end = stage.dV(mu, t=t_end)
    assert np.isclose(F_end, F_max)


def test_lin_feed_with_constant_growth(default_init_kwargs):
    # `test_defaults` already makes sure there are no differences between the numerical
    # and analytical solutions, but it doesn't check that the absolute growth rate is
    # indeed constant

    # we first check that the analytical solution is the same as the numerical one (but
    # the numerical uses `F0` and `dF`); define a dummy stage to get `F0` and `dF`
    dummy_stage = process_stages.LinearStageConstantGrowthAnalytical(
        **default_init_kwargs
    )
    G = 2  # constant growth rate in g/h
    F0, df = dummy_stage.F0_and_dF_for_constant_growth(G)
    compare_results(
        process_stages.LinearStageIntegrate,
        process_stages.LinearStageConstantGrowthAnalytical,
        default_init_kwargs,
        eval_kwargs_1={"F0": F0, "dF": df},
        eval_kwargs_2={"G": G},
    )

    # now we check that the absolute growth rate is indeed constant (i.e. that it's the
    # same at each time step)
    stage = process_stages.LinearStageConstantGrowthAnalytical(**default_init_kwargs)
    df = stage.evaluate_at_t(TIME_POINTS, G=G)

    delta_X = df["X"].diff().dropna()
    assert np.allclose(delta_X, delta_X.iloc[0])


def test_lin_feed_with_constant_growth_G_max_from_F_max(default_init_kwargs):
    """
    Get `G_max` using Newton's method and check that we get the same value as the
    analytical expression in `LinearFeedConstantGrowth.get_G_max_from_F_max()`.
    """
    stage = process_stages.LinearStageConstantGrowthAnalytical(**default_init_kwargs)
    # get random values for `V_end` (between 3.5 and 23 L) and `F_max` (between 0.01 and
    # 1 L/h)
    V_end = np.random.uniform(V_BATCH + 0.5, V_BATCH + 20)
    F_max = 10 ** np.random.uniform(-2, 0)
    G_max_analytical = stage.get_G_max_from_F_max(V_end=V_end, F_max=F_max)

    def diff_to_F_max(G):
        """
        Get the difference between `F_max` and the feed rate corresponding to `G` at
        `V=V_end`.
        """
        t_end = stage.t_until_V(V=V_end, G=G)
        F_end = stage.dV(G, t=t_end)
        return F_end - F_max

    G_max_newton = scipy.optimize.newton(func=diff_to_F_max, x0=0, disp=True)
    assert np.isclose(G_max_analytical, G_max_newton)
