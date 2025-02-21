import numpy as np

import params
import process_stages


def compare_series(series_1, series_2, name):
    """Compare two series with detailed error message.

    :param series_1: first series to compare
    :param series_2: second series to compare
    :param name: name of the variable being compared (e.g., 'V', 'X', 'P')
    :raises AssertionError: if the series are not close
    """
    if not np.allclose(series_1, series_2):
        # find where they differ
        diff_mask = ~np.isclose(series_1, series_2)
        # get index of first difference
        diff_idx = series_1.index[np.where(diff_mask)[0][0]]
        raise AssertionError(
            f"difference in {name} at t={diff_idx}: "
            f"{series_1[diff_idx]} != {series_2[diff_idx]}"
        )


def compare_results(cls_1, cls_2, init_kwargs, eval_kwargs, t_max=20):
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

    # test time points
    t = np.linspace(0, t_max, 101)

    # get solutions
    df_1 = stage_1.evaluate_at_t(t, **eval_kwargs)
    df_2 = stage_2.evaluate_at_t(t, **eval_kwargs)

    # compare results
    compare_series(df_1["V"], df_2["V"], "V")
    compare_series(df_1["X"], df_2["X"], "X")
    compare_series(df_1["P"], df_2["P"], "P")


def test_defaults():
    # set up initial conditions from defaults
    defaults = params.defaults["valine_two_stage"]["values"]

    init_kwargs = {
        "V0": defaults["V_batch"],
        "X0": defaults["x_batch"] * defaults["V_batch"],
        "P0": 0,
        "stage_params": {
            k.split("s1_")[-1]: v for k, v in defaults.items() if k.startswith("s1_")
        },
    }

    # test constant feed
    compare_results(
        process_stages.ConstantStageIntegrate,
        process_stages.ConstantStageAnalytical,
        init_kwargs,
        {"F": 0.05},
    )

    # test exponential feed
    compare_results(
        process_stages.ExponentialStageIntegrate,
        process_stages.ExponentialStageAnalytical,
        init_kwargs,
        {"mu": 0.2},
    )

    # test logistic feed
    compare_results(
        process_stages.LogisticStageIntegrate,
        process_stages.LogisticStageAnalytical,
        init_kwargs,
        {"mu": 0.2, "phi_inf": 0.2},
    )
