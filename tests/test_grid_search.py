import pandas as pd
import pytest

import grid_search
import process_stages

import utils_for_tests as util


@pytest.mark.parametrize(
    "name, stage_1_class",
    [
        ("const", process_stages.ConstantStageAnalytical),
        ("lin", process_stages.LinearStageConstantGrowthAnalytical),
        ("exp", process_stages.ExponentialStageAnalytical),
    ],
)
def test_valine_defaults(name, stage_1_class, test_data_dir, valine_defaults_parsed):
    parsed_params = valine_defaults_parsed
    X_batch = parsed_params["common"]["V_batch"] * parsed_params["common"]["x_batch"]
    s1 = stage_1_class(
        V0=parsed_params["common"]["V_batch"],
        X0=X_batch,
        P0=0,
        **parsed_params["s1"],
    )
    results = grid_search.run(
        stage_1=s1,
        input_params=parsed_params,
    )
    # we don't need the index just for comparing
    results = results.reset_index()
    expected = pd.read_csv(test_data_dir / f"grid_search_{name}.csv")
    assert results.shape == expected.shape
    assert (results.columns == expected.columns).all()
    for col in results.columns:
        util.compare_series(
            results[col],
            expected[col],
            f"grid_search_comb: {col}",
            atol=1e-12,
        )
