import pandas as pd
import pytest

import grid_search

import utils_for_tests as util


@pytest.mark.parametrize(
    "stage_1_class",
    grid_search.STAGE_1_TYPES,
)
def test_valine_defaults(stage_1_class, test_data_dir, valine_defaults_parsed):
    parsed_params = valine_defaults_parsed
    X_batch = parsed_params["common"]["V_batch"] * parsed_params["common"]["x_batch"]
    s1 = stage_1_class(
        V0=parsed_params["common"]["V_batch"],
        X0=X_batch,
        P0=0,
        **parsed_params["s1"],
    )
    results_df = grid_search.run(
        stage_1=s1,
        input_params=parsed_params,
    )
    # we don't need the index just for comparing
    results_df = results_df.reset_index()
    expected_df = pd.read_csv(
        test_data_dir / f"grid_search_{stage_1_class.feed_type}.csv"
    )
    assert results_df.shape == expected_df.shape
    assert (results_df.columns == expected_df.columns).all()
    for col in results_df.columns:
        util.compare_series(
            results_df[col],
            expected_df[col],
            f"grid_search_comb: {col}",
            atol=1e-12,
        )
