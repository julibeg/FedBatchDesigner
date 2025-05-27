import pandas as pd
import pytest

from shiny.playwright import controller
from shiny.pytest import create_app_fixture

import utils_for_tests as util

import grid_search

app = create_app_fixture("../../FedBatchDesigner/app.py")


@pytest.fixture(scope="module")
def page(browser, app):
    """
    Fixture to create a new Playwright page for the entire module (the default `page`
    fixture automatically provided by Playwright is just function-scoped and we can't
    use it as fixture in the module-scoped `grid_search_valine_two_stage_defaults`
    below).
    """
    context = browser.new_context()
    page = context.new_page()
    yield page
    # be a good citizen and clean up
    page.close()
    context.close()


def download_grid_search_results(page, feed_type):
    """
    Navigates to the respective results panel and downloads the grid search results.
    """
    controller.NavsetBar(page, "main_navbar").set(f"{feed_type}_results")
    download_btn = controller.DownloadButton(
        page, f"{feed_type}_results-download_grid_search_results"
    )
    with page.expect_download() as download_info:
        download_btn.click()
    return pd.read_csv(download_info.value.path())


def compare_dfs(df1, df2, atol=1e-12):
    assert df1.shape == df2.shape
    assert (df1.columns == df2.columns).all()
    for col in df1.columns:
        util.compare_series(df1[col], df2[col], f"grid_search_comb: {col}", atol=1e-12)


@pytest.fixture(scope="module")
def grid_search_valine_two_stage_defaults(page, app):
    """
    Runs the grid search with the default parameters for the two-stage valine case
    study.
    """
    page.goto(app.url)

    # define controllers
    defaults_open_modal_btn = controller.InputActionButton(page, "populate_defaults")
    defaults_radio_select = controller.InputRadioButtons(page, "selected_defaults")
    defaults_confirm_btn = controller.InputActionButton(page, "apply_defaults")
    submit_btn = controller.InputActionButton(page, "submit")

    defaults_open_modal_btn.click()
    defaults_radio_select.set("valine_two_stage")
    defaults_confirm_btn.click()
    submit_btn.click()


@pytest.mark.parametrize("stage_1_type", grid_search.STAGE_1_TYPES)
def test_grid_search_results(
    page, grid_search_valine_two_stage_defaults, stage_1_type, test_data_dir
):
    """Test the valine grid search results for each feed type."""
    results_df = download_grid_search_results(page, stage_1_type.feed_type)
    expected_df = pd.read_csv(
        test_data_dir / f"grid_search_{stage_1_type.feed_type}.csv"
    )
    compare_dfs(results_df, expected_df)
