import contextlib
import functools
import time

import numpy as np
import pandas as pd
from shiny import reactive
from shiny.express import input, module, render, session as root_session, ui
from shiny_validate import InputValidator
from shinywidgets import render_plotly

import colors
import icons
import info
from logger import logger
import params
import plots
import util
from process_stages import (
    ExponentialStageAnalytical as ExpS1,
    ConstantStageAnalytical as ConstS1,
    LinearStageConstantGrowthAnalytical as LinS1,
    NoGrowthConstantStage as NoGrowthS2,
)

# some global variables
NAVBAR_OPTIONS = ui.navbar_options(bg="#efefef")
APP_NAME = "FedBatchDesigner"
MAIN_NAVBAR_ID = "main_navbar"
N_MINIMUM_LEVELS_FOR_GROWTH_PARAM = 15
MU_MIN_FACTOR = 0.05
V_FRAC_STEP = 0.02
ROUND_DIGITS = util.ROUND_DIGITS
N_CONTOURS = 30

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

# reactive values
GRID_SEARCH_DFS = {stage_cls: reactive.value(None) for stage_cls in STAGE_1_TYPES}
PARSED_PARAMS = reactive.value(None)

ui.page_opts(title="FedBatchDesigner", full_width=True, id="page")

# include MathJax CDN in the head of the HTML document; to render formulas in Shiny HTML
# elements, use `\(...\)`; in plotly plots use `$...$`.
ui.head_content(
    # it looks like only the CDN script is needed and the other two scripts below
    # actually don't seem to be necessary (but they are added when exporting a plot
    # with `plotly.io.to_html()`)
    ui.tags.script(
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/"
        + "MathJax.js?config=TeX-AMS-MML_SVG"
    ),
    # MathJax configuration
    ui.tags.script(
        """
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {
            window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});
        }
        """,
        type="text/javascript",
    ),
    # Plotly MathJax configuration
    ui.tags.script(
        "window.PlotlyConfig = {MathJaxConfig: 'local'};",
        type="text/javascript",
    ),
    # smaller font size for the navbar titles
    ui.tags.style(
        """
        .navbar-brand {
            font-size: 115%;
        }
        """
    ),
)


@contextlib.contextmanager
def card_with_nav_header(title):
    """
    Create a card with a navset_bar (with a single title-less panel) as header.

    Instead of using `ui.card_header()` we're using this hack because it looks nicer and
    is more consistent with the navset_bar and the two panels for the stage-specific
    params.
    """
    with ui.card():
        with ui.navset_bar(
            title=title, navbar_options=ui.navbar_options(bg="#efefef", underline=False)
        ):
            with ui.nav_panel(None):
                yield


def go_to_navbar_panel(panel):
    # we define this function for updating the navbar in order to capture the
    # `root_session` because `ui.update_navs()` per default always uses the enclosing
    # session when the function is called, which might be the session of a module and
    # not the root session (in which case the navbar would not be updated;
    # https://github.com/posit-dev/py-shiny/issues/1841)
    ui.update_navs(id=MAIN_NAVBAR_ID, selected=panel, session=root_session)


def run_grid_search(stage_1, input_params):
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
    )

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
                s_f=input_params["common"]["s_f"],
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
    # calculate a few extra metrics (final biomass and product concentration,
    # productivity, space-time yield, etc.)
    df_comb["x2"] = df_comb["X2"] / df_comb["V2"]
    df_comb["p2"] = df_comb["P2"] / df_comb["V2"]
    df_comb["productivity"] = df_comb["P2"] / df_comb["t_end"]
    df_comb["space_time_yield"] = df_comb["productivity"] / df_comb["V2"]
    # calculate total amount of substrate added and per-substrate yield
    V_add_s1 = df_comb["V1"] - input_params["common"]["V_batch"]
    V_add_s2 = df_comb["V2"] - df_comb["V1"]
    S_add_s1 = V_add_s1 * input_params["common"]["s_f"]
    S_add_s2 = V_add_s2 * input_params["common"]["s_f"]
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
    return df_comb.round(ROUND_DIGITS)


@reactive.Effect
@reactive.event(input.submit)
def submit_button():
    """
    Validate the input params, run the grid search (while showing a modal), and jump to
    the "Results" panel.
    """

    try:
        if not input_validator.is_valid():
            # inputs are not valid; show a notification and return
            ui.notification_show(
                "Invalid input parameters. Please check the fields marked in red.",
                type="error",
                duration=10,
            )
            return

        # parse the input parameters
        parse_params()
        parsed_params = PARSED_PARAMS.get()

        # define the constant and exponential first stage and make sure the params are
        # feasible
        X_batch = (
            parsed_params["common"]["x_batch"] * parsed_params["common"]["V_batch"]
        )
        stage_instances = {
            stage_type: stage_type(
                V0=parsed_params["common"]["V_batch"],
                X0=X_batch,
                P0=0,
                s_f=parsed_params["common"]["s_f"],
                **parsed_params["s1"],
            )
            for stage_type in STAGE_1_TYPES
        }

        # (for now we only check if `F_max` is smaller than the minimum feed rate
        # required to add enough medium in the first instant of constant feed, but we
        # should add more checks in the future)
        if stage_instances[ConstS1].F_min > parsed_params["common"]["F_max"]:
            # parameters are infeasible; `F_max` needs to be increased
            ui.notification_show(
                """
                Submitted parameters are infeasible. The minimum F value required in
                order to add enough feed medium in the first instant of the constant
                feed phase is larger than the maximum F value provided. Increase 'F_max'
                or adjust other parameters (e.g. X_batch, s_f, maintenance requirement,
                etc).
                """,
                type="error",
                duration=30,
            )
            return
        PARSED_PARAMS.set(parsed_params)

        # show a modal while the grid search is running
        m = ui.modal(
            title="Calculating results...",
            footer="This may take a moment.",
            easy_close=True,
        )
        ui.modal_show(m)

        for stage_cls, instance in stage_instances.items():
            GRID_SEARCH_DFS[stage_cls].set(run_grid_search(instance, parsed_params))

        # calculations are done; remove the modal and jump to "Results"
        ui.modal_remove()
        go_to_navbar_panel("Results constant feed")

    except ZeroDivisionError:
        ui.notification_show(
            """
            Divide by zero error. Please check the input parameters and make sure that
            all values are non-zero.
            """,
            type="error",
            duration=10,
        )
        time.sleep(2)
        ui.modal_remove()

    except Exception as e:
        logger.exception(e)

        ui.notification_show(
            """
            An error occurred. Please check the input parameters and make sure that all
            values are valid.
            """,
            type="error",
            duration=10,
        )
        time.sleep(2)
        ui.modal_remove()


@module
def results(input, output, session, stage_1_type):
    growth_param = stage_1_type.growth_param
    feed_type = stage_1_type.feed_type

    @render.express
    def _results():
        linked_plots = []
        parsed_params = PARSED_PARAMS.get()
        grid_search_df = GRID_SEARCH_DFS[stage_1_type].get()
        selected_process = reactive.Value({})

        # check if the grid search has been run; if not (i.e. we don't have results
        # yet), show a message and return
        if grid_search_df is None:
            with ui.card():
                ui.h4("No results yet", style="text-align: center;")
                with ui.div(style="justify-content: center; display: flex"):
                    ui.input_action_button(
                        "back_button",
                        "Back to inputs",
                        class_="btn btn-primary",
                    )
            return

        ui.tags.p(
            f"""
            Concentration profiles were calculated for different values of
            {growth_param} (the {params.results[growth_param].description.lower()}) and
            the {params.results["V_frac"].description.lower()}.
            """,
            ui.tags.br(),
            "You can click into one of the plots to select a process.",
        )

        # three columns:
        # - left: contour plots of productivity and yield
        # - center: productivity-vs-yield traces and plot with the trajectory of the
        #   selected process
        # - right: table-like details of the optimal and selected process
        #
        # shiny uses the bootstrap 12-wide grid layout system (which means we can
        # only use integers between 1 and 12 to specify column widths). for finer
        # control, we first define two columns of widths 9, 3 and then split the
        # first column into two of widths 6, 6 (since we can't have three columns of
        # widths 4.5, 4.5, 3)

        with ui.layout_columns(col_widths=(9, 3)):
            with ui.layout_columns(col_widths=(6, 6)):
                with ui.card():
                    # put the two contour plots (productivity and yield; each with
                    # `V_frac` and `mu` on the axes) into the left column
                    @render_plotly
                    def productivity_contour_plot():
                        return plots.ContourPlot(
                            linked_plots=linked_plots,
                            selected_process=selected_process,
                            df=grid_search_df,
                            growth_param=growth_param,
                            x_col="V_frac",
                            y_col=growth_param,
                            z_col="space_time_yield",
                            text_col="p2",
                            n_contours=N_CONTOURS,
                        ).fig

                    @render_plotly
                    def yield_contour_plot():
                        return plots.ContourPlot(
                            linked_plots=linked_plots,
                            selected_process=selected_process,
                            df=grid_search_df,
                            growth_param=growth_param,
                            x_col="V_frac",
                            y_col=growth_param,
                            z_col="p2",
                            text_col="space_time_yield",
                            n_contours=N_CONTOURS,
                        ).fig

                with ui.card():
                    # the productivity-vs-yield traces and plot with the trajectory
                    # of the selected process go into the center column

                    @render_plotly
                    def prod_vs_yield_plot():
                        return plots.LinePlot(
                            linked_plots=linked_plots,
                            selected_process=selected_process,
                            df=grid_search_df,
                            growth_param=growth_param,
                            x_col="p2",
                            y_col="space_time_yield",
                        ).fig

                    @render_plotly
                    def selected_process_plot():
                        return plots.SelectedProcessPlot(
                            linked_plots=linked_plots,
                            selected_process=selected_process,
                            df=grid_search_df,
                            growth_param=growth_param,
                            params=parsed_params,
                            stage_1_class=stage_1_type,
                            stage_2_class=NoGrowthS2,
                        ).fig

            # this is the column with width 3; it contains a button to download the grid
            # search results as CSV and the details of the optimal and selected
            # processes
            with ui.div():
                with ui.card():
                    with ui.card_header():
                        with ui.tooltip():
                            ui.tags.span(
                                "Download grid search results ", icons.question_circle
                            )
                            """
                            Download the results of the grid search as a CSV file. The
                            file contains the values for titer, space-time yield, and
                            per-substrate yield for each simulated feed profile.
                            """

                    @render.download(
                        label="Download CSV",
                        filename=(
                            f"{APP_NAME}_grid_search_results_{feed_type}_feed.csv"
                        ),
                    )
                    def download_grid_search_results():
                        yield grid_search_df.to_csv()

                metrics_for_tables = stage_1_type.extra_columns + [
                    growth_param,
                    "V_frac",
                    "V1",
                    "V2",
                    "F2",
                    "X2",
                    "x2",
                    "P2",
                    "p2",
                    "t_end",
                    "t_switch",
                    "space_time_yield",
                    "substrate_yield",
                ]

                @render.express
                def optimal_process_details():
                    with ui.tags.div(id="optimal-details"):
                        # add style tag to change color of all text in the div
                        ui.tags.style(
                            "#optimal-details * { color: %s; }"
                            % colors.optimal_marker_color
                        )

                        opt_row = util.get_df_row_with_index(
                            grid_search_df, grid_search_df["productivity"].idxmax()
                        )

                        with ui.card():
                            with ui.card_header():
                                with ui.tooltip():
                                    ui.tags.span(
                                        "Optimal process ", icons.question_circle
                                    )
                                    """
                                    Parameters of the process that maximizes the average
                                    volumetric productivity (i.e. called space-time
                                    yield) are listed below. Hover over the labels for
                                    details on the individual parameters. Note that the
                                    batch phase is not included when calculating the TRY
                                    metrics (final titer, space-time yield,
                                    per-substrate yield).
                                    """

                            with ui.p():
                                for p in metrics_for_tables:
                                    with ui.tooltip():
                                        ui.tags.b(f"{params.results[p].label}:")
                                        params.results[p].description
                                    ui.tags.span(
                                        f"{opt_row[p]:.3g} {params.results[p].unit}",
                                        style="float: right;",
                                    )
                                    ui.br()

                            @render.download(
                                label="Download CSV",
                                filename=(
                                    f"{APP_NAME}_optimal_process_{feed_type}_feed.csv"
                                ),
                            )
                            def download_optimal():
                                yield opt_row.to_csv(
                                    index_label="param", header=["value"]
                                )

                @render.express
                def selected_process_details():
                    with ui.tags.div(id="selected-details"):
                        # add style tag to change color of all text in the div
                        ui.tags.style(
                            "#selected-details * { color: %s; }"
                            % colors.selected_marker_color
                        )

                        with ui.card():
                            with ui.card_header():
                                with ui.tooltip():
                                    ui.tags.span(
                                        "Selected process ", icons.question_circle
                                    )
                                    """
                                    Parameters of the selected process are listed below.
                                    Hover over the labels for details on the individual
                                    parameters. Note that the batch phase is not
                                    included when calculating the TRY metrics (final
                                    titer, space-time yield, per-substrate yield).
                                    """

                            sel_params = selected_process.get()
                            if sel_params:
                                sel_row = util.get_df_row_with_index(
                                    grid_search_df,
                                    (sel_params[growth_param], sel_params["V_frac"]),
                                )

                            with ui.p():
                                for p in metrics_for_tables:
                                    with ui.tooltip():
                                        ui.tags.b(f"{params.results[p].label}:")
                                        params.results[p].description
                                    ui.tags.span(
                                        (
                                            f"{sel_row[p]:.3g} {params.results[p].unit}"
                                            if sel_params
                                            else "-"
                                        ),
                                        style="float: right;",
                                    )
                                    ui.br()

                            if not sel_params:
                                # no process been selected yet; shiny uses a link
                                # (styled like a button) for downloads, which can't be
                                # disabled. We therefore use a dummy disabled button
                                # instead that will be replaced by the download button
                                # once a process is selected

                                ui.input_action_button(
                                    "dummy_download_selected_button",
                                    "Download_CSV",
                                    disabled=True,
                                )

                            else:

                                @render.download(
                                    label="Download CSV",
                                    filename=(
                                        f"{APP_NAME}_selected_process_"
                                        f"{feed_type}_feed.csv"
                                    ),
                                )
                                def download_selected():
                                    yield sel_row.to_csv(
                                        index_label="param", header=["value"]
                                    )

    @reactive.effect
    @reactive.event(input.back_button)
    def back_to_inputs():
        # go back to the input parameters panel
        go_to_navbar_panel("Input parameters")


with ui.navset_bar(id=MAIN_NAVBAR_ID, title=None, navbar_options=NAVBAR_OPTIONS):
    with ui.nav_panel("Input parameters"):
        # top section with intro in left and buttons in right column
        with ui.layout_columns(col_widths=(6, 6)):
            with ui.div():
                with card_with_nav_header(title="Introduction"):
                    ui.p(
                        """
                        Please provide some basic information about your process
                        setup below. Parameters in the left column are shared by
                        both stages. Parameters in the right column are specific
                        to each stage. Any parameter not provided for the second
                        stage specifically will be assumed to be the same as in
                        the first stage.
                        """
                    )
                    ui.p(
                        f"""
                        Some of these parameters are determined by your
                        experimental setup (e.g. {params.feed["V_max"].label}).
                        Others can be found in the literature (e.g.
                        {params.rates["rho"].label} for your production
                        organism) and some need to be estimated from
                        experimental data (usually the specific productivities
                        {params.rates["pi_0"].label} and
                        {params.rates["pi_1"].label}). Tutorials on how to do
                        this can be found
                        """,
                        " ",
                        ui.tags.a(
                            "on github",
                            href=(
                                "https://github.com/julibeg/"
                                "FedBatchDesigner/tree/main/case-studies"
                            ),
                            target="_blank",
                        ),
                        ".",
                    )
                    ui.p(
                        "Please see also the ",
                        ui.input_action_link("info_link", "info panel"),
                        " for more details regarding the underlying assumptions etc.",
                    )

                    with ui.div(
                        style=(
                            "display: flex;"
                            "justify-content: space-evenly;"
                            "align-items: center;"
                            "width: 100%;"
                            "margin-top: 1em;"
                            "margin-bottom: 1em;"
                        )
                    ):
                        ui.input_action_button(
                            "submit", "Submit", class_="btn btn-primary"
                        )
                        ui.input_action_button(
                            "populate_defaults",
                            "Populate empty fields with defaults",
                            class_="btn btn-secondary",
                        )
                        ui.input_action_button("clear", "Clear", class_="btn btn-info")

                with card_with_nav_header(title="Feed parameters"):
                    ui.p(
                        f"""
                        {params.feed["V_max"].label} - {params.batch["V_batch"].label}
                        is the volume of medium added during the feed phase. Three feed
                        strategies are considered:
                        """
                    )
                    ui.tags.ul(
                        ui.tags.li(
                            """
                            Constant feed: The feed rate doesn't change throuout the
                            feed phase.
                            """
                        ),
                        ui.tags.li(
                            """
                            Linear feed: The feed rate increases linearly according
                            to \(F = F_0 + t \cdot dF\).
                            """
                        ),
                        ui.tags.li(
                            """
                            Exponential feed: The feed rate increases exponentially
                            according to \(F = F_0 \cdot e^{\mu t}\).
                            """
                        ),
                    )
                    ui.p(
                        """
                        For linear and exponential feed, the initial feed rate \(F_0\)
                        is chosen such that the total amount of biomass also increases
                        linearly or exponentially (taking substrate requirements for
                        product formation and maintenance into account).
                        """
                    )
                    ui.p(
                        f"""
                        For optimizing the exponential feed, specific growth rates up to
                        {params.feed["mu_max_feed"].label} are considered. The
                        parameters {params.stage_specific["mu_max_phys"].label} and
                        {params.feed["F_max"].label} are used to ensure that neither the
                        maximum physiological growth rate of the organism nor the
                        maximum feed rate of the reactor are exceeded by any feed
                        strategy.
                        """,
                    )
                    with ui.layout_column_wrap(width=1 / 2):
                        for k, v in params.feed.items():
                            with ui.div():
                                ui.input_text(id=k, label=str(v))
                                ui.p(v.description)

            with ui.div():
                with card_with_nav_header(title="Batch parameters"):
                    ui.p(
                        "The batch phase is assumed to have already "
                        "been passed and won't be optimized."
                    )
                    with ui.layout_column_wrap(width=1 / 2):
                        for k, v in params.batch.items():
                            with ui.div():
                                ui.input_text(id=k, label=str(v))
                                ui.p(v.description)

                # stage-specific parameters
                with ui.card():
                    with ui.navset_bar(
                        title="Stage-specific parameters", navbar_options=NAVBAR_OPTIONS
                    ):
                        for stage_idx in [1, 2]:
                            with ui.nav_panel(f"Stage {stage_idx}"):
                                with ui.layout_columns(col_widths=(8, 4)):
                                    with ui.div():
                                        ui.tags.p(
                                            """
                                            These parameters (yield coefficients and
                                            specific ATP consumption and product
                                            formation rates) can change between the two
                                            stages and make up the specific substrate
                                            consumption rate according to
                                            """
                                        )
                                        ui.tags.p(
                                            r""" \(
                                            \sigma = \frac{\mu}{Y_{X/S}} + \frac{\pi_0 +
                                            \mu \pi_1}{Y_{P/S}} + \frac{\rho}{Y_{ATP/S}}
                                            \) """,
                                            style=(
                                                "text-align: center; font-size: 130%;"
                                            ),
                                        )
                                    # with ui.div(style="text-align: center;"):
                                    with ui.div():
                                        with ui.card():
                                            # put `mu_max_phys` on its own and group the
                                            # yields and rates
                                            k = "mu_max_phys"
                                            v = params.stage_specific[k]
                                            with ui.div():
                                                ui.input_text(
                                                    id=f"s{stage_idx}_{k}",
                                                    label=str(v),
                                                )
                                                ui.tags.p(v.description)
                                with ui.card():
                                    ui.card_header("Yield coefficients")
                                    with ui.layout_column_wrap(width=1 / 3):
                                        for k, v in params.yields.items():
                                            ui.input_text(
                                                id=f"s{stage_idx}_{k}", label=str(v)
                                            )
                                    ui.tags.p(
                                        """
                                        Yield coefficients of biomass, product, and ATP
                                        (in grams per gram substrate consumed).
                                        """
                                    )
                                with ui.card():
                                    ui.card_header("Specific rates")
                                    with ui.layout_column_wrap(width=1 / 3):
                                        for k, v in params.rates.items():
                                            with ui.div():
                                                ui.input_text(
                                                    id=f"s{stage_idx}_{k}", label=str(v)
                                                )
                                                ui.tags.p(v.description)

    for stage_1_type in STAGE_1_TYPES:
        with ui.nav_panel(f"Results {stage_1_type.feed_type} feed"):
            results(f"{stage_1_type.feed_type}_results", stage_1_type)

    with ui.nav_panel("Further information"):
        info.info()


@reactive.Effect
@reactive.event(input.info_link)
def jump_to_info_panel():
    go_to_navbar_panel("Further information")


@reactive.Effect
@reactive.event(input.populate_defaults)
def populate_defaults_button():
    """Show modal with default set options"""
    modal = ui.modal(
        ui.p(
            """
            Select a set of default parameters to populate empty fields with. The
            available options are (for details see below):
            """
        ),
        ui.div(
            ui.input_radio_buttons(
                "selected_defaults",
                None,
                choices={k: ui.HTML(v["title"]) for k, v in params.defaults.items()},
                selected="",
                inline=True,
                width="100%",
            ),
            ui.tags.style(
                # add style to center the radio buttons
                """
                #selected_defaults .shiny-options-group {
                    justify-content: center;
                    gap: 10px;
                }
                """
            ),
        ),
        ui.div(
            ui.input_action_button(
                "apply_default",
                "Apply selected defaults",
                disabled=True,
                class_="btn btn-secondary",
            ),
            style="display: flex; justify-content: center; align-items: center;",
        ),
        ui.tags.ul(
            *[
                ui.tags.li(ui.HTML(f"<b>{v['title']}</b>:<br>{v['description']}"))
                for v in params.defaults.values()
            ]
        ),
        title="Select Default Parameter Set",
        easy_close=True,
        footer=None,
        size="l",
    )
    ui.modal_show(modal)


@reactive.Effect
@reactive.event(input.selected_defaults)
def make_apply_defaults_button_clickable():
    ui.update_action_button("apply_default", disabled=False)


@reactive.Effect
@reactive.event(input.apply_default)
def apply_selected_defaults():
    """Apply the selected default parameter set"""
    try:
        selected = params.defaults[input.selected_defaults()]
        default_set_name = selected["title"]
        default_values = selected["values"]
        for k, v in default_values.items():
            # only use default if field is empty
            if not input[k]():
                ui.update_text(k, value=round(v, 3))
        ui.notification_show(
            ui.HTML(f"Applied defaults from '{default_set_name}'"),
            type="message",
            duration=3,
        )
    except KeyError:
        ui.notification_show(
            ui.HTML(f"Error: Could not load defaults for '{default_set_name}'"),
            type="error",
            duration=3,
        )
    finally:
        ui.modal_remove()


@reactive.Effect
@reactive.event(input.clear)
def clear_all_inputs():
    """Clear all input fields"""
    for k in params.common.keys():
        ui.update_text(k, value="")

    for k in params.stage_specific.keys():
        ui.update_text(f"s1_{k}", value="")
        ui.update_text(f"s2_{k}", value="")

    ui.notification_show("All fields cleared", type="message", duration=3)


def validate_param(value, required):
    """Make sure the input values are numeric and non-negative."""
    if required and (value is None or value == ""):
        return "Required"
    if not value:
        return
    try:
        value = float(value)
    except ValueError:
        return "Needs to be numeric"
    if value < 0:
        return "Needs to be non-negative"


def parse_params():
    """Parse the params input by the user."""
    if not input_validator.is_valid():
        # return early if inputs not valid
        return
    parsed = {}
    parsed["common"] = {
        k: float(input[k]()) for k, v in params.common.items() if v.required
    }
    parsed["s1"] = {k: float(input[f"s1_{k}"]()) for k in params.stage_specific.keys()}
    # use params from stage 1 in stage two unless specified otherwise
    parsed["s2"] = parsed["s1"].copy()
    for k in params.stage_specific.keys():
        value = input[f"s2_{k}"]()
        if value:
            parsed["s2"][k] = float(value)
    PARSED_PARAMS.set(parsed)


# input validation rules; validate each param to be numeric etc
input_validator = InputValidator()
for k, v in params.common.items():
    input_validator.add_rule(k, functools.partial(validate_param, required=v.required))
for k in params.stage_specific.keys():
    # remember that the IDs of the stage-specific params are prefixed with `s1_` and
    # `s2_`
    input_validator.add_rule(
        f"s1_{k}", functools.partial(validate_param, required=True)
    )
for k in params.stage_specific.keys():
    input_validator.add_rule(
        f"s2_{k}", functools.partial(validate_param, required=False)
    )


def validate_V_batch(value):
    # we still need to call `validate_param()` since `InputValidator` only keeps track
    # of the last validation rule (i.e. `validate_param` added as rule above is
    # overwritten)
    msg = validate_param(value, True)
    if msg is not None:
        return msg
    try:
        # only compare with `V_max` if it has already been provided
        V_max = float(input["V_max"]())
    except ValueError:
        return
    if float(value) >= V_max:
        return "'V_batch' must be smaller than 'V_max'"


def validate_V_max(value):
    # we still need to call `validate_param()` since `InputValidator` only keeps track
    # of the last validation rule (i.e. `validate_param` added as rule above is
    # overwritten)
    msg = validate_param(value, True)
    if msg is not None:
        return msg
    try:
        # only compare with `V_batch` if it has already been provided
        V_batch = float(input["V_batch"]())
    except ValueError:
        return
    if float(value) <= V_batch:
        return "'V_max' must be larger than 'V_batch'"


def validate_mu_max_feed(value):
    # we still need to call `validate_param()` since `InputValidator` only keeps track
    # of the last validation rule (i.e. `validate_param` added as rule above is
    # overwritten)
    msg = validate_param(value, True)
    if msg is not None:
        return msg
    try:
        # only compare with `s1_mu_max_phys` if it has already been provided
        mu_max_phys = float(input["s1_mu_max_phys"]())
    except ValueError:
        return
    if float(value) > mu_max_phys:
        return "cannot be larger than the physiological µ_max of the organism"


def validate_mu_max_phys(value):
    # we still need to call `validate_param()` since `InputValidator` only keeps track
    # of the last validation rule (i.e. `validate_param` added as rule above is
    # overwritten)
    msg = validate_param(value, True)
    if msg is not None:
        return msg
    try:
        # only compare with `mu_max_feed` if it has already been provided
        mu_max_feed = float(input["mu_max_feed"]())
    except ValueError:
        return
    if float(value) <= mu_max_feed:
        return "must be at least as big as the maximum growth rate of the feed"


input_validator.add_rule("V_batch", validate_V_batch)
input_validator.add_rule("V_max", validate_V_max)
input_validator.add_rule("mu_max_feed", validate_mu_max_feed)
input_validator.add_rule("s1_mu_max_phys", validate_mu_max_phys)
input_validator.add_rule("s2_mu_max_phys", validate_mu_max_phys)


@reactive.effect
def _():
    input_validator.enable()
