import functools
import time

import numpy as np
import pandas as pd
from shiny import reactive
from shiny.express import input, module, render, ui
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
    NoGrowthConstantStage as NoGrowthS2,
)

# some global variables
APP_NAME = "FedBatchDesigner"
N_MINIMUM_LEVELS_FOR_MU_OR_F = 15
V_FRAC_STEP = 0.02
ROUND_DIGITS = 5
N_CONTOURS = 30

# reactive values
PARSED_PARAMS = reactive.value(None)
DF_GRID_SEARCH_MU = reactive.value(None)
DF_GRID_SEARCH_F = reactive.value(None)

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
)


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

    mu_or_F = "mu" if isinstance(stage_1, ExpS1) else "F"

    if mu_or_F == "mu" and (mu_min := input_params["common"]["mu_min"]) is None:
        # handle `mu_min` separately as per default we use `mu_max / 20` if not required
        # (and in that case we'll just get a linspace from `mu_min` to `mu_max` with 20
        # values)
        mu_max = input_params["common"]["mu_max"]
        mus_or_Fs = np.linspace(mu_max / 20, mu_max, 20).round(ROUND_DIGITS)
    else:
        # we either got constant feed or a user-provided `mu_min`
        min_mu_or_F = stage_1.F_min if mu_or_F == "F" else mu_min
        # get range of `mu` (or `F`) and `V_frac` values (and round to avoid floating
        # point issues)
        mus_or_Fs = util.get_first_nice_value_range_with_at_least_N_values(
            min_val=min_mu_or_F,
            max_val=input_params["common"][f"{mu_or_F}_max"],
            min_n_values=N_MINIMUM_LEVELS_FOR_MU_OR_F,
        ).round(ROUND_DIGITS)
    if mus_or_Fs[-1] < input_params["common"][f"{mu_or_F}_max"]:
        mus_or_Fs = np.append(mus_or_Fs, input_params["common"][f"{mu_or_F}_max"])
    V_frac = np.arange(0, 1 + V_FRAC_STEP, V_FRAC_STEP).round(ROUND_DIGITS)
    V_intervals = (
        input_params["common"]["V_batch"]
        + (input_params["common"]["V_max"] - input_params["common"]["V_batch"]) * V_frac
    )

    # initialise empty results df
    df_comb = pd.DataFrame(
        np.nan,
        index=pd.MultiIndex.from_product(
            (mus_or_Fs, V_frac), names=[mu_or_F, "V_frac"]
        ),
        columns=["V1", "X1", "P1", "t_switch", "V2", "X2", "P2", "F2", "t_end"],
    )

    # for each `F`, get the points in the exponential feed at all `V_frac`; then "fill
    # up" until `V_max` with a constant feed rate and calculate productivity based on
    # the end time
    for m_or_F in mus_or_Fs:
        df_s1 = stage_1.evaluate_at_V(Vs=V_intervals, **{mu_or_F: m_or_F})
        df_s1["V_frac"] = V_frac
        for t_switch, row_s1 in df_s1.iterrows():
            stage_2 = NoGrowthS2(
                *row_s1[["V", "X", "P"]],
                s_f=input_params["common"]["s_f"],
                **input_params["s2"],
            )
            row_s2 = stage_2.evaluate_at_V(input_params["common"]["V_max"]).squeeze()
            # get constant feed rate in stage 2 and the end time
            F2 = stage_2.F
            t_end = row_s2.name + t_switch
            df_comb.loc[(m_or_F, row_s1["V_frac"])] = [
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
    # of feed
    if mu_or_F == "mu":
        # exponential feed --> add substrate start volume and initial feed rate for each
        # `mu` to the results
        for mu, _ in df_comb.groupby("mu"):
            ssv = stage_1.substrate_start_volume(mu)
            df_comb.loc[(mu, slice(None)), "substrate_start_volume"] = ssv
            df_comb.loc[(mu, slice(None)), "F0"] = ssv / mu
    else:
        # constant feed --> add maximum growth rate (at first instance of feed)
        for F, _ in df_comb.groupby("F"):
            df_comb.loc[(F, slice(None)), "mu_max"] = stage_1.calculate_initial_mu(F)
    return df_comb


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
        X_batch = parsed_params["common"]["x_batch"] * parsed_params["common"]["V_batch"]
        const_s1 = ConstS1(
            V0=parsed_params["common"]["V_batch"],
            X0=X_batch,
            P0=0,
            s_f=parsed_params["common"]["s_f"],
            **parsed_params["s1"],
        )
        exp_s1 = ExpS1(
            V0=parsed_params["common"]["V_batch"],
            X0=X_batch,
            P0=0,
            s_f=parsed_params["common"]["s_f"],
            **parsed_params["s1"],
        )
        # (for now we only check if `mu_max` or `F_max` are larger than the minimum feed
        # rate required to add enough medium in the first instant of the feed phase, but we
        # could add more checks in the future)
        if const_s1.F_min > parsed_params["common"]["F_max"]:
            # parameters are infeasible; `F_max` needs to be increased
            ui.notification_show(
                """
                Submitted parameters are infeasible. The minimum F value required in order
                to add enough feed medium in the first instant of the feed phase is larger
                than the maximum F value provided. Increase 'F_max' or adjust other parameters
                (e.g. X_batch, s_f, maintenance requirement, etc).
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

        df_grid_mu = run_grid_search(exp_s1, parsed_params)
        DF_GRID_SEARCH_MU.set(df_grid_mu)
        df_grid_F = run_grid_search(const_s1, parsed_params)
        DF_GRID_SEARCH_F.set(df_grid_F)

        # calculations are done; remove the modal and jump to "Results" (we also need to
        # remove the "no results yet" message on the Results panels)
        ui.remove_ui(".no_results_yet", multiple=True)
        ui.modal_remove()
        ui.update_navs(id="navbar", selected="Results constant feed")

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
def results(input, output, session, exp_or_const):

    mu_or_F = {"exponential": "mu", "constant": "F"}[exp_or_const]

    @render.express
    def _results():
        linked_plots = []
        parsed_params = PARSED_PARAMS.get()
        df_comb = (DF_GRID_SEARCH_MU if mu_or_F == "mu" else DF_GRID_SEARCH_F).get()
        selected_process = reactive.Value({})
        # check if the grid search has been run; if not (i.e. we don't have results
        # yet), show a message and return
        if df_comb is None:
            return

        ui.tags.p(
            f"""
            Concentration profiles were calculated for different values of {mu_or_F}
            and the {params.results["V_frac"].description.lower()}.
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
                            df=df_comb,
                            mu_or_F=mu_or_F,
                            x_col="V_frac",
                            y_col=mu_or_F,
                            z_col="space_time_yield",
                            text_col="p2",
                            n_contours=N_CONTOURS,
                        ).fig

                    @render_plotly
                    def yield_contour_plot():

                        return plots.ContourPlot(
                            linked_plots=linked_plots,
                            selected_process=selected_process,
                            df=df_comb,
                            mu_or_F=mu_or_F,
                            x_col="V_frac",
                            y_col=mu_or_F,
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
                            df=df_comb,
                            mu_or_F=mu_or_F,
                            x_col="p2",
                            y_col="space_time_yield",
                        ).fig

                    @render_plotly
                    def selected_process_plot():

                        return plots.SelectedProcessPlot(
                            linked_plots=linked_plots,
                            selected_process=selected_process,
                            df=df_comb,
                            mu_or_F=mu_or_F,
                            params=parsed_params,
                            stage_1_class=ConstS1 if mu_or_F == "F" else ExpS1,
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
                            f"{APP_NAME}_grid_search_results_{exp_or_const}_feed.csv"
                        ),
                    )
                    def download_grid_search_results():
                        yield df_comb.to_csv()

                params_for_tables = (
                    ["substrate_start_volume", "F0"]
                    if exp_or_const == "exponential"
                    else ["mu_max"]
                ) + [
                    mu_or_F,
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
                            df_comb, df_comb["productivity"].idxmax()
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
                                for p in params_for_tables:
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
                                    f"{APP_NAME}_optimal_process_"
                                    f"{exp_or_const}_feed.csv"
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
                                    df_comb, (sel_params[mu_or_F], sel_params["V_frac"])
                                )

                            with ui.p():
                                for p in params_for_tables:
                                    with ui.tooltip():
                                        ui.tags.b(f"{params.results[p].label}:"),
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
                                        f"{exp_or_const}_feed.csv"
                                    ),
                                )
                                def download_selected():
                                    yield sel_row.to_csv(
                                        index_label="param", header=["value"]
                                    )


with ui.navset_bar(id="navbar", title=None):
    with ui.nav_panel("Input parameters"):
        # top section with intro in left and buttons in right column
        with ui.layout_columns(col_widths=(6, 6)):
            with ui.div():
                with ui.card():
                    ui.card_header("Introduction")
                    ui.p(
                        """
                        Please provide some basic information about your process setup
                        below. Parameters in the left column are shared by both stages.
                        Parameters in the right column are specific to each stage. Any
                        parameter not provided for the second stage specifically will be
                        assumed to be the same as in the first stage.
                        """
                    )
                    ui.p(
                        f"""
                        Some of these parameters are determined by your experimental setup
                        (e.g. {params.feed["V_max"].label}). Others can be found in the
                        literature (e.g. {params.stage_specific["rho"].label} for your
                        production organism) and some need to be estimated from experimental
                        data (usually the specific productivities
                        {params.stage_specific["pi_0"].label} and
                        {params.stage_specific["pi_1"].label}). Tutorials on how to do this
                        can be found
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
                        ui.input_action_link("info_link", "Info panel"),
                        " for more details.",
                    )

                # batch + feed parameters
                with ui.card():
                    ui.card_header("Batch parameters")
                    ui.p(
                        "The batch phase is assumed to have already "
                        "been passed and won't be optimized."
                    )
                    with ui.layout_column_wrap(width=1 / 2):
                        for k, v in params.batch.items():
                            with ui.div():
                                ui.input_text(id=k, label=str(v))
                                ui.p(v.description)

                with ui.card():
                    ui.card_header("Feed parameters")
                    ui.p(
                        f"""
                        {params.feed['V_max'].label} -
                        {params.batch['V_batch'].label} is the volume of medium
                        added during the feed phase. Specific growth rates up to
                        {params.feed['mu_max'].label} are considered when optimizing
                        the the exponential feed and feed rates up to
                        {params.feed['F_max'].label} are considered when optimizing
                        the constant feed.
                        """
                    )
                    with ui.layout_column_wrap(width=1 / 2):
                        for k, v in params.feed.items():
                            with ui.div():
                                ui.input_text(id=k, label=str(v))
                                ui.p(v.description)

            with ui.div():
                with ui.div(
                    style=(
                        "display: flex;"
                        "justify-content: space-evenly;"
                        "align-items: center;"
                        "width: 100%;"
                        "margin-top: 2em;"
                        "margin-bottom: 2em;"
                    )
                ):
                    ui.input_action_button("submit", "Submit", class_="btn btn-primary")
                    ui.input_action_button(
                        "populate_defaults",
                        "Populate empty fields with defaults",
                        class_="btn btn-secondary",
                    )
                    ui.input_action_button("clear", "Clear", class_="btn btn-info")

                # stage-specific parameters
                with ui.card():
                    ui.card_header("Stage-specific parameters")
                    with ui.navset_bar(title=None):
                        with ui.nav_panel("Stage 1"):
                            with ui.layout_column_wrap(width=1 / 2):
                                for k, v in params.stage_specific.items():
                                    with ui.div():
                                        ui.input_text(id=f"s1_{k}", label=str(v))
                                        ui.p(v.description)
                        with ui.nav_panel("Stage 2 (optional)"):
                            with ui.layout_column_wrap(width=1 / 2):
                                for k, v in params.stage_specific.items():
                                    with ui.div():
                                        ui.input_text(id=f"s2_{k}", label=str(v))
                                        ui.p(v.description)

    for exp_or_const in ["constant", "exponential"]:
        with ui.nav_panel(f"Results {exp_or_const} feed"):
            # add message saying that there are no results yet and a button to go back
            # to the Inputs panel (this will be removed once the grid search is run)
            with ui.div(class_="no_results_yet"):
                with ui.card():
                    ui.h4("No results yet", style="text-align: center;")
                    with ui.div(style="justify-content: center; display: flex"):
                        ui.input_action_button(
                            f"back_button_{exp_or_const}",
                            "Back to inputs",
                            class_="btn btn-primary",
                        )

            # now render the results (this returns early without rendering anything if
            # the grid search hasn't been run yet)
            results(f"{exp_or_const}_results", exp_or_const=exp_or_const)

    with ui.nav_panel("Info"):
        info.info()


@reactive.Effect
@reactive.event(input.info_link)
def jump_to_info_panel():
    ui.update_navs(id="navbar", selected="Info")


@reactive.Effect
@reactive.event(input.back_button_constant, input.back_button_exponential)
def back_to_inputs():
    """Jump back to the "Input parameters" panel."""
    ui.update_navs(id="navbar", selected="Input parameters")


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
                width="100%",  # Add this to ensure full width
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
    # handle `mu_min` separately as it's not required
    if not (mu_min := input["mu_min"]()):
        parsed["common"]["mu_min"] = None
    else:
        parsed["common"]["mu_min"] = float(mu_min)
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


def validate_mu_max(value):
    # we still need to call `validate_param()` since `InputValidator` only keeps track
    # of the last validation rule (i.e. `validate_param` added as rule above is
    # overwritten)
    msg = validate_param(value, True)
    if msg is not None:
        return msg
    try:
        # only compare with `mu_max` if it has already been provided
        mu_min = float(input["mu_min"]())
    except ValueError:
        return
    if float(value) <= mu_min:
        return "'mu_max' must be larger than 'mu_min'"


def validate_mu_min(value):
    # `mu_min` isn't required -> return if hasn't been provided
    if not value:
        return
    # we still need to call `validate_param()` since `InputValidator` only keeps track
    # of the last validation rule (i.e. `validate_param` added as rule above is
    # overwritten)
    msg = validate_param(value, False)
    if msg is not None:
        return msg
    try:
        # only compare with `mu_max` if it has already been provided
        mu_max = float(input["mu_max"]())
    except ValueError:
        return
    if float(value) >= mu_max:
        return "'mu_min' must be smaller than 'mu_max'"


input_validator.add_rule("V_batch", validate_V_batch)
input_validator.add_rule("V_max", validate_V_max)
input_validator.add_rule("mu_min", validate_mu_min)
input_validator.add_rule("mu_max", validate_mu_max)


@reactive.effect
def _():
    input_validator.enable()
