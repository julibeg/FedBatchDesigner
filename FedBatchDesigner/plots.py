from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import colors
from params import results

HIGHLIGHT_LINE_KWARGS = dict(
    line_dash="dash",
    line_width=1.5,
    opacity=0.8,
)


def _get_max_prod_params(df):
    mu_or_F, V_frac = df.loc[df["space_time_yield"].idxmax()].name
    return mu_or_F, V_frac


class Plot(ABC):

    linked_plots = None
    _fig = None

    def __init__(self, df, mu_or_F, linked_plots=None, selected_process=None):
        self.df = df
        self.mu_or_F = mu_or_F
        if linked_plots is not None:
            self.linked_plots = linked_plots
            self.linked_plots.append(self)
        self.selected_process = selected_process
        self.fig.update_layout(modebar={"orientation": "v"})

    def _update_plots(self, **kwargs):
        """Update all plots by calling their `.update()` method."""
        # check kwargs are valid
        if unexpected_kwargs := (
            set(kwargs.keys()) - {self.mu_or_F, "V_frac", "space_time_yield", "p2"}
        ):
            raise ValueError(f"Invalid kwargs: {unexpected_kwargs}")
        self.selected_process.set(kwargs)
        for plot in self.linked_plots:
            plot.update(**kwargs)

    @property
    def fig(self):
        if self._fig is None:
            self._create_fig()
        return self._fig

    @abstractmethod
    def _create_fig(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError


class PlotWithMarkers(Plot):
    """Parent class for plots with markers for the optimal and selected process."""

    has_selected_process = False

    def __init__(self, *, x_col, y_col, **kwargs):
        self.x_col = x_col
        self.y_col = y_col
        super().__init__(**kwargs)

    def add_marker_for_optimal_process(self, x, y):
        # add optimal marker (immobile) and selected marker (updateable)
        self.add_highlight_marker(
            x,
            y,
            color=colors.optimal_marker_color,
            marker_symbol="star",
            marker_size=15,
        )

    def add_highlight_marker(
        self,
        x,
        y,
        color,
        marker_symbol="circle",
        marker_size=10,
        updateable=False,
    ):
        # use `add_scattergl` instead of `add_scatter` so that the markers are on top of
        # the lines added with `px.line` (https://stackoverflow.com/a/77145622/8106901)
        self._fig.add_scattergl(
            x=[x],
            y=[y],
            mode="markers",
            marker_symbol=marker_symbol,
            marker_size=marker_size,
            marker_color=color,
            showlegend=False,
        )
        # add dashed vertical and horizontal lines to highlight the selected point
        self._fig.add_vline(
            x=x,
            line_color=color,
            **HIGHLIGHT_LINE_KWARGS,
        )
        self._fig.add_hline(
            y=y,
            line_color=color,
            **HIGHLIGHT_LINE_KWARGS,
        )
        if updateable:
            # define a function to update the scatter trace containing the single point
            # and the lines (i.e. simply the most recently added trace and shapes)
            marker_selector = len(self.fig.data) - 1
            vline_selector = len(self.fig.layout.shapes) - 2
            hline_selector = len(self.fig.layout.shapes) - 1

            def _update_marker(x, y):
                self.fig.update_traces(selector=marker_selector, x=[x], y=[y])
                # update the vertical and horizontal lines
                self.fig.update_shapes(selector=vline_selector, x0=x, x1=x)
                self.fig.update_shapes(selector=hline_selector, y0=y, y1=y)

            self.update_marker = _update_marker

    def update(self, **kwargs):
        x, y = kwargs[self.x_col], kwargs[self.y_col]
        if not self.has_selected_process:
            self.add_highlight_marker(
                x,
                y,
                color=colors.selected_marker_color,
                updateable=True,
                marker_symbol="circle",
                marker_size=11,
            )
            self.has_selected_process = True

        self.update_marker(x, y)


class ContourPlot(PlotWithMarkers):
    """
    Contour plot with with `V_frac` and `mu` or `F` as axes and any column in the input
    `DataFrame` as z variable.
    """

    def __init__(self, *, z_col, text_col, n_contours=50, **kwargs):
        self.z_col = z_col
        self.text_col = text_col
        self.n_contours = n_contours
        super().__init__(**kwargs)

    def _create_fig(self):
        mu_or_F_opt, V_frac_opt = _get_max_prod_params(self.df)

        pivoted = self.df.reset_index().pivot(
            index=self.y_col, columns=self.x_col, values=[self.z_col, self.text_col]
        )

        z_df = pivoted[self.z_col]
        text_df = pivoted[self.text_col]

        # create hovertemplate string
        hovertemplate = []
        for col, fmt in zip(
            [self.x_col, self.y_col, self.z_col, self.text_col],
            ["x", "y", "z:.4g", "text:.4g"],
        ):
            symbol = results[col].label
            unit = results[col].unit
            hovertemplate.append(f"{symbol}=%{{{fmt}}} {unit}")
        hovertemplate = "<br>".join(hovertemplate)

        # start with contour
        self._fig = go.Figure(
            data=go.Contour(
                x=z_df.columns,
                y=z_df.index,
                z=z_df.values,
                colorscale=colors.scale,
                name="",
                meta=self.z_col,
                hovertemplate=hovertemplate,
                text=text_df.values,
                ncontours=self.n_contours,
            )
        )

        self._fig.update_layout(
            margin=dict(l=20, r=20, b=20),
            xaxis_range=[z_df.columns.min(), z_df.columns.max()],
            yaxis_range=[z_df.index.min(), z_df.index.max()],
            title=results[self.z_col].label,
            xaxis_title=str(results[self.x_col]),
            yaxis_title=str(results[self.y_col]),
        )

        # turn into a widget so that it's reactive
        self._fig = go.FigureWidget(self._fig)

        # add a point and horizontal + vertical lines to mark the "optimal" as well as
        # the "selected" process (i.e. with a specific `mu` and `V_frac`)
        self.add_marker_for_optimal_process(V_frac_opt, mu_or_F_opt)

        # define a callback when clicked on the plot to select a point with `mu` and
        # `V_frac`
        def click(trace, points, _selector):
            (x_clicked,) = points.xs
            (y_clicked,) = points.ys

            (idx,) = points.point_inds
            z_clicked = trace.z[*idx]
            text_clicked = trace.text[*idx]
            # update all plots with the selected values
            self._update_plots(
                **{
                    self.x_col: x_clicked,
                    self.y_col: y_clicked,
                    self.z_col: z_clicked,
                    self.text_col: text_clicked,
                },
            )

        # add the click callback to the contour trace
        contour = self._fig.data[0]
        contour.on_click(click)


class LinePlot(PlotWithMarkers):
    """
    For each `mu` or `F`, plot the traces of `x` vs `y` from `V_frac=0` to `V_frac=1`.
    """

    def _create_fig(self):
        mu_or_F_opt, V_frac_opt = _get_max_prod_params(self.df)

        # colour scale to colour the lines based on the value of `mu`
        mus_or_Fs = self.df.reset_index()[self.mu_or_F].unique()
        mus_or_Fs_scaled = (mus_or_Fs - mus_or_Fs.min()) / (
            mus_or_Fs.max() - mus_or_Fs.min()
        )
        myscale = px.colors.sample_colorscale(
            colorscale=colors.scale,
            samplepoints=mus_or_Fs_scaled,
            colortype="rgb",
        )

        self._fig = px.line(
            self.df.reset_index(),
            x=self.x_col,
            y=self.y_col,
            color=self.mu_or_F,
            color_discrete_sequence=myscale,
            custom_data=[self.mu_or_F, "V_frac"],
            template="simple_white",
        )

        # create hovertemplate string
        hovertemplate = []
        for col, fmt in zip(
            [self.x_col, self.y_col, self.mu_or_F, "V_frac"],
            ["x:.4g", "y:.4g", "customdata[0]", "customdata[1]"],
        ):
            symbol = results[col].label
            unit = results[col].unit
            hovertemplate.append(f"{symbol}=%{{{fmt}}} {unit}")
        hovertemplate = "<br>".join(hovertemplate)

        self._fig.update_traces(hovertemplate=(hovertemplate))

        # convert to `FigureWidget` for interactivity
        self._fig = go.FigureWidget(self._fig)

        # get values of optimal point (which is always the first point we highlight) and
        # add the marker + hline / vline
        x_opt, y_opt = self.df.loc[(mu_or_F_opt, V_frac_opt), [self.x_col, self.y_col]]

        # add a point and horizontal + vertical lines to mark the "optimal" as well as
        # the "selected" process (i.e. with a specific `mu` or `F` and `V_frac`)
        self.add_marker_for_optimal_process(x_opt, y_opt)

        self._fig.update_layout(
            margin=dict(l=20, r=20, b=20),
            title=(
                results[self.y_col].label + " vs. " + results[self.x_col].label.lower()
            ),
            xaxis_title=str(results[self.x_col]),
            yaxis_title=str(results[self.y_col]),
            legend_title=str(results[self.mu_or_F]),
        )

        # create click callback to select a different point / `mu` + `V_frac`
        def click(trace, points, _selector):
            # when multiple traces in a plot have callbacks, each callback will be
            # called on click (i.e. also for the other traces and not just the one that
            # contains the clicked point); in the other traces `points` is empty and we
            # don't do anything
            if len(points.point_inds) == 0:
                return

            (idx,) = points.point_inds
            (clicked_x,) = points.xs
            (clicked_y,) = points.ys
            mu_or_F, V_frac = trace.customdata[idx]

            self._update_plots(
                **{
                    self.mu_or_F: mu_or_F,
                    "V_frac": V_frac,
                    self.x_col: clicked_x,
                    self.y_col: clicked_y,
                },
            )

        # add click callback to each line trace
        for line in self._fig.data:
            line.on_click(click)


class SelectedProcessPlot(Plot):
    """Simple line chart showing V, X, P vs. process time."""

    def __init__(self, *, params, stage_1_class, stage_2_class, **kwargs):
        self.params = params
        self.stage_1_class = stage_1_class
        self.stage_2_class = stage_2_class
        super().__init__(**kwargs)

    def get_process_trajectory(self, mu_or_F, V_frac):
        # get V, X, P vs t for the selected `mu` and `V_frac`
        row = self.df.loc[(mu_or_F, V_frac)]
        t1 = np.linspace(0, row["t_switch"], 100)
        t2 = np.linspace(0, row["t_end"] - row["t_switch"], 100)

        stage_1 = self.stage_1_class(
            V0=self.params["base"]["V_batch"],
            X0=self.params["base"]["V_batch"] * self.params["base"]["x_batch"],
            P0=0,
            **self.params["s1"],
        )
        stage_2 = self.stage_2_class(*row[["V1", "X1", "P1"]], **self.params["s2"])

        df_s1 = stage_1.evaluate_at_t(t1, **{self.mu_or_F: mu_or_F})
        df_s2 = stage_2.evaluate_at_t(t2)
        df_s2.index += row["t_switch"]
        # combine the two stages and columns for the biomass and product concentrations
        df_comb = pd.concat([df_s1[["V", "X", "P"]], df_s2])
        df_comb["x"] = df_comb["X"] / df_comb["V"]
        df_comb["p"] = df_comb["P"] / df_comb["V"]
        return df_comb

    def _create_fig(self):
        # get process trajectory for the optimal `mu` (or `F`) and `V_frac`
        mu_or_F_opt, V_frac_opt = _get_max_prod_params(self.df)
        row_opt = self.df.loc[(mu_or_F_opt, V_frac_opt)]
        df = self.get_process_trajectory(mu_or_F_opt, V_frac_opt)

        # TODO: can't we initialise like this for all plots?
        fig = go.FigureWidget()

        time_param = results[df.index.name]

        # plot V
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["V"],
                name="",
                line=dict(color="black"),
                yaxis="y1",
                hovertemplate=(
                    f"{time_param.label}=%{{x:.2f}} {time_param.unit}<br>"
                    f"{results['V'].label}=%{{y:.2f}} {results['V'].unit}"
                ),
            )
        )

        # plot X (absolute value)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["X"],
                name="",
                line=dict(color=colors.red),
                yaxis="y2",
                hovertemplate=(
                    f"{time_param.label}=%{{x:.2f}} {time_param.unit}<br>"
                    f"{results['X'].label}=%{{y:.2f}} {results['X'].unit}"
                ),
            )
        )

        # plot P (absolute value)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["P"],
                name="",
                line=dict(color=colors.blue),
                yaxis="y3",
                hovertemplate=(
                    f"{time_param.label}=%{{x:.2f}} {time_param.unit}<br>"
                    f"{results['P'].label}=%{{y:.2f}} {results['P'].unit}"
                ),
            )
        )

        # plot x (concentration)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["x"],
                name="",
                line=dict(color=colors.red),
                yaxis="y2",
                hovertemplate=(
                    f"{time_param.label}=%{{x:.2f}} {time_param.unit}<br>"
                    f"{results['x'].label}=%{{y:.2f}} {results['x'].unit}"
                ),
                visible=False,
            )
        )

        # plot p (concentration)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["p"],
                name="",
                line=dict(color=colors.blue),
                yaxis="y3",
                hovertemplate=(
                    f"{time_param.label}=%{{x:.2f}} {time_param.unit}<br>"
                    f"{results['p'].label}=%{{y:.2f}} {results['p'].unit}"
                ),
                visible=False,
            )
        )

        # vertical line at t_switch
        fig.add_vline(
            x=row_opt["t_switch"],
            line_dash="dash",
            line_color="black",
            # `line_width` and `opacity` needs to be set explicitly with the
            # "simple_white" theme (https://stackoverflow.com/a/67997733/8106901)
            line_width=1.5,
            opacity=0.8,
        )

        fig.update_layout(
            margin=dict(l=20, r=20, b=20),
            title="Process trajectory",
            xaxis_title=str(results["t"]),
            template="simple_white",
            xaxis=dict(
                domain=[0, 0.87],
            ),
            yaxis=dict(
                title=str(results["V"]),
                titlefont=dict(color="black"),
                tickfont=dict(color="black"),
            ),
            yaxis2=dict(
                title=str(results["X"]),
                titlefont=dict(color=colors.red),
                tickfont=dict(color=colors.red),
                overlaying="y",
                position=1,
            ),
            # TODO: instead of having an extra y-axis, we could plot X and P on the same
            # axis but with a factor for P to make it better visible (and add the factor
            # to the label / legend)
            yaxis3=dict(
                title=str(results["P"]),
                titlefont=dict(color=colors.blue),
                tickfont=dict(color=colors.blue),
                overlaying="y",
                side="right",
                position=1,
            ),
            showlegend=False,
            updatemenus=[
                dict(
                    direction="left",
                    type="buttons",
                    x=0.5,
                    xanchor="center",
                    y=1.05,
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Total",
                            method="update",
                            args=[
                                {"visible": [True, True, True, False, False]},
                                {
                                    "yaxis2.title": str(results["X"]),
                                    "yaxis3.title": str(results["P"]),
                                },
                            ],
                        ),
                        dict(
                            label="Concentrations",
                            method="update",
                            args=[
                                {"visible": [True, False, False, True, True]},
                                {
                                    "yaxis2.title": str(results["x"]),
                                    "yaxis3.title": str(results["p"]),
                                },
                            ],
                        ),
                    ],
                )
            ],
        )

        self._fig = fig

    def update(self, **kwargs):
        mu_or_F = kwargs[self.mu_or_F]
        V_frac = kwargs["V_frac"]
        # get the trajectory of the new process
        df = self.get_process_trajectory(mu_or_F, V_frac)
        # update the traces
        for i, col in enumerate(["V", "X", "P", "x", "p"]):
            self.fig.update_traces(selector=i, x=df.index, y=df[col])
        # update the vertical line at `t_switch`
        t_switch = self.df.loc[(mu_or_F, V_frac)]["t_switch"]
        self.fig.update_shapes(selector=0, x0=t_switch, x1=t_switch)
