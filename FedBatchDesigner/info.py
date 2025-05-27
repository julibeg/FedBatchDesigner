from contextlib import contextmanager

from shiny.express import expressify, ui


@contextmanager
@expressify
def formula_list():
    with ui.tags.ul(class_="formula-list") as ul:
        yield ul


@expressify
def info():
    # CSS for list of text + formula pairs
    ui.tags.style(
        """
        .formula-item {
            display: flex;
            align-items: center; /* Center items vertically */
        }
        .formula-item .text {
            flex: 1; /* text takes up half the space */
        }
        .formula-item .math {
            flex: 1; /* formula takes up the remaining space */
            text-align: center; /* center align formulas */
        }

        /* add vertical gap between items in the UL */
        .formula-list {
            display: flex;
            flex-direction: column;
            gap: 2em;
        }
        """
    )

    with ui.card():
        ui.card_header("Background")
        ui.tags.p(
            """
            The aim of this web tool is to explore the design space of fed-batch
            fermentations with a growth-arrested second stage. The feed profile of the
            first stage is either constant, linear, or exponential. During the second,
            growth-arrested, stage, the feed rate is kept constant (at a value
            determined by the maintenance coefficient \(\\rho\) and growth-independent
            production rate \(\pi_0\)).
            """,
        )
        with ui.layout_column_wrap(width=1 / 2):
            with ui.card():
                ui.tags.p("The tunable variables for the process design are:")
                with ui.tags.ul():
                    ui.tags.li(
                        """
                        The parameter determining the feed profile during the first
                        stage:
                        """,
                        ui.tags.ul(
                            ui.tags.li(
                                ui.HTML("The feed rate \(F\) for <b>constant feed</b>.")
                            ),
                            ui.tags.li(
                                ui.HTML(
                                    """
                                    The absolute growth rate \(dX\) for <b>linear
                                    feed</b>. This is the rate of change of the total
                                    amount of biomass during the first stage (e.g. 2
                                    g/h). The initial feed rate \(F_0\) and its change
                                    over time \(dF\) are chosen such that \(dX\) is
                                    constant (for details see below).
                                    """
                                ),
                            ),
                            ui.tags.li(
                                ui.HTML(
                                    """
                                    The specific growth rate \(\mu\) for <b>exponential
                                    feed</b>. The initial feed rate \(F_0\) is chosen
                                    such that the feed rate \(F\) and the total amount
                                    of biomass \(X\) both grow exponentially and at the
                                    same rate (for details see below).
                                    """
                                ),
                            ),
                        ),
                    )
                    ui.tags.li(
                        """
                        The point of switching between the first and second stage
                        (\(t_{switch}\)) or, more practically, the fraction of the total
                        feed volume that is fed during the first stage (\(V_{frac} =
                        \\frac{V_{s_1}}{V_{tot}}\)).
                        """
                    )
            with ui.card():
                with ui.tags.div(
                    style="""
                        display: flex;
                        flex-direction: column;
                        justify-content: space-around;
                        height: 100%;
                        """
                ):
                    ui.tags.p(
                        """
                        Outputs include visualizations of how the average volumetric
                        productivity (also called space-time yield) and the final
                        product titer relate to these parameters.
                        """
                    )
                    ui.tags.p(
                        """
                        Thanks to analytical solutions for the calculation of biomass
                        and product formation in the first stage, the whole design space
                        can be evaluated with a brute force approach in little time. The
                        assumptions that those analytical formulas rely upon are usually
                        satisfied in microbial fed-batch settings with a growth-arrested
                        second stage (for details on assumptions see below).
                        """
                    )
                    ui.tags.p(
                        """
                        One crucial aspect to note is that we assume no growth in the
                        second stage with the feed rate held constant at the exact value
                        that satisfies the substrate requirements for maintenance and
                        product formation (i.e. the best case scenario).
                        """
                    )

    @expressify
    def formula_item(text, formula):
        with ui.tags.li():
            with ui.tags.div(class_="formula-item"):
                ui.tags.span(text, class_="text")
                ui.tags.span(formula, class_="math")

    with ui.card():
        ui.card_header("Underlying assumptions")
        ui.h5("General (both stages):")
        with formula_list():
            formula_item(
                "No substrate accumulation.",
                r"\(S = 0\)",
            )
            formula_item(
                "Product formation is proportional to biomass and growth.",
                "\(\dot P = X \cdot \pi_0 + \dot X \cdot \pi_1\)",
            )
            formula_item(
                "There is a maintenance requirement.",
                r"\(M = X \cdot \rho\)",
            )
        ui.h5("Stage 1: Growth phase")
        with formula_list():
            formula_item(
                """
                Biomass growth is determined by the amount of substrate remaining after
                taking maintenance and product formation into account.
                """,
                r"""\(
                    \dot X = Y_{X/S} \cdot
                    (F \cdot s_F - \frac{M}{Y_{ATP/S}} - \frac{\dot P}{Y_{P/S}})
                \)""",
            )
            formula_item(
                """
                The feed rate is constant in case of the constant feed strategy.
                """,
                r"\(\dot V = F = const\)",
            )
            formula_item(
                """
                For the linear feed strategy, the feed rate increases linearly with
                time. By choosing the appropriate initial feed rate \(F_0\) and rate of
                change \(dF\) (taking the substrate requirements for maintenance and
                product formation into account), total biomass growth is ensured to also
                be linear. In other words, when the user increases the value for the
                maintenance factor \(\\rho\), \(F_0\) and \(dF\) will increase
                accordingly and biomass still grows in a linear fashion throughout the
                whole first stage.
                """,
                r"""\(
                    \begin{gather}
                        F = F_0 + t \cdot dF\\
                        \mu_0 = \frac{dX}{X_0}\\
                        F_0 = \frac{X_0}{s_F} \left(\frac{\mu_0}{Y_{X/S}} +
                            \frac{\rho}{Y_{ATP/S}} + \frac{\pi_0 +
                            \mu_0 \pi_1}{Y_{P/S}}\right)\\
                        \alpha = \frac{1}{1 + \pi_1 \frac{Y_{X/S}}{Y_{P/S}}}\\
                        \beta = \pi_0\frac{Y_{X/S}}{Y_{P/S}}
                            + \rho\frac{Y_{X/S}}{Y_{ATP/S}}\\
                        dF = \alpha \beta
                            (F_0 - \frac{X_0 \cdot \beta}{s_F \cdot Y_{X/S}})
                    \end{gather}
                \)""",
            )
            formula_item(
                """
                In case of the exponential feed strategy, total biomass and the feed
                rate both grow exponentially and at the same rate (again by choosing the
                appropriate initial feed rate \(F_0\) similar to the linear feed).
                """,
                r"""\(
                    \begin{gather}
                        F = F_0 \cdot e^{\mu t}\\
                        F_0 = \frac{X_0}{s_F} \left(\frac{\mu}{Y_{X/S}} +
                            \frac{\rho}{Y_{ATP/S}} + \frac{\pi_0 +
                            \mu \pi_1}{Y_{P/S}}\right)
                    \end{gather}
                \)""",
            )
        ui.h5("Stage 2: Production phase")
        with formula_list():
            formula_item(
                "No growth",
                r"\(\dot X = 0\)",
            )
            formula_item(
                """
                The feed rate is constant at a value that exactly satisfies maintenance
                and product formation.
                """,
                r"""\(
                    F = \frac{1}{s_F} (\frac{M}{Y_{ATP/S}} + \frac{\dot P}{Y_{P/S}})
                \)""",
            )

    with ui.layout_column_wrap(width=1 / 2):
        with ui.card():
            ui.card_header("Limitations")
            ui.tags.li(
                """
                Productivity and volumetric productivity relate only to the feed phase
                (i.e. they don't take the duration of and amount of product produced
                during the batch phase into account).
                """
            )
            ui.tags.li(
                """
                Non-feed volume changes (e.g. evaporation, base addition for pH control)
                are not considered.
                """
            )
        with ui.card():
            ui.card_header("Extra considerations for anaerobic fermentation products")
            ui.tags.p(
                """
                The framework assumes that substrate is consumed for either maintenance,
                biomass, or product formation independently. In the case of anaerobic
                fermentation products (like ethanol or butanol), however, this is not
                the case because then no substrat is fully "used up" for ATP production.
                Instead, ATP is produced via substrate-level phosphorylation
                concomitantly with product formation. To represent this in the model,
                the maintenance coefficient \(\\rho\) is set to zero. This is done
                automatically when ticking the corresponding checkbox in the
                stage-specific input panels.
                """
            )
