from shiny.express import expressify, ui


@expressify
def info():
    # CSS for list of text + formula pairs
    ui.tags.style(
        """
        .formula-item {
            display: flex;
            align-items: center; /* Center items vertically */
            margin-bottom: 0.5em; /* Add spacing between list items */
        }
        .formula-item .text {
            flex: 1; /* text takes up half the space */
        }
        .formula-item .math {
            flex: 1; /* formula takes up the remaining space */
            text-align: center; /* center align formulas */
        }
        """
    )

    with ui.card():
        ui.card_header("Background")
        ui.tags.p(
            """
            The aim of this web tool is to explore the design space of fed-batch
            fermentations with a growth-arrested second stage. The feed profile of the
            first stage is either exponential or constant, while the feed rate is kept
            constant during the second stage (at a value determined by the
            maintenance coefficient \(\\rho\) and growth-independent production rate
            \(\pi_0\)).
            """,
        )
        ui.tags.p("The tunable variables for the process design are:")
        with ui.tags.ul():
            ui.tags.li(
                """
                The feed rate ((\(F\), for constant feed) or specific growth rate
                ((\(\mu\), for exponential feed) during the first stage.
                """
            )
            ui.tags.li(
                """
                The point of switching between the first and second stage
                (\(t_{switch}\)) or, more practically, the fraction of the total feed
                volume that is fed during the first stage (\(V_{frac} =
                \\frac{V_{s_1}}{V_{tot}}\)).
                """
            )
        ui.tags.p(
            """
            Outputs include visualizations of how the average volumetric productivity
            (also called space-time yield) and the final product titer relate to these
            parameters.
            """
        )
        ui.tags.p(
            """
            Thanks to analytical solutions for the calculation of biomass and product
            formation in the first stage, the whole design space can be evaluated with a
            brute force approach in little time. The assumptions that those analytical
            formulas rely upon are usually satisfied in microbial fed-batch settings
            (for details on assumptions see below).
            """
        )
        ui.tags.p(
            """
            One crucial aspect to note is that we assume no growth in the second stage
            with the feed rate held constant at the exact value that satisfies the
            substrate requirements for maintenance and product formation (i.e. the best
            case scenario).
            """
        )

    @expressify
    def formula_item(text, formula):
        with ui.tags.li():
            with ui.tags.div(class_="formula-item"):
                ui.tags.span(
                    text,
                    class_="text",
                )
                ui.tags.span(
                    formula,
                    class_="math",
                )

    with ui.card():
        ui.card_header("Underlying assumptions")
        ui.h5("General (both stages):")
        with ui.tags.ul(class_="formula-list"):
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
        with ui.tags.ul():
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
                In case of the exponential feed strategy, biomass and feed rate both
                grow exponentially and with the same (constant) rate. This is ensured by
                choosing the appropriate "substrate start volume" (\(V_{virt}\), for
                details see below), which takes the substrate requirements for
                maintenance and product formation into account. In other words, when the
                user increases the value for the maintenance factor \(\rho\), the
                substrate start volume will increase as well to ensure biomass still
                grows with the same \(mu\).
                """,
                r"""\(
                    \begin{gather}
                        \mu = const\\
                        \dot V = F = V_{virt} \cdot \mu \cdot e^{\mu t}\\
                        V_{virt} = \frac{X_0}{s_F~\mu} \left(\frac{\mu}{Y_{X/S}} +
                            \frac{\rho}{Y_{ATP/S}} + \frac{\pi}{Y_{P/S}}\right)
                    \end{gather}
                \)""",
            )
            formula_item(
                """
                As the name suggests, the feed rate is constant in case of the constant
                feed strategy.
                """,
                r"\(\dot V = F = const\)",
            )
        ui.h5("Stage 2: Production phase")
        with ui.tags.ul():
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

    with ui.card():
        ui.card_header("Limitations")
        with ui.tags.ul():
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
