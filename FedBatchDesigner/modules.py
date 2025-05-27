import functools

from shiny import reactive
from shiny.express import ui, expressify, module
from shiny_validate import InputValidator

import icons
import params
import util


@module
def stage_specific_inputs(input, _output, session, stage_idx, inputs_obj):
    # get the module ID to use as label for the local validator and prefix for the input
    # IDs below
    module_id = session.ns
    # create a local `InputValidator` for this module and add it to the global one
    local_validator = InputValidator()
    inputs_obj.add_validator(local_validator, label=module_id)

    @expressify
    def text_content(stage_idx):
        with ui.div():
            ui.tags.p(
                """
                These parameters (substrate concentration in the feed,
                yield coefficients, etc.) can change between the two
                stages. Substrate uptake (as determined by the yield
                coefficients and specific rates) always matches the
                amount of substrate added in the feed:
                """
            )
            ui.tags.p(
                r"""\(
                F s_f = X \left(\frac{\mu}{Y_{X/S}} + \frac{\pi_0 +
                \mu \pi_1}{Y_{P/S}} + \frac{\rho}{Y_{ATP/S}}\right)
                \) .""",
                style=("text-align: center; font-size: 130%;"),
            )
            ui.tags.p(
                """
                As there is no growth in the second stage (\(\mu = 0\)),
                some of these parameters are only relevant (and thus can
                only be set) for the first stage.
                """
            )
            if stage_idx == 2:
                ui.tags.p(
                    """
                Parameters for the second stage are optional (values
                from the first stage will be used if left blank).
                """
                )

    @expressify
    def anaerobic_checkbox():
        with ui.div():
            ui.input_checkbox(
                "anaerobic",
                "Anaerobic fermentative product",
                value=False,
            )
            ui.p(
                """
                Tick this box if your product is generated in anaerobic fermentation
                (e.g. ethanol or lactic acid) and this stage is anaerobic.
                """
            )

    if stage_idx == 1:
        # show the text first and then the inputs below
        text_content(stage_idx)

        with ui.card():
            with ui.layout_column_wrap(width=1 / 3):
                with ui.div(style="display: flex; align-items: center; height: 100%;"):
                    anaerobic_checkbox()
                for k in ["mu_max_phys", "s_f"]:
                    v = params.stage_specific[k]
                    with ui.div():
                        input_id = f"{module_id}_{k}"
                        inputs_obj.add_input(
                            id=input_id, label=str(v), validator=local_validator
                        )
                        inputs_obj[input_id].add_rule(
                            functools.partial(util.validate_param, required=True)
                        )
                        ui.tags.p(v.description)

        # add a custom validation rule for `s1_mu_max_phys` to ensure it is at least as
        # big as the maximum growth rate of the feed
        def validate_mu_max_phys(value):
            # we still need to call `util.validate_param()` since `InputValidator` only
            # keeps track of the last validation rule (i.e. `util.validate_param` added
            # as rule above is overwritten)
            msg = util.validate_param(value, True)
            if msg is not None:
                return msg
            if not value:
                return
            try:
                # only compare with `mu_max_feed` if it has already been provided
                mu_max_feed = float(inputs_obj["mu_max_feed"].get())
            except ValueError:
                return
            if float(value) < mu_max_feed:
                return "must be at least as big as the maximum growth rate of the feed"

        inputs_obj["s1_mu_max_phys"].add_rule(validate_mu_max_phys)

    else:
        with ui.layout_columns(col_widths=(8, 4)):
            # `mu_max_phys` only makes sense for stage 1; put text into
            # left column and `mu_max_feed` input into the rgith
            text_content(stage_idx)
            with ui.div(
                style=("display: flex;" "align-items: center;" "height: 100%;")
            ):
                with ui.card():
                    anaerobic_checkbox()
                    k = "s_f"
                    v = params.stage_specific[k]
                    with ui.div():
                        input_id = f"{module_id}_{k}"
                        inputs_obj.add_input(
                            id=input_id, label=str(v), validator=local_validator
                        )
                        inputs_obj[input_id].add_rule(
                            functools.partial(util.validate_param, required=False)
                        )
                        ui.tags.p(v.description)

    with ui.card():
        ui.card_header("Yield coefficients")
        with ui.layout_column_wrap(width=1 / (3 if stage_idx == 1 else 2)):
            for k, v in params.yields.items():
                if stage_idx == 2 and v.stage_1_only:
                    continue
                input_id = f"{module_id}_{k}"
                inputs_obj.add_input(
                    id=input_id, label=str(v), validator=local_validator
                )
                inputs_obj[input_id].add_rule(
                    functools.partial(util.validate_param, required=stage_idx == 1),
                )
        ui.tags.p(
            """
            Yield coefficients of biomass, product, and ATP
            (in grams per gram substrate consumed).
            """
        )

    with ui.card():
        ui.card_header("Specific rates")
        with ui.layout_column_wrap(width=1 / (3 if stage_idx == 1 else 2)):
            for k, v in params.rates.items():
                if stage_idx == 2 and v.stage_1_only:
                    continue
                with ui.div():
                    input_id = f"{module_id}_{k}"
                    inputs_obj.add_input(
                        id=input_id, label=str(v), validator=local_validator
                    )
                    inputs_obj[input_id].add_rule(
                        functools.partial(
                            util.validate_param,
                            required=stage_idx == 1,
                        ),
                    )
                    ui.tags.p(v.description)

    @reactive.effect
    @reactive.event(input.anaerobic)
    async def anaerobic():
        """Disable the `rho` input if the stage is anaerobic."""
        rho_input = inputs_obj[f"{module_id}_rho"]
        await rho_input.set_disabled(input.anaerobic())
        rho_input.set(0 if input.anaerobic() else "")
