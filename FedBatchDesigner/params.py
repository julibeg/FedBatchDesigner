from dataclasses import dataclass


@dataclass
class _Param:
    label: str
    unit: str
    description: str

    def __str__(self):
        if self.unit:
            return f"{self.label} [{self.unit}]"
        return self.label


@dataclass
class InputParam(_Param):
    required: bool = True
    stage_1_only: bool = False


@dataclass
class ResultParam(_Param):
    short_label: str | None = None

    def short_str(self):
        if self.short_label is None:
            return self.__str__()
        return f"{self.short_label} [{self.unit}]"


batch = {
    "V_batch": InputParam(
        "\(V_\\textrm{batch}\)",
        "L",
        "Fluid volume in the reactor at the end of the batch phase",
    ),
    "x_batch": InputParam(
        "\(x_\\textrm{batch}\)",
        "g/L",
        "Biomass concentration at the end of the batch phase",
    ),
}

feed = {
    "V_max": InputParam(
        "\(V_\\textrm{max}\)", "L", "Maximum available volume in the reactor"
    ),
    "mu_max_feed": InputParam(
        "\(\mu_\\textrm{max}^{F,\\textrm{exp}}\)",
        "/h",
        "Maximum specific growth rate to consider when optimizing exponential feed",
    ),
    "F_max": InputParam(
        "\(F_\\textrm{max}\)",
        "L/h",
        "Maximum feed rate of the reactor",
    ),
}

common = {**batch, **feed}

yields = {
    "Y_XS": InputParam(
        "\(Y_{X/S}\)",
        "g CDM / g substrate",
        (
            "Biomass yield coefficient "
            "(grams biomass produced per gram of substrate consumed)"
        ),
        stage_1_only=True,
    ),
    "Y_PS": InputParam(
        "\(Y_{P/S}\)",
        "g product / g substrate",
        (
            "Product yield coefficient "
            "(grams product formed per gram of substrate consumed)",
        ),
    ),
    "Y_AS": InputParam(
        "\(Y_{ATP/S}\)",
        "g ATP / g substrate",
        "ATP yield coefficient (grams ATP generated per gram of substrate consumed)",
    ),
}

rates = {
    "rho": InputParam(
        "\(\\rho\)",
        "g ATP / (g CDM h)",
        "Maintenance factor (specific ATP consumption rate)",
    ),
    "pi_0": InputParam(
        "\(\pi_0\)",
        "g product / (g CDM h)",
        "Non-growth-associated specific product formation rate",
    ),
    "pi_1": InputParam(
        "\(\pi_1\)",
        "g product / g CDM",
        "Growth-associated specific product formation rate",
        stage_1_only=True,
    ),
}

stage_specific = {
    **yields,
    **rates,
    "mu_max_phys": InputParam(
        "\(\mu_\\textrm{max}^\\textrm{phys}\)",
        "/h",
        "Maximum specific growth rate (physiological)",
        stage_1_only=True,
    ),
    "s_f": InputParam(
        "\(s_f\)",
        "g / L",
        "Substrate concentration in the feed",
    ),
}

input_ids = [
    *[key for key in common.keys()],
    *[f"s1_{key}" for key in stage_specific.keys()],
    *[
        f"s2_{key}"
        for key in stage_specific.keys()
        if not stage_specific[key].stage_1_only
    ],
]

results = {
    "mu": ResultParam(
        "µ",
        "/h",
        "Specific growth rate",
    ),
    "mu_max": ResultParam(
        "Maximum specific growth rate",
        "/h",
        """The largest specific growth rate encountered during the feed phase""",
    ),
    "F": ResultParam(
        "Feed rate first stage",
        "L/h",
        "Constant feed rate during the growth stage",
        short_label="F",
    ),
    "G": ResultParam(
        "Absolute growth rate in first stage",
        "g/h",
        "constant absolute growth rate during the growth stage",
        short_label="G",
    ),
    "F0": ResultParam(
        "Initial feed rate",
        "L/h",
        "Feed rate at t=0 of the exponential or linear feed phase",
    ),
    "dF": ResultParam(
        "Feed rate change",
        "(L/h)/h",
        "Change of feed rate per hour (for linear feed)",
    ),
    "F_end": ResultParam(
        "Final feed rate in first stage",
        "L/h",
        "The feed rate at the end of the growth stage",
    ),
    "V_frac": ResultParam(
        "Fraction of feed volume in first stage",
        "",
        """
        The fraction of the total feed volume fed during the growth stage (i.e. before
        t_switch)
        """,
    ),
    "V1": ResultParam(
        "Volume after first stage",
        "L",
        "Reactor volume at the end of the growth stage",
    ),
    "V2": ResultParam(
        "Final volume",
        "L",
        "Reactor volume at the end of the process",
    ),
    "productivity": ResultParam(
        "Productivity",
        "g/h",
        "Total amount of product formed during the feed phase divided by feed time",
    ),
    "space_time_yield": ResultParam(
        "Space-time yield",
        "g/(L h)",
        """
        Total amount of product formed during the feed phase divided by final reactor
        volume and feed time. Sometimes also referred to as "average volumetric
        productivity". Corresponds to the 'R' in the TRY metrics
        """,
    ),
    "substrate_yield": ResultParam(
        "Per-substrate yield",
        "g/g",
        """
        Total amount of product formed during the feed phase divided by the total amount
        of substrate used during feed. Corresponds to the 'Y' in the TRY metrics
        """,
    ),
    "substrate_start_volume": ResultParam(
        "Substrate start volume",
        "L",
        """
        Amount of "virtual" start feed volume needed to set up the exponential feed so
        that biomass indeed grows with the desired exponential rate (despite some
        substrate being spent on maintenance and product formation)
        """,
    ),
    "P": ResultParam(
        "Product",
        "g",
        "Grams of product (in total) that have been formed by this point in time",
    ),
    "p": ResultParam(
        "Product concentration",
        "g/L",
        "Product concentration at this point in time",
    ),
    "P2": ResultParam(
        "Total product",
        "g",
        "Amount of product produced in total",
    ),
    "p2": ResultParam(
        "Final titer",
        "g/L",
        """
        Product concentration at the end of the process. Note that this ignores any
        product formed during the batch phase. Corresponds to the 'T' in the TRY
        metrics
        """,
    ),
    "F2": ResultParam(
        "Feed rate second stage",
        "L/h",
        "Feed rate (constant) in the no-growth phase of the fed-batch",
    ),
    "t_end": ResultParam(
        "Total feed time",
        "h",
        "Total time of the feed phase",
    ),
    "t_switch": ResultParam(
        "Switch time",
        "h",
        "Time at which the switch from the first to the second stage occurs",
    ),
    "t": ResultParam(
        "Time",
        "h",
        "",
    ),
    "V": ResultParam(
        "Volume",
        "L",
        "Reactor volume at this point in time",
    ),
    "X": ResultParam(
        "Total biomass",
        "g",
        "Total biomass in the reactor at this point in time",
    ),
    "x": ResultParam(
        "Biomass concentration",
        "g/L",
        "Biomass concentration at this point in time",
    ),
    "X2": ResultParam(
        "Total biomass",
        "g",
        "Total biomass at the end of the process",
    ),
    "x2": ResultParam(
        "Final biomass concentration",
        "g/L",
        "Biomass concentration at the end of the process",
    ),
}

# physiological parameters for E. coli taken from Klamt et al. (2018)
# (https://doi.org/10.1002/biot.201700539)
defaults_E_coli = {
    "s1_Y_XS": 98 / 180,  # g CDM / g glc
    "s1_Y_AS": 23.5 * 507 / 180,  # g ATP / g glc (23.5 mol ATP / mol glc)
    "s1_rho": 7.7e-3 * 507,  # g ATP / (g CDM h)
    "s1_mu_max_phys": 0.6,  # /h (approximate value for E. coli on common minimal media)
}

# physiological parameters for S. cerevisiae taken from Vos et al. (2016)
# (https://doi.org/10.1186/s12934-016-0501-z)
defaults_S_cerevisiae = {
    "s1_Y_XS": 0.5,  # g CDM / g glc
    # for ATP yield from glucose:
    # - rate of ATP usage for maintenance: 0.63 mmol ATP / (g CDM h)
    # - rate of glucose consumption for ATP: 0.039 mmol glc / (g CDM h);
    "s1_Y_AS": 0.63e-3 * 507 / (0.039e-3 * 180),
    "s1_rho": 0.63e-3 * 507,  # g ATP / (g CDM h)
}

# unless otherwise noted, values for the L-valine case study are taken from the fit to
# Fig. 5 of Hao et al. (https://doi.org/10.1016/j.ymben.2020.09.007); the fit assumed
# `OD_to_x = 0.33` (see notebook in case studies for details)
defaults_case_study_valine_one_stage = defaults_E_coli | {
    "V_batch": 3,  # L (from paper)
    "x_batch": 6.7,  # g/L
    "V_max": 4.17,  # L
    "mu_max_feed": 0.3,  # /h (50% of `mu_max_phys`)
    "F_max": 0.5,  # L/h (reasonable value for 3-10 L reactor scale pumps)
    "s1_s_f": 800,  # g glc / L (from paper & confirmed in correspondence with authors)
    "s1_Y_PS": 0.65,  # g product / g glc (theoretical maximum)
    "s1_pi_0": 0.073,  # g product / (g CDM h)
    "s1_pi_1": 0.298,  # g product / g CDM
}

defaults_case_study_valine_two_stage = defaults_case_study_valine_one_stage | {
    "V_max": 4.1,  # L
    "s2_Y_PS": 0.46,  # g product / g glc (from paper)
    # we set `rho` to zero for stage 2; this is necessary in anaerobic conditions when
    # the synthesis of the product is used to regenerate NAD+ (since technically no
    # glucose is fully "consumed" to generate ATP in such a case; we get ATP from
    # glycolysis, but the resulting pyruvate is used for valine production)
    "s2_rho": 0,
    # `Y_AS` is a lot lower in anaerobic conditions; however, this is irrelevant because
    # we're setting `rho=0` in stage 2 anyway
    "s2_Y_AS": 2 * 507 / 180,  # g ATP / g glc (this is irrelevant since `rho=0`)
    "s2_pi_0": 0.114,  # g product / (g CDM h)
}

# The `pi` values for the S. cerevisiae ethanol case study were fit from data in Zahoor
# et al. (2020) (https://doi.org/10.1186/s13068-020-01822-9). See notebook in case
# studies for details).
s_cerevisiae_ethanol_atp_wasting = {
    "V_batch": 3,  # L
    "x_batch": 10,  # g/L
    "V_max": 4.5,  # L
    "F_max": 1,  # L/h
    "s1_s_f": 500,
    "mu_max_feed": 0.2,  # /h
    # anaerobic biomass yield on glucose taken from Zakhartsev et al. (2015)
    # (https://doi.org/10.1016/j.jtherbio.2015.05.008)
    "s1_Y_XS": 0.141,  # g CDM / g glucose (anaerobic)
    "s1_Y_PS": 0.511,  # g ethanol / g glucose (theoretical)
    "s1_Y_AS": 2 * 507 / 180,  # g ATP / g glucose (2 mol ATP / mol glucose)
    "s1_mu_max_phys": 0.235,
    # like for valine above we set `rho` to zero as this is a fermentative product
    "s1_rho": 0,  # g ATP / (g CDM h)
    # the `s1_pi` values were fit from the data in Figure 2 of Zahoor et al. (2020)
    "s1_pi_0": 0.454,  # g ethanol / (g CDM h)
    "s1_pi_1": 3.62,  # g ethanol / g CDM
    # `s2_pi_0` was taken from Table 3 of Zahoor et al. (2020)
    "s2_pi_0": 0.263,  # g ethanol / (g CDM h)
}

defaults = {
    "e_coli": {
        "title": "<i>E. coli</i> (aerobic)",
        "description": """
            Physiological parameters for <i>Escherichia coli</i> (biomass yield, ATP
            yield, maintenance factor) taken from <a
            href="https://doi.org/10.1002/biot.201700539">Klamt et al. (2018)</a>.
            """,
        "values": defaults_E_coli,
    },
    "s_cerevisiae": {
        "title": "<i>S. cerevisiae</i> (aerobic)",
        "description": """
            Physiological parameters for <i>Saccharomyces cerevisiae</i> (biomass yield,
            ATP yield, maintenance factor) taken from <a
            href="https://doi.org/10.1186/s12934-016-0501-z">Vos et al. (2016)</a>.
            """,
        "values": defaults_S_cerevisiae,
    },
    "valine_one_stage": {
        "title": "Case study: L-valine (one-stage)",
        "description": """
            <a href="https://doi.org/10.1016/j.ymben.2020.09.007">Hao et al. (2020)</a>
            achieved impressive titers and productivities of L-valine with an engineered
            <i>E. coli</i> strain. They performed an aerobic one-stage fed-batch process
            and a two-stage process with a microaerobic second stage. The values below
            are based on a fit to the data in Fig. 5 of the publication.
            """,
        "values": defaults_case_study_valine_one_stage,
    },
    "valine_two_stage": {
        "title": "Case study: L-valine (two-stage)",
        "description": """
            Parameters obtained for the two-stage process of <a
            href="https://doi.org/10.1016/j.ymben.2020.09.007">Hao et al. (2020)</a>.
        """,
        "values": defaults_case_study_valine_two_stage,
    },
    "s_cerevisiae_ethanol_atp_wasting": {
        "title": "Case study: ethanol production with ATP wasting",
        "description": """
        <a href="https://doi.org/10.1186/s13068-020-01822-9">Zahoor et al. (2020)</a>
        enhanced ethanol production in <i>S. cerevisiae</i> through enforced ATP wasting
        and nitrogen starvation. <tt>FedBatchDesigner</tt> could be used to inform
        fed-batch experiments after their initial shake flask cultivations.
        """,
        "values": s_cerevisiae_ethanol_atp_wasting,
    },
}
