import abc

import numpy as np
import pandas as pd
import scipy

from logger import logger
import util

# maximum step size (in hours of time domain) for the integration
MAX_INTEGRATION_STEP = 1


class NotEnoughSubstrateError(Exception):
    pass


class FedBatchStage(abc.ABC):
    def __init__(
        self,
        V0,
        X0,
        P0,
        s_f,
        Y_AS,
        Y_PS,
        Y_XS,
        rho,
        pi_0,
        pi_1,
        mu_max_phys=None,
        debug=False,
    ):
        self.V0 = V0
        self.X0 = X0
        self.P0 = P0
        self.s_f = s_f
        self.rho = rho
        self.Y_AS = Y_AS
        self.Y_PS = Y_PS
        self.Y_XS = Y_XS
        self.pi_0 = pi_0
        self.pi_1 = pi_1
        self.mu_max_phys = mu_max_phys
        self.debug = debug
        # calculate substrate required for maintenance and product formation at t=0
        self.initial_substrate_for_rho_and_pi_0 = X0 * (rho / Y_AS + pi_0 / Y_PS)
        # these expressions are handy for some calculations
        self.alpha = 1 / (1 + pi_1 * Y_XS / Y_PS)
        self.beta = pi_0 * Y_XS / Y_PS + rho * Y_XS / Y_AS
        # calculate the smallest possible constant feed rate (with any feed rate smaller
        # than this, not enough substrate for maintenance and product formation would be
        # added in the first instance of the batch)
        self.F0_min = self.initial_substrate_for_rho_and_pi_0 / self.s_f

    @abc.abstractmethod
    def evaluate_at_V(self, V, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_at_t(self, t, **kwargs):
        raise NotImplementedError


# define mixin classes to add some functionality to exponential vs constant feed stages
class ConstantFeed:
    @property
    def F_min(self):
        return self.F0_min

    def dV(self, F, t):
        return F

    def calculate_initial_mu_from_F(self, F):
        """Calculate the specific growth rate at the first instance of constant feed."""
        # put parameters into variables for convenience
        Y_XS = self.Y_XS
        Y_AS = self.Y_AS
        Y_PS = self.Y_PS
        s_f = self.s_f
        rho = self.rho
        pi_0 = self.pi_0
        pi_1 = self.pi_1
        X = self.X0

        mu = (
            Y_XS
            * (F * Y_AS * Y_PS * s_f - X * Y_AS * pi_0 - X * Y_PS * rho)
            / (X * Y_AS * (Y_PS + Y_XS * pi_1))
        )

        return mu

    def calculate_F_from_initial_mu(self, mu):
        """
        Calculate feed rate required to achieve a certain specific growth rate at the
        beginning of the feed.
        """
        Y_XS = self.Y_XS
        Y_AS = self.Y_AS
        Y_PS = self.Y_PS
        s_f = self.s_f
        rho = self.rho
        pi_0 = self.pi_0
        pi_1 = self.pi_1
        X = self.X0

        F = (
            X
            * (Y_AS * Y_PS * mu + Y_AS * Y_XS * (mu * pi_1 + pi_0) + Y_PS * Y_XS * rho)
            / (Y_AS * Y_PS * Y_XS * s_f)
        )
        return F


class LinearFeed(ConstantFeed):
    def dV(self, dF, F0, t):
        return dF * t + F0


class LinearFeedConstantGrowth(LinearFeed):
    """
    Linear feed profile with `F0` and `dF` chosen such that the absolute growth rate is
    constant (e.g. 2 g/h) throughout the whole feed phase.
    """

    def dV(self, G, t):
        F0, dF = self.F0_and_dF_for_constant_growth(G)
        return super().dV(t=t, dF=dF, F0=F0)

    def F0_for_constant_growth(self, dF):
        """This is the initial feed rate that ensures biomass growth is constant."""
        return self.X0 * self.beta / self.s_f / self.Y_XS + dF / self.alpha / self.beta

    def F0_and_dF_for_constant_growth(self, G):
        """
        Calculate `F0` and `dF` that ensure constant growth `G = X * mu` throughout the
        feed.
        """
        # `G` is constant throughout the feed, i.e. we can use it to get `mu` at `t=0`
        # and then calculate `F0` that way
        mu_0 = G / self.X0
        F0 = (
            self.X0
            / self.s_f
            * (
                mu_0 / self.Y_XS
                + self.rho / self.Y_AS
                + (self.pi_0 + mu_0 * self.pi_1) / self.Y_PS
            )
        )
        dF = (
            (F0 - self.X0 * self.beta / (self.s_f * self.Y_XS)) * self.alpha * self.beta
        )
        return F0, dF

    def get_G_max_from_F_max(self, V_end, F_max):
        """
        Calculate the maximum constant growth rate that doesn't exceed `F_max` at the
        end of the feed (at `V=V_end`).
        """
        # extract attributes for conciceness in the long formula below
        Y_AS = self.Y_AS
        Y_PS = self.Y_PS
        Y_XS = self.Y_XS
        rho = self.rho
        pi_0 = self.pi_0
        pi_1 = self.pi_1
        alpha = self.alpha
        beta = self.beta
        s_f = self.s_f
        V0 = self.V0
        X0 = self.X0

        # define a few commonly used expressions
        term_1 = Y_XS * alpha**2 * beta**2 * s_f
        term_2 = X0 * alpha * beta**2
        term_3 = Y_PS**2 + 2 * Y_PS * Y_XS * pi_1 + Y_XS**2 * pi_1**2
        term_4 = Y_AS * Y_PS * alpha * beta * s_f

        return (
            Y_AS
            * Y_PS
            * np.sqrt(Y_XS)
            * np.sqrt(s_f)
            * (Y_PS + Y_XS * pi_1) ** 2
            * np.sqrt(
                F_max**2 * Y_XS * s_f
                + V0**2 * term_1
                - 2 * V0 * V_end * term_1
                - 2 * V0 * term_2
                + V_end**2 * term_1
                + 2 * V_end * term_2
            )
            - Y_XS
            * term_3
            * (-V0 * term_4 + V_end * term_4 + X0 * Y_AS * pi_0 + X0 * Y_PS * rho)
        ) / (Y_AS * (Y_PS + Y_XS * pi_1) * term_3)


class ExponentialFeed:
    def dV(self, mu, t):
        return self.F0(mu) * np.exp(mu * t)

    def substrate_start_volume(self, mu):
        if np.isclose(mu, 0):
            return np.inf
        return self.F0(mu) / mu

    def F0(self, mu):
        """
        Caclulate the feed rate that satisfies `dX / dt = X * mu` in the first instance
        of the fed-batch. This is the same as the product of the substrate start volume
        and mu.
        """
        return self.X0 / (self.s_f * self.Y_XS) * (mu / self.alpha + self.beta)

    def calculate_mu_for_F_max(self, F_max, V_end):
        """
        Determine the specific growth rate `mu` that ensures that the feed rate never
        exceeds `F_max` (i.e. it makes sure that `F = F_max` at `V=V_end`).
        """
        if V_end <= self.V0:
            raise ValueError(
                f"`V_end` ({V_end}) must be greater than `V0` ({self.V0})."
            )
        if F_max <= self.F0_min:
            raise ValueError(f"`F_max` ({F_max}) must be positive.")

        # put parameters into some variables for convenience
        term = self.X0 / (self.s_f * self.Y_XS * self.alpha)
        numerator = F_max - term * self.beta * self.alpha
        denominator = term + V_end - self.V0

        # make sure numerator and denominator are both positive
        if denominator <= 0 or numerator <= 0:
            raise ValueError(
                "Calculation resulted in zero or negative denominator or numerator. "
                "Looks like problem is ill-defined."
            )

        mu = numerator / denominator

        return mu


class LogisticFeed(ExponentialFeed):
    def dV(self, F_inf, mu, t):
        return F_inf / (1 + np.exp(-mu * t) * (F_inf / self.F0(mu) - 1))


class NoGrowthConstantStage(FedBatchStage, ConstantFeed):
    """
    Process stage with constant feed rate and `mu = 0`.

    The feed rate is such that the added substrate exactly covers the amount required
    for maintenance and product formation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = self.initial_substrate_for_rho_and_pi_0 / self.s_f
        self.dP = self.X0 * self.pi_0

    def evaluate_at_t(self, t):
        if not util.is_iterable(t):
            t = [t]
        # make sure we got an array
        t = np.asarray(t)
        V = self.V0 + self.F * t
        P = self.P0 + self.dP * t

        df = pd.DataFrame(columns=["V", "X", "P"], index=t)
        df.index.name = "t"
        df["V"] = V
        df["X"] = self.X0
        df["P"] = P
        return df

    def evaluate_at_V(self, V):
        # make sure `V` is an iterable
        if not util.is_iterable(V):
            V = [V]
        V = np.array(V)

        t = (V - self.V0) / self.F
        return self.evaluate_at_t(t)


class FedBatchStageIntegrate(FedBatchStage):
    """
    Base class for process stages that numerically integrate ODEs to calculate
    concentration profiles etc.
    """

    # define the maximum time to integrate; this should never be reached as we terminate
    # the integration when a condition is met
    t_max = 1e5

    @abc.abstractmethod
    def dV(self):
        """Child classes inheriting mostly differ in their implementation of this."""
        raise NotImplementedError

    def get_ODEs(
        self,
        debug_df=None,
        allow_not_enough_for_rho=False,
        allow_not_enough_for_pi=False,
        **dV_kwargs,
    ):
        """Return function representing the right hand side of the ODEs."""

        def ODEs(t, state):
            # unpack the state
            V, X, P = state

            if self.debug:
                debug_df.loc[t] = [V, X, P]

            # get dV and the amount of substrate thus added
            dV = self.dV(t=t, **dV_kwargs)
            substr_add = dV * self.s_f

            # get substrate needed for maintenance and for non-growth-dependent
            # production
            substr_mnt = X * self.rho / self.Y_AS
            dP_pi_0 = X * self.pi_0
            substr_P_pi_0 = dP_pi_0 / self.Y_PS

            # maintenance could require more substrate than was added; make sure to not
            # have negative substrate in that case
            if substr_add < substr_mnt:
                # not enough substrate for maintenance
                if allow_not_enough_for_rho:
                    substr = 0
                else:
                    raise NotEnoughSubstrateError(
                        "More substrate required by maintenance than was added.\n"
                        f"{t=:.5g}, {V=:.5g}, {X=:.5g}, {P=:.5g}, "
                        f"{substr_add=:.5g}, {substr_mnt=:.5g}"
                    )
            else:
                # there's enough substrate for maintenance
                substr = substr_add - substr_mnt
            # make sure that there is enough substrate for product formation
            if substr < substr_P_pi_0:
                if allow_not_enough_for_pi:
                    # use all the remaining substrate for product formation (this will
                    # be less than `X * pi_0`)
                    dP_pi_0 = substr * self.Y_PS
                    substr = 0
                else:
                    raise NotEnoughSubstrateError(
                        "Product formation requires more substrate than was added.\n"
                        f"{t=:.5g}, {V=:.5g}, {X=:.5g}, {P=:.5g}, "
                        f"{substr=:.5g}, {substr_P_pi_0=:.5g}"
                    )
            else:
                substr -= substr_P_pi_0
            # use the remaining substrate for growth and growth-coupled production
            dX = substr / (1 / self.Y_XS + self.pi_1 / self.Y_PS)
            dP_pi_1 = dX * self.pi_1
            dP = dP_pi_0 + dP_pi_1
            return dV, dX, dP

        return ODEs

    def evaluate_at_t(self, t, **dV_kwargs):
        # `dV_kwargs` is passed to `dV`, which needs to be implemented by children. The
        # rest of the functionality doesn't need to change depending on the process
        # stage type.
        if util.is_iterable(t):
            t_end = t[-1]
            t_eval = t
        else:
            t_end = t
            t_eval = [t]
        ODEs = self.get_ODEs(**dV_kwargs)
        res = scipy.integrate.solve_ivp(
            fun=ODEs,
            t_span=(0, t_end),
            t_eval=t_eval,
            y0=(self.V0, self.X0, self.P0),
            max_step=MAX_INTEGRATION_STEP,
        )
        df = pd.DataFrame(
            res.y.T,
            index=res.t,
            columns=["V", "X", "P"],
        )
        df.index.name = "t"
        rates = [ODEs(t, row) for t, row in df.iterrows()]
        df[["dV", "dX", "dP"]] = np.array(rates)
        return df

    def evaluate_at_V(self, V, **dV_kwargs):
        # `dV_kwargs` is passed to `dV`, which needs to be implemented by children. The
        # rest of the functionality doesn't need to change depending on the process
        # stage type. make sure `V` is an iterable
        if not util.is_iterable(V):
            V = [V]

        if self.debug:
            debug_df = pd.DataFrame(columns=["V", "X", "P"])
        else:
            debug_df = None

        # get list of event functions (one for each V_interval)
        events = [lambda _t, state, v=v: state[0] - v for v in V]
        # make the last even terminal
        events[-1].terminal = True

        ODEs = self.get_ODEs(debug_df=debug_df, **dV_kwargs)

        try:
            res = scipy.integrate.solve_ivp(
                fun=ODEs,
                t_span=(0, self.t_max),
                y0=(self.V0, self.X0, self.P0),
                events=events,
                max_step=MAX_INTEGRATION_STEP,
            )
        except NotEnoughSubstrateError as e:
            if self.debug:
                logger.exception(e)
                return debug_df
            else:
                raise e

        df = pd.DataFrame(
            np.vstack(res.y_events),
            index=np.hstack(res.t_events),
            columns=["V", "X", "P"],
        )
        df.index.name = "t"
        rates = [ODEs(t, row) for t, row in df.iterrows()]
        df[["dV", "dX", "dP"]] = np.array(rates)
        return df


class ConstantStageIntegrate(FedBatchStageIntegrate, ConstantFeed):
    def dV(self, *args, **kwargs):
        return ConstantFeed.dV(self, *args, **kwargs)


class LinearStageConstantGrowthIntegrate(
    FedBatchStageIntegrate, LinearFeedConstantGrowth
):
    def dV(self, *args, **kwargs):
        return LinearFeedConstantGrowth.dV(self, *args, **kwargs)


class LinearStageIntegrate(FedBatchStageIntegrate, LinearFeed):
    def dV(self, *args, **kwargs):
        return LinearFeed.dV(self, *args, **kwargs)


class ExponentialStageIntegrate(FedBatchStageIntegrate, ExponentialFeed):
    def dV(self, *args, **kwargs):
        return ExponentialFeed.dV(self, *args, **kwargs)


class LogisticStageIntegrate(FedBatchStageIntegrate, LogisticFeed):
    def dV(self, *args, **kwargs):
        return LogisticFeed.dV(self, *args, **kwargs)


class FedBatchStageAnalytical(FedBatchStage):
    """
    Base class for process stages using analytical expressions for calculating
    process trajectories.
    """

    @abc.abstractmethod
    def t_until_V(self, V, *args, **kwargs):
        """Calculate the time needed to reach V."""
        raise NotImplementedError

    def evaluate_at_V(self, V, **kwargs):
        """
        Blanket implementation so that children only need to implement `t_until_V` and
        evaluate_at_t.
        """
        if not util.is_iterable(V):
            V = [V]
        # for each volume, determine the amount of time needed to get there
        t = np.array([self.t_until_V(v, **kwargs) for v in V])
        return self.evaluate_at_t(t, **kwargs)


class ConstantStageAnalytical(FedBatchStageAnalytical, ConstantFeed):
    def t_until_V(self, V, F):
        if V < self.V0:
            raise ValueError(f"`V` needs to be larger than `V0` (V0={self.V0}).")
        return (V - self.V0) / F

    def evaluate_at_t(self, t, F):
        if not util.is_iterable(t):
            t = [t]
        # make sure we got an array
        t = np.asarray(t)

        # calculate volume
        V = self.V0 + F * t

        # define a few commonly used expressions for sake of conciseness below
        s_f = self.s_f
        Y_XS = self.Y_XS
        pi_0 = self.pi_0
        pi_1 = self.pi_1

        expr_1 = F * s_f * Y_XS / self.beta
        expr_2 = np.exp(-self.alpha * self.beta * t)

        # analytical solution for total biomass
        X = expr_1 + (-expr_1 + self.X0) * expr_2

        # analytical solution for total product
        P = self.P0 + (pi_0 / self.beta) * F * s_f * Y_XS * t + (
            pi_0 / (self.alpha * self.beta) - pi_1
        ) * (self.X0 - expr_1) * (1 - expr_2)

        df = pd.DataFrame({"V": V, "X": X, "P": P}, index=t)
        df.index.name = "t"
        return df


class ExponentialStageAnalytical(FedBatchStageAnalytical, ExponentialFeed):
    def common_expressions(self, mu):
        """
        Evaluate a couple expressions that come up multiple times in the equations for
        the exponential and logistic case.
        """
        expr_1 = self.F0(mu) * self.s_f * self.Y_XS * self.alpha
        expr_2 = self.alpha * self.beta + mu
        return expr_1, expr_2

    def t_until_V(self, V, mu):
        if V < self.V0:
            raise ValueError(f"`V` needs to be larger than `V0` (V0={self.V0}).")
        F0 = self.F0(mu)
        V_rest = V - self.V0
        return np.log(V_rest * mu / F0 + 1) / mu

    def evaluate_at_t(self, t, mu):
        if not util.is_iterable(t):
            t = [t]
        # make sure we got an array
        t = np.asarray(t)

        F0 = self.F0(mu)
        V = self.V0 + (np.exp(mu * t) - 1) * F0 / mu
        # define a few commonly used expressions for sake of conciseness below
        expr_1, expr_2 = self.common_expressions(mu)
        X = (
            np.exp(-t * self.alpha * self.beta)
            * ((-1 + np.exp(t * expr_2)) * expr_1 + self.X0 * expr_2)
            / expr_2
        )
        P = self.P0 + (
            np.exp(-t * self.alpha * self.beta)
            * (
                mu
                * (-expr_1 + self.X0 * expr_2)
                * (-self.pi_0 + self.alpha * self.beta * self.pi_1)
                + np.exp(t * expr_2)
                * expr_1
                * self.alpha
                * self.beta
                * (self.pi_0 + mu * self.pi_1)
                - np.exp(t * self.alpha * self.beta)
                * expr_2
                * (
                    expr_1 * self.pi_0
                    + self.X0 * mu * (-self.pi_0 + self.alpha * self.beta * self.pi_1)
                )
            )
            / (self.alpha * self.beta * mu * expr_2)
        )
        df = pd.DataFrame({"V": V, "X": X, "P": P}, index=t)
        df.index.name = "t"
        return df


class LogisticStageAnalytical(ExponentialStageAnalytical, LogisticFeed):
    def t_until_V(self, V, mu, F_inf):
        if V < self.V0:
            raise ValueError(f"`V` needs to be larger than `V0` (V0={self.V0}).")
        F0 = self.F0(mu)
        t = (
            np.log(
                (
                    F0
                    - F_inf
                    + np.exp((V * mu - self.V0 * mu + np.log(F_inf**F_inf)) / F_inf)
                )
                / F0
            )
            / mu
        )
        return t

    def get_V(self, F_inf, mu, t):
        F0 = self.F0(mu)
        V = (
            F_inf * np.log(F0 * (np.exp(mu * t) - 1) + F_inf)
            - F_inf * np.log(F_inf)
            + mu * self.V0
        ) / mu
        return V

    def get_X(self, F_inf, mu, t):
        F0 = self.F0(mu)
        # define some common terms
        exp_mu_t = np.exp(mu * t)
        exp_alpha_beta_t = np.exp(self.alpha * self.beta * t)
        exp_alpha_beta_neg_t = np.exp(-self.alpha * self.beta * t)
        delta_F = F0 - F_inf
        hyp_term_1 = scipy.special.hyp2f1(
            1,
            self.alpha * self.beta / mu + 1,
            self.alpha * self.beta / mu + 2,
            (exp_mu_t * F0) / delta_F,
        )
        hyp_term_2 = scipy.special.hyp2f1(
            1,
            self.alpha * self.beta / mu + 1,
            self.alpha * self.beta / mu + 2,
            F0 / delta_F,
        )

        # define a few commonly used expressions for sake of conciseness below
        expr_1, expr_2 = self.common_expressions(mu)

        numerator = exp_alpha_beta_neg_t * (
            expr_1 * F_inf * exp_alpha_beta_t * exp_mu_t * hyp_term_1
            - expr_1 * F_inf * hyp_term_2
            - self.alpha * self.beta * F0 * self.X0
            - F0 * mu * self.X0
            + self.alpha * self.beta * F_inf * self.X0
            + F_inf * mu * self.X0
        )

        denominator = delta_F * expr_2
        result = -numerator / denominator
        return result

    def get_P(self, F_inf, mu, t):
        F0 = self.F0(mu)
        alpha = self.alpha
        beta = self.beta
        s_f = self.s_f
        Y_XS = self.Y_XS
        pi_0 = self.pi_0
        pi_1 = self.pi_1

        # define some common terms
        delta_F = F0 - F_inf
        exp_alpha_beta_neg_t = np.exp(-alpha * beta * t)
        exp_alpha_beta_t = np.exp(alpha * beta * t)
        exp_muf_t = np.exp(mu * t)
        exp_combined = np.exp(t * (alpha * beta + mu))

        # hypergeometric terms
        hyp_term_1 = scipy.special.hyp2f1(
            1,
            alpha * beta / mu + 1,
            alpha * beta / mu + 2,
            (exp_muf_t * F0) / delta_F,
        )
        hyp_term_2 = scipy.special.hyp2f1(
            1, alpha * beta / mu + 1, alpha * beta / mu + 2, F0 / delta_F
        )

        # logarithmic terms
        log_term_1 = np.log(F0 * (exp_muf_t - 1) + F_inf)
        log_term_2 = np.log(F_inf)

        # define a commonly used expressions for sake of conciseness below
        expr_1, expr_2 = self.common_expressions(mu)

        # numerator: first major term
        numerator_part1 = (
            -expr_1
            * mu
            * F_inf
            * (alpha * beta * pi_1 - pi_0)
            * exp_combined
            * hyp_term_1
            + F_inf * mu * expr_1 * (alpha * beta * pi_1 - pi_0) * hyp_term_2
        )

        # numerator: second major term
        numerator_part2 = -(delta_F * (expr_2)) * (
            -alpha * F_inf * pi_0 * s_f * Y_XS * exp_alpha_beta_t * log_term_1
            + alpha * F_inf * pi_0 * s_f * Y_XS * log_term_2 * exp_alpha_beta_t
            - mu * self.X0 * (exp_alpha_beta_t - 1) * (pi_0 - alpha * beta * pi_1)
        )

        numerator = exp_alpha_beta_neg_t * (numerator_part1 + numerator_part2)
        denominator = alpha * beta * mu * delta_F * (expr_2)
        return self.P0 + numerator / denominator

    def evaluate_at_t(self, t, mu, F_inf):
        if not util.is_iterable(t):
            t = [t]
        # make sure we got an array
        t = np.asarray(t)

        V = self.get_V(F_inf, mu, t)
        X = self.get_X(F_inf, mu, t)
        P = self.get_P(F_inf, mu, t)
        df = pd.DataFrame({"V": V, "X": X, "P": P}, index=t)
        df.index.name = "t"
        return df


class LinearStageAnalytical(FedBatchStageAnalytical, LinearFeed):
    def t_until_V(self, V, dF, F0):
        if V < self.V0:
            raise ValueError(f"`V` needs to be larger than `V0` (V0={self.V0}).")
        # use quadratic formula to solve for t
        return util.quadratic_formula(a=dF / 2, b=F0, c=self.V0 - V, plus_only=True)

    def evaluate_at_t(self, t, dF, F0):
        if not util.is_iterable(t):
            t = [t]
        # make sure we got an array
        t = np.asarray(t)

        X0 = self.X0
        sf = self.s_f
        Yxs = self.Y_XS
        Yas = self.Y_AS
        Yps = self.Y_PS
        rho = self.rho
        pi0 = self.pi_0
        pi1 = self.pi_1

        # calculate V
        V = self.V0 + t * F0 + dF * t**2 / 2

        # get a few common expressions
        term1 = (1 / Yxs) + pi1 / Yps
        term2 = (rho / Yas + pi0 / Yps) / term1
        exp_bt = np.exp(term2 * t)
        exp_neg_bt = np.exp(-term2 * t)
        term3 = (F0 / term2) * (exp_bt - 1)
        term4 = dF * ((exp_bt * (t * term2 - 1) + 1) / (term2**2))
        integral_term = (sf / term1) * (term3 + term4)

        # calculate X
        X = exp_neg_bt * (X0 + integral_term)

        term5 = (sf / (term1 * term2)) * (F0 - dF / term2)
        term6 = (sf * dF) / (term1 * term2)

        X_int = (
            term5 * t + 0.5 * term6 * t**2 + ((X0 - term5) / term2) * (1 - exp_neg_bt)
        )

        P = self.P0 + pi0 * X_int + pi1 * (X - X0)

        df = pd.DataFrame({"V": V, "X": X, "P": P}, index=t)
        df.index.name = "t"
        return df


class LinearStageConstantGrowthAnalytical(
    LinearStageAnalytical, LinearFeedConstantGrowth
):
    def evaluate_at_t(self, t, G):
        F0, dF = self.F0_and_dF_for_constant_growth(G)
        return super().evaluate_at_t(t, dF=dF, F0=F0)

    def t_until_V(self, V, G):
        F0, dF = self.F0_and_dF_for_constant_growth(G)
        return super().t_until_V(V, dF=dF, F0=F0)
