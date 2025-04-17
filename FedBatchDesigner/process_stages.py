import abc

import numpy as np
import pandas as pd
import scipy

from logger import logger
import util

# maximum step size (in hours of time domain) for the integration
MAX_INTEGRATION_STEP = 1


class NotEnoughGlucoseError(Exception):
    pass


class FedBatchStage(abc.ABC):
    def __init__(self, V0, X0, P0, s_f, Y_AS, Y_PS, Y_XS, rho, pi_0, pi_1, debug=False):
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
        self.debug = debug
        # calculate glucose required for maintenance and product formation at t=0
        self.initial_glc_for_rho_and_pi_0 = X0 * (rho / Y_AS + pi_0 / Y_PS)

    @abc.abstractmethod
    def evaluate_at_V(self, Vs, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_at_t(self, t, **kwargs):
        raise NotImplementedError


# define mixin classes to add some functionality to exponential vs constant feed stages
class ConstantFeed:
    @property
    def F_min(self):
        """
        Calculate the smallest possible constant feed rate.

        With any feed rate smaller than this, not enough glucose for maintenance and
        product formation would be added in the first instance of the batch.
        """
        return self.initial_glc_for_rho_and_pi_0 / self.s_f

    def calculate_initial_mu(self, F):
        """Calculate the growth rate at the first instance of constant feed."""
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


class LinearFeed(ConstantFeed):
    # for now no difference to constant stage
    pass


class ExponentialFeed:
    def substrate_start_volume(self, mu):
        return (
            self.X0
            / (self.s_f * mu)
            * (
                mu / self.Y_XS
                + self.rho / self.Y_AS
                + (self.pi_0 + mu * self.pi_1) / self.Y_PS
            )
        )

    def phi_0(self, mu):
        # this `phi_0` is the feed rate that satisfies `dX / dt = X * mu` in the first
        # instance of the fed-batch
        return self.substrate_start_volume(mu) * mu


class LogisticFeed(ExponentialFeed):
    # for now no difference to exponential stage
    pass


class NoGrowthConstantStage(FedBatchStage, ConstantFeed):
    """
    Process stage with constant feed rate and `mu = 0`.

    The feed rate is such that the added substrate exactly covers the amount required
    for maintenance and product formation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = self.initial_glc_for_rho_and_pi_0 / self.s_f
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
        # make sure `Vs` is an iterable
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

    # define the maximum time to integrate; this should never be reached as we terminate the
    # integration when a condition is met
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

            # get dV and the amount of glucose thus added
            dV = self.dV(t=t, **dV_kwargs)
            glc_add = dV * self.s_f

            # get glucose needed for maintenance and for non-growth-dependent production
            glc_mnt = X * self.rho / self.Y_AS
            dP_pi_0 = X * self.pi_0
            glc_P_pi_0 = dP_pi_0 / self.Y_PS

            # maintenance could require more glucose than was added; make sure to not
            # have negative glucose in that case
            if glc_add < glc_mnt:
                # not enough glucose for maintenance
                if allow_not_enough_for_rho:
                    glc = 0
                else:
                    raise NotEnoughGlucoseError(
                        "More glucose required by maintenance than was added.\n"
                        f"{t=:.5g}, {V=:.5g}, {X=:.5g}, {P=:.5g}, "
                        f"{glc_add=:.5g}, {glc_mnt=:.5g}"
                    )
            else:
                # there's enough glucose for maintenance
                glc = glc_add - glc_mnt
            # make sure that there is enough glucose for product formation
            if glc < glc_P_pi_0:
                if allow_not_enough_for_pi:
                    # use all the remaining glucose for product formation (this will be
                    # less than `X * pi_0`)
                    dP_pi_0 = glc * self.Y_PS
                    glc = 0
                else:
                    raise NotEnoughGlucoseError(
                        "Product formation requires more glucose than was added.\n"
                        f"{t=:.5g}, {V=:.5g}, {X=:.5g}, {P=:.5g}, "
                        f"{glc=:.5g}, {glc_P_pi_0=:.5g}"
                    )
            else:
                glc -= glc_P_pi_0
            # use the remaining glucose for growth and growth-coupled production
            dX = glc / (1 / self.Y_XS + self.pi_1 / self.Y_PS)
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

    def evaluate_at_V(self, Vs, **dV_kwargs):
        # `dV_kwargs` is passed to `dV`, which needs to be implemented by children. The
        # rest of the functionality doesn't need to change depending on the process
        # stage type. make sure `Vs` is an iterable
        if not util.is_iterable(Vs):
            Vs = [Vs]

        if self.debug:
            debug_df = pd.DataFrame(columns=["V", "X", "P"])
        else:
            debug_df = None

        # get list of event functions (one for each V_interval)
        events = [lambda _t, state, v=v: state[0] - v for v in Vs]
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
        except NotEnoughGlucoseError as e:
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
    def dV(self, F, t):
        return F


class LinearStageIntegrate(FedBatchStageIntegrate, LinearFeed):
    def dV(self, k, t):
        dV_min = self.initial_glc_for_rho_and_pi_0 / self.s_f
        return k * t + dV_min


class ExponentialStageIntegrate(FedBatchStageIntegrate, ExponentialFeed):
    def dV(self, mu, t):
        return self.substrate_start_volume(mu) * mu * np.exp(mu * t)


class LogisticStageIntegrate(FedBatchStageIntegrate, LogisticFeed):
    def dV(self, phi_inf, mu, t):
        return phi_inf / (1 + np.exp(-mu * t) * (phi_inf / self.phi_0(mu) - 1))


class FedBatchStageAnalytical(FedBatchStage):
    """
    Base class for process stages using analytical expressions for calculating
    process trajectories.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pi_0, pi_1, rho, Y_XS, Y_PS, Y_AS = (
            self.pi_0,
            self.pi_1,
            self.rho,
            self.Y_XS,
            self.Y_PS,
            self.Y_AS,
        )
        self.alpha = 1 / (1 + pi_1 * Y_XS / Y_PS)
        self.beta = pi_0 * Y_XS / Y_PS + rho * Y_XS / Y_AS

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
        ts = self.t_until_V(V, **kwargs)
        return self.evaluate_at_t(ts, **kwargs)


class ConstantStageAnalytical(FedBatchStageAnalytical, ConstantFeed):
    def t_until_V(self, V, F):
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
        P = (pi_0 / self.beta) * F * s_f * Y_XS * t + (
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
        expr_1 = self.phi_0(mu) * self.s_f * self.Y_XS * self.alpha
        expr_2 = self.alpha * self.beta + mu
        return expr_1, expr_2

    def t_until_V(self, V, mu):
        phi_0 = self.phi_0(mu)
        V_rest = V - self.V0
        return np.log(V_rest * mu / phi_0 + 1) / mu

    def evaluate_at_t(self, t, mu):
        if not util.is_iterable(t):
            t = [t]
        # make sure we got an array
        t = np.asarray(t)

        phi_0 = self.phi_0(mu)
        V = self.V0 + (np.exp(mu * t) - 1) * phi_0 / mu
        # define a few commonly used expressions for sake of conciseness below
        expr_1, expr_2 = self.common_expressions(mu)
        X = (
            np.exp(-t * self.alpha * self.beta)
            * ((-1 + np.exp(t * expr_2)) * expr_1 + self.X0 * expr_2)
            / expr_2
        )
        P = (
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
    def t_until_V(self, V, mu, phi_inf):
        phi_0 = self.phi_0(mu)
        t = (
            np.log(
                (
                    phi_0
                    - phi_inf
                    + np.exp(
                        (V * mu - self.V0 * mu + np.log(phi_inf**phi_inf)) / phi_inf
                    )
                )
                / phi_0
            )
            / mu
        )
        return t

    def get_V(self, phi_inf, mu, t):
        phi_0 = self.phi_0(mu)
        V = (
            phi_inf * np.log(phi_0 * (np.exp(mu * t) - 1) + phi_inf)
            - phi_inf * np.log(phi_inf)
            + mu * self.V0
        ) / mu
        return V

    def get_X(self, phi_inf, mu, t):
        phi_0 = self.phi_0(mu)
        # define some common terms
        exp_mu_t = np.exp(mu * t)
        exp_alpha_beta_t = np.exp(self.alpha * self.beta * t)
        exp_alpha_beta_neg_t = np.exp(-self.alpha * self.beta * t)
        delta_F = phi_0 - phi_inf
        hyp_term_1 = scipy.special.hyp2f1(
            1,
            self.alpha * self.beta / mu + 1,
            self.alpha * self.beta / mu + 2,
            (exp_mu_t * phi_0) / delta_F,
        )
        hyp_term_2 = scipy.special.hyp2f1(
            1,
            self.alpha * self.beta / mu + 1,
            self.alpha * self.beta / mu + 2,
            phi_0 / delta_F,
        )

        # define a few commonly used expressions for sake of conciseness below
        expr_1, expr_2 = self.common_expressions(mu)

        numerator = exp_alpha_beta_neg_t * (
            expr_1 * phi_inf * exp_alpha_beta_t * exp_mu_t * hyp_term_1
            - expr_1 * phi_inf * hyp_term_2
            - self.alpha * self.beta * phi_0 * self.X0
            - phi_0 * mu * self.X0
            + self.alpha * self.beta * phi_inf * self.X0
            + phi_inf * mu * self.X0
        )

        denominator = delta_F * expr_2
        result = -numerator / denominator
        return result

    def get_P(self, phi_inf, mu, t):
        phi_0 = self.phi_0(mu)
        alpha = self.alpha
        beta = self.beta
        s_f = self.s_f
        Y_XS = self.Y_XS
        pi_0 = self.pi_0
        pi_1 = self.pi_1

        # define some common terms
        delta_F = phi_0 - phi_inf
        exp_alpha_beta_neg_t = np.exp(-alpha * beta * t)
        exp_alpha_beta_t = np.exp(alpha * beta * t)
        exp_muf_t = np.exp(mu * t)
        exp_combined = np.exp(t * (alpha * beta + mu))

        # hypergeometric terms
        hyp_term_1 = scipy.special.hyp2f1(
            1,
            alpha * beta / mu + 1,
            alpha * beta / mu + 2,
            (exp_muf_t * phi_0) / delta_F,
        )
        hyp_term_2 = scipy.special.hyp2f1(
            1, alpha * beta / mu + 1, alpha * beta / mu + 2, phi_0 / delta_F
        )

        # logarithmic terms
        log_term_1 = np.log(phi_0 * (exp_muf_t - 1) + phi_inf)
        log_term_2 = np.log(phi_inf)

        # define a commonly used expressions for sake of conciseness below
        expr_1, expr_2 = self.common_expressions(mu)

        # numerator: first major term
        numerator_part1 = (
            -expr_1
            * mu
            * phi_inf
            * (alpha * beta * pi_1 - pi_0)
            * exp_combined
            * hyp_term_1
            + phi_inf * mu * expr_1 * (alpha * beta * pi_1 - pi_0) * hyp_term_2
        )

        # numerator: second major term
        numerator_part2 = -(delta_F * (expr_2)) * (
            -alpha * phi_inf * pi_0 * s_f * Y_XS * exp_alpha_beta_t * log_term_1
            + alpha * phi_inf * pi_0 * s_f * Y_XS * log_term_2 * exp_alpha_beta_t
            - mu * self.X0 * (exp_alpha_beta_t - 1) * (pi_0 - alpha * beta * pi_1)
        )

        numerator = exp_alpha_beta_neg_t * (numerator_part1 + numerator_part2)

        denominator = alpha * beta * mu * delta_F * (expr_2)

        result = numerator / denominator
        return result

    def evaluate_at_t(self, t, mu, phi_inf):
        if not util.is_iterable(t):
            t = [t]
        # make sure we got an array
        t = np.asarray(t)

        V = self.get_V(phi_inf, mu, t)
        X = self.get_X(phi_inf, mu, t)
        P = self.get_P(phi_inf, mu, t)
        df = pd.DataFrame({"V": V, "X": X, "P": P}, index=t)
        df.index.name = "t"
        return df


class LinearGrowthStageAnalytical(FedBatchStageAnalytical, LinearFeed):
    def F0(self, k):
        return self.X0 * self.beta / self.s_f / self.Y_XS + k / self.alpha / self.beta

    def t_until_V(self, V, dF):
        F0 = self.F0(dF)
        # use quadratic formula to solve for t
        return util.quadratic_formula(a=dF / 2, b=F0, c=self.V0 - V, plus_only=True)

    def evaluate_at_t(self, t, dF):
        if not util.is_iterable(t):
            t = [t]
        # make sure we got an array
        t = np.asasarray(t)

        # define a few commonly used expresions for sake of conciseness
        X0 = self.X0
        Y_AS = self.Y_AS
        Y_PS = self.Y_PS
        Y_XS = self.Y_XS
        rho = self.rho
        s_f = self.s_f
        pi_0 = self.pi_0
        pi_1 = self.pi_1

        # G is constant; we can calculate it at t=0
        F0 = self.F0(dF)
        mu0 = (
            Y_XS
            * (F0 * Y_AS * Y_PS * s_f - X0 * Y_AS * pi_0 - X0 * Y_PS * rho)
            / (X0 * Y_AS * (Y_PS + Y_XS * pi_1))
        )
        G = X0 * mu0

        V = self.V0 + t * F0 + dF * t**2 / 2

        X = (
            Y_AS
            * (
                F0 * Y_PS * Y_XS * s_f
                - G * Y_PS
                - G * Y_XS * pi_1
                + Y_PS * Y_XS * dF * s_f * t
            )
            / (Y_XS * (Y_AS * pi_0 + Y_PS * rho))
        )

        P = (
            Y_PS
            * t
            * (
                2 * F0 * Y_AS * Y_XS * pi_0 * s_f
                - 2 * G * Y_AS * pi_0
                + 2 * G * Y_XS * pi_1 * rho
                + Y_AS * Y_XS * dF * pi_0 * s_f * t
            )
            / (2 * Y_XS * (Y_AS * pi_0 + Y_PS * rho))
        )

        df = pd.DataFrame({"V": V, "X": X, "P": P}, index=t)
        df.index.name = "t"
        return df
