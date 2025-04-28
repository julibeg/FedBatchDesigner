# %% ###################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

# %% ###################################################################################
"""
we assume:
- mu_max_feed    = 0.15 /h
- F_max          = 0.5 L/h

From the paper
- V_batch       = 2L
- V_max         = 5L
- s_f           = 400 g/L
    - (looks like they used a small constant feed with pure glycerol in the second
      stage; the density of glycerol is 1260 g/L)

from the literature:
- doi: 10.1016/j.biortech.2008.01.059:
    - mu_max_phys (glycerol)                                        = 0.23 /h
    - Y_XS (glycerol)                                               = 0.55 g/g
- reasonable assumtion:
    - Y_AS                                                          = 18-20 mol ATP / mol glycerol
- https://doi.org/10.1016/j.mec.2019.e00103
    - rho                                                           = 0.55 mmol ATP / (g CDW h)
        - this is pretty low; perhaps we should use 1 mmol ATP / (g CDW h)
    - Y_PS (theoretical; need to calculate for this substrate)      = 0.73 g/g
        - molecular weight of CalB: 35511 Da
        - # of carbon atoms in CalB: 1584
        - if all the carbon atoms in CalB are from glycerol, then we need
            1584 / 3 = 528 glycerol molecules (which is 528 * 92.09 g/mol = 48.6 kg).
            Therefore, we need 48.6 kg of glycerol to produce 35.5 kg of CalB
            -> Y_PS = 35.5 / 48.6 = 0.73 g/g

from the data:
- x_batch       = 24 g/L
- s1_pi_0
- s2_pi_0
- pi_1
"""
# %% ###################################################################################
mu = 0.025
t = np.linspace(0, 60, 101)
x0 = 22  # g/L
V0 = 2  # L
X = x0 * V0 * np.exp(mu * t)

F0 = 0.004
mu_f = mu
V = V0 + F0 * np.exp(mu_f * t) / mu_f
x = X / V

print(x[-1])
print(f"Yxs = {(X[-1] - X[0]) / 0.4 / (V[-1] - V0)}")

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(t, V, "k--")
ax2.plot(t, x, "r-")

# %% ###################################################################################
import sys

sys.path.append("../../FedBatchDesigner")
import process_stages

# %% ###################################################################################
# read data (extracted from Figure 5 of the paper)
df = pd.read_csv("data-from-fig5-PS.csv", index_col=0)

df.plot(marker="o", linestyle="--", color=["b", "r"])
# %% ###################################################################################
# define some params
V0 = 2  # L
mu = 0.15  # /h
P0 = 0  # kAU
X0 = df.loc[0, "x [g/L]"] * V0  # g
s1_s_f = 400  # g/L
# s2_s_f = 1260  # density of pure glycerol
s2_s_f = s1_s_f
Y_XS = 0.55  # g biomass / g glycerol
# Y_PS = 0.73  # g product / g glycerol
Y_AS = 18 * 507 / 92.1  # g ATP / g glycerol
rho = 0.55e-3 * 507  # g ATP / (g CDW h)
# %% ###################################################################################
# fit the growth stage
df_s1 = df.iloc[0:3]


def predict_stage_1(Y_PS, s1_pi_0, s1_pi_1):
    stage = process_stages.ExponentialStageAnalytical(
        V0=V0,
        X0=X0,
        P0=P0,
        pi_0=s1_pi_0,
        pi_1=s1_pi_1,
        s_f=s1_s_f,
        Y_XS=Y_XS,
        Y_PS=Y_PS,
        Y_AS=Y_AS,
        rho=rho,
    )
    pred = stage.evaluate_at_t(t=df_s1.index, mu=mu).eval("x = X/V")
    return pred


def residuals_stage_1(params):
    Y_PS, s1_pi_0, s1_pi_1 = params
    pred = predict_stage_1(Y_PS, s1_pi_0, s1_pi_1)
    residuals = pred[["x", "P"]].values - df_s1.values
    return residuals.flatten()


res_s1 = scipy.optimize.least_squares(
    residuals_stage_1,
    x0=[1, 0, 0],
    bounds=([1, 0, 0], [np.inf, np.inf, np.inf]),
    # method="trf",
)

fit_df_s1 = predict_stage_1(*res_s1.x)

fig, ax = plt.subplots()

df_s1.plot(ax=ax, marker="o", linestyle="-", color=["b", "r"])
fit_df_s1[["x", "P"]].rename(columns={"x": "x fit", "P": "titer fit"}).plot(
    ax=ax, marker="x", linestyle="--", color=["b", "r"]
)

# %% ###################################################################################
# fit the prodcution stage
df_s2 = df.iloc[2:]
df_s2.index -= df_s2.index[0]


def predict_production_stage(Y_PS, s2_pi_0, s2_pi_1, F):
    stage = process_stages.ConstantStageAnalytical(
        V0=fit_df_s1["V"].iloc[-1],
        X0=fit_df_s1["X"].iloc[-1],
        P0=fit_df_s1["P"].iloc[-1],
        pi_0=s2_pi_0,
        pi_1=s2_pi_1,
        s_f=1260,
        Y_XS=Y_XS,
        Y_PS=Y_PS,
        Y_AS=Y_AS,
        rho=rho,
    )
    assert F > stage.F_min
    pred = stage.evaluate_at_t(t=df_s2.index, F=F).eval("x = X/V")
    return pred


def residuals_production_stage(params):
    Y_PS, s2_pi_0, s2_pi_1, F = params
    pred = predict_production_stage(Y_PS, s2_pi_0, s2_pi_1, F)
    residuals = pred[["x", "P"]].values - df_s2.values
    return residuals.flatten()


res_s2 = scipy.optimize.least_squares(
    residuals_production_stage,
    x0=[1, 0, 0, 1],
    bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
    # method="trf",
)

fit_df_s2 = predict_production_stage(*res_s2.x)

fig, ax = plt.subplots()
df_s2.plot(ax=ax, marker="o", linestyle="-", color=["b", "r"])
fit_df_s2[["x", "P"]].rename(columns={"x": "x fit", "P": "titer fit"}).plot(
    ax=ax, marker="x", linestyle="--", color=["b", "r"]
)
# %% ###################################################################################


def predict_both_stages(Y_PS, s1_pi_0, s1_pi_1, s2_pi_0, s2_pi_1, s2_F):
    stage_1 = process_stages.ExponentialStageAnalytical(
        V0=V0,
        X0=X0,
        P0=P0,
        pi_0=s1_pi_0,
        pi_1=s1_pi_1,
        s_f=s1_s_f,
        Y_XS=Y_XS,
        Y_PS=Y_PS,
        Y_AS=Y_AS,
        rho=rho,
    )
    pred_s1 = stage_1.evaluate_at_t(t=df_s1.index, mu=mu).eval("x = X/V")
    stage_2 = process_stages.ConstantStageAnalytical(
        V0=pred_s1["V"].iloc[-1],
        X0=pred_s1["X"].iloc[-1],
        P0=pred_s1["P"].iloc[-1],
        pi_0=s2_pi_0,
        pi_1=s2_pi_1,
        s_f=s2_s_f,
        Y_XS=Y_XS,
        Y_PS=Y_PS,
        Y_AS=Y_AS,
        rho=rho,
    )
    assert s2_F > stage_2.F_min
    pred_s2 = stage_2.evaluate_at_t(t=df_s2.index, F=s2_F).eval("x = X/V")
    # the first index of pred_s2 is the last index of pred_s1 (and can be skipped)
    pred_s2 = pred_s2.iloc[1:]
    pred_s2.index += df_s1.index[-1]
    return pd.concat([pred_s1, pred_s2], axis=0)


def residuals_both_stages(params):
    Y_PS, s1_pi_0, s1_pi_1, s2_pi_0, s2_pi_1, s2_F = params
    pred = predict_both_stages(Y_PS, s1_pi_0, s1_pi_1, s2_pi_0, s2_pi_1, s2_F)
    residuals = pred[["x", "P"]].values - df.values
    return residuals.flatten()

s1_pi_1_zero = False
s2_pi_1_zero = True

res_both_stages = scipy.optimize.least_squares(
    residuals_both_stages,
    x0=[1, 0, 0, 0, 0, 0.1],
    bounds=(
        [0, 0, 0, 0, 0, 0],
        [
            np.inf,
            np.inf,
            1e-9 if s1_pi_1_zero else np.inf,
            np.inf,
            1e-9 if s2_pi_1_zero else np.inf,
            np.inf,
        ],
    ),
    # method="trf",
)
fit_df_both_stages = predict_both_stages(*res_both_stages.x)

fig, ax = plt.subplots()

df.plot(ax=ax, marker="o", linestyle="-", color=["b", "r"])
fit_df_both_stages[["x", "P"]].rename(columns={"x": "x fit", "P": "titer fit"}).plot(
    ax=ax, marker="x", linestyle="--", color=["b", "r"]
)

fitted_params = dict(
    zip(
        ["Y_PS", "s1_pi_0", "s1_pi_1", "s2_pi_0", "s2_pi_1", "F_s2"],
        res_both_stages.x,
    )
)
for k, v in fitted_params.items():
    print(f"{k}: {v:.3g}")

# %% ###################################################################################
