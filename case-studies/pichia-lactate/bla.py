# %% ###################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

# %% ###################################################################################
"""
we assume:
- mu_max_feed    = 0.15 /h
- F_max          = 0.01 L/h (they used 4 mL/h in the paper)

From the paper
- V_batch       = 320 mL
- V_max         = 350 mL
- s1_s_f        = 1260 g/L (density of pure glycerol)
- s2_s_f        = 2163 g/L (this is a dummy value corresponding to 64.9 g in 30 mL)

from the literature:
- doi: 10.1016/j.biortech.2008.01.059:
    - mu_max_phys (glycerol)                    = 0.23 /h
    - s1_Y_XS (glycerol)                        = 0.5 g/g
    - we don't need a yield for methanol since there is no growth in the second stage
- reasonable assumtion:
    - s1_Y_AS                                   = 18-20 mol ATP / mol glycerol
    - s2_Y_AS                                   = 0.44 mmol ATP / 2.81 mg methanol
        - this corresponds to 156.6 mmol ATP / g methanol
        - which corresponds to 79.4 g ATP / g methanol

- https://doi.org/10.1016/j.mec.2019.e00103
    - rho                                       =  0.55 mmol ATP / (g CDW h)
        - this is pretty low; perhaps we should use 1 mmol ATP / (g CDW h)
    - s1_Y_PS (glycerol -> lactate theoretical) = 0.978
    - s2_Y_PS (methanol -> lactate theoretical) = 0.9375


from the data:
- x_batch       = 24 g/L
- s1_pi_0
- s2_pi_0
- pi_1
"""
# %% ###################################################################################
from julsy.plots import plot_df
import sys

sys.path.append("../../FedBatchDesigner")
import process_stages

# %% ###################################################################################
# read data (extracted from Figure 5 of the paper)
df_orig = pd.read_csv("data-from-fig-7.csv", index_col=0)

plot_df(df_orig[["x [g/L]", "p [g/L]"]])
# %% ###################################################################################
df = df_orig.loc[18:].copy()
df.index = df.index - df.index[0]
# %% ###################################################################################
# define some params
V0 = 320 / 1000  # L
P0 = 0  # g lactate
X0 = df.loc[0, "x [g/L]"] * V0  # g

s1_s_f = 1260  # density of pure glycerol
s1_F = 4 / 1000  # L/h
s2_s_f = 2163  # dummy value for pure methanol

s1_Y_XS = 0.5  # g biomass / g glycerol

s1_Y_PS = 0.978  # g lactate / g glycerol
s2_Y_PS = 0.9375  # g lactate / g methanol

s1_Y_AS = 18 * 507 / 92.1  # g ATP / g glycerol
s2_Y_AS = 79.4  # g ATP / g methanol

rho = 0.55e-3 * 507  # g ATP / (g CDW h)
# %% ###################################################################################
# fit the growth stage
df_s1 = df.iloc[:2][["x [g/L]", "p [g/L]"]]


def predict_stage_1(s1_pi_0, s1_pi_1):
    stage = process_stages.ConstantStageAnalytical(
        V0=V0,
        X0=X0,
        P0=P0,
        pi_0=s1_pi_0,
        pi_1=s1_pi_1,
        s_f=s1_s_f,
        Y_XS=s1_Y_XS,
        Y_PS=s1_Y_PS,
        Y_AS=s1_Y_AS,
        rho=rho,
    )
    pred = stage.evaluate_at_t(t=df_s1.index, F=s1_F).eval("x = X/V").eval("p = P/V")
    return pred


def residuals_stage_1(params):
    s1_pi_0, s1_pi_1 = params
    pred = predict_stage_1(s1_pi_0, s1_pi_1)
    residuals = pred[["x", "p"]].values - df_s1.values
    return residuals.flatten()


res_s1 = scipy.optimize.least_squares(
    residuals_stage_1,
    x0=[1, 1],
    bounds=([0, 0], [np.inf, np.inf]),
    # method="trf",
)

fit_df_s1 = predict_stage_1(*res_s1.x)

fig, ax = plt.subplots()

df_s1.plot(ax=ax, marker="o", linestyle="-", color=["b", "r"])
fit_df_s1[["x", "p"]].rename(columns={"x": "x fit", "p": "titer fit"}).plot(
    ax=ax, marker="x", linestyle="--", color=["b", "r"]
)

# %% ###################################################################################
# fit the prodcution stage
df_s2 = df.iloc[1:][["x [g/L]", "p [g/L]"]]
df_s2.index -= df_s2.index[0]


def predict_production_stage(s2_pi_0, s2_pi_1):
    stage = process_stages.NoGrowthConstantStage(
        V0=fit_df_s1["V"].iloc[-1],
        X0=fit_df_s1["X"].iloc[-1],
        P0=fit_df_s1["P"].iloc[-1],
        pi_0=s2_pi_0,
        pi_1=s2_pi_1,
        s_f=s2_s_f,
        Y_XS=0,
        Y_PS=s2_Y_PS,
        Y_AS=s2_Y_AS,
        rho=rho,
    )
    pred = stage.evaluate_at_t(t=df_s2.index).eval("x = X/V").eval("p = P/V")
    return pred


def residuals_production_stage(params):
    s2_pi_0, s2_pi_1 = params
    pred = predict_production_stage(s2_pi_0, s2_pi_1)
    residuals = pred[["x", "p"]].values - df_s2.values
    return residuals.flatten()


res_s2 = scipy.optimize.least_squares(
    residuals_production_stage,
    x0=[0, 0],
    bounds=([0, 0], [np.inf, np.inf]),
    # method="trf",
)

fit_df_s2 = predict_production_stage(*res_s2.x)

print(f"res_s2: {res_s2.x}")
fig, ax = plt.subplots()
df_s2.plot(ax=ax, marker="o", linestyle="-", color=["b", "r"])
fit_df_s2[["x", "p"]].rename(columns={"x": "x fit", "p": "titer fit"}).plot(
    ax=ax, marker="x", linestyle="--", color=["b", "r"]
)
# %% ###################################################################################


def predict_both_stages(s1_pi_0, s1_pi_1, s2_pi_0, s2_pi_1):
    stage_1 = process_stages.ConstantStageAnalytical(
        V0=V0,
        X0=X0,
        P0=P0,
        pi_0=s1_pi_0,
        pi_1=s1_pi_1,
        s_f=s1_s_f,
        Y_XS=0.4,
        Y_PS=s1_Y_PS,
        Y_AS=s1_Y_AS,
        rho=rho,
    )
    pred_s1 = (
        stage_1.evaluate_at_t(t=df_s1.index, F=s1_F).eval("x = X/V").eval("p = P/V")
    )
    stage_2 = process_stages.NoGrowthConstantStage(
        V0=pred_s1["V"].iloc[-1],
        X0=pred_s1["X"].iloc[-1],
        P0=pred_s1["P"].iloc[-1],
        pi_0=s2_pi_0,
        pi_1=s2_pi_1,
        s_f=s2_s_f,
        Y_XS=0,
        Y_PS=s2_Y_PS,
        Y_AS=s2_Y_AS,
        rho=rho,
    )
    pred_s2 = stage_2.evaluate_at_t(t=df_s2.index).eval("x = X/V").eval("p = P/V")
    # the first index of pred_s2 is the last index of pred_s1 (and can be skipped)
    pred_s2 = pred_s2.iloc[1:]
    pred_s2.index += df_s1.index[-1]
    return pd.concat([pred_s1, pred_s2], axis=0)


def residuals_both_stages(params):
    s1_pi_0, s1_pi_1, s2_pi_0, s2_pi_1 = params
    pred = predict_both_stages(s1_pi_0, s1_pi_1, s2_pi_0, s2_pi_1)
    residuals = pred[["x", "p"]].values - df[["x [g/L]", "p [g/L]"]].values
    return residuals.flatten()


s1_pi_1_zero = True
s2_pi_1_zero = False

res_both_stages = scipy.optimize.least_squares(
    residuals_both_stages,
    # x0=[*res_s1.x, *res_s2.x],
    x0=[0, 0, 0.11, 0],
    bounds=(
        [0, 0, 0.00, 0],
        # [0] * 4,
        [
            # 0.05,
            np.inf,
            1e-9 if s1_pi_1_zero else np.inf,
            np.inf,
            1e-9 if s2_pi_1_zero else np.inf,
        ],
    ),
    # method="trf",
)
fit_df_both_stages = predict_both_stages(*res_both_stages.x)

fig, ax = plt.subplots()

df[["x [g/L]", "p [g/L]"]].plot(ax=ax, marker="o", linestyle="-", color=["b", "r"])
fit_df_both_stages[["x", "p"]].rename(columns={"x": "x fit", "p": "titer fit"}).plot(
    ax=ax, marker="x", linestyle="--", color=["b", "r"]
)

print(f"cost = {res_both_stages.cost}")

fitted_params = dict(
    zip(
        ["s1_pi_0", "s1_pi_1", "s2_pi_0", "s2_pi_1"],
        res_both_stages.x,
    )
)
for k, v in fitted_params.items():
    print(f"{k}: {v:.3g}")
# %% ###################################################################################
# fit_params_unconstr = res_both_stages.x.copy()

# print((1 / 2 * residuals_both_stages(fit_params_unconstr) ** 2).sum())
# res_both_stages.cost
# # %% ###################################################################################
# # fit_params_constr = res_both_stages.x.copy()

# print((1 / 2 * residuals_both_stages(fit_params_constr) ** 2).sum())
# res_both_stages.cost
# # %% ###################################################################################
# plt.plot(df.index, residuals_both_stages(fit_params_unconstr).reshape(-1, 2))
# # %% ###################################################################################
# plt.plot(df.index, residuals_both_stages(fit_params_constr).reshape(-1, 2))
# # %% ###################################################################################
# plot_df(predict_both_stages(*fit_params_constr)[["x", "p"]])
# plot_df(predict_both_stages(*fit_params_unconstr)[["x", "p"]])

# df[["x [g/L]", "p [g/L]"]]
# # %% ###################################################################################
# predict_both_stages(*fit_params_constr)[["x", "p"]]

# # %% ###################################################################################
# df.eval("X = `x [g/L]` * `V [mL]` / 1000").eval("P = `p [g/L]` * `V [mL]` / 1000")