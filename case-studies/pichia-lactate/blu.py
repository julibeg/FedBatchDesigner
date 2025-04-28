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
# interpolate batch (exponential growth)
mu_batch = df_orig
# %% ###################################################################################
df = df_orig.loc[:100].copy()
df.columns = ["V_pre_sample", "x", "p", "sample_volume"]
df[["V_pre_sample", "sample_volume"]] /= 1000
df = df.eval("V_post_sample = V_pre_sample - sample_volume")
df
# %% ###################################################################################


def predict_batch(pi_0, pi_1, t, X0, P0, mu):
    def ODEs(t, y):
        X, _P = y
        dXdt = mu * X
        dPdt = X * (pi_0 + pi_1 * mu)
        return dXdt, dPdt

    res = scipy.integrate.solve_ivp(
        ODEs,
        t_span=(t[0], t[-1]),
        y0=(X0, P0),
        t_eval=t,
    )
    res_df = pd.DataFrame(res.y.T, columns=["X", "P"], index=t)
    res_df.index.name = "t"
    return res_df


X_pre_batch = df.iloc[0]["x"] * df.iloc[0]["V_post_sample"]
X_post_batch = df.iloc[1]["x"] * df.iloc[1]["V_pre_sample"]

delta_P_batch = df.iloc[1]["p"] * df.iloc[1]["V_pre_sample"] - 0

batch_duration = df.index[1] - df.index[0]
mu_batch = np.log(X_post_batch / X_pre_batch) / batch_duration
t_batch = np.linspace(0, batch_duration, 101)
predict_batch(0.1, 0.1, t_batch, X_pre_batch, 0, mu_batch)
# %% ###################################################################################


def predict_linear(pi_0, pi_1, t, X0, P0, k):
    def ODEs(t, y):
        X, _P = y
        dXdt = k
        mu = k / X
        dPdt = X * (pi_0 + pi_1 * mu)
        return dXdt, dPdt

    res = scipy.integrate.solve_ivp(
        ODEs,
        t_span=(t[0], t[-1]),
        y0=(X0, P0),
        t_eval=t,
    )

    res_df = pd.DataFrame(res.y.T, columns=["X", "P"], index=t)
    res_df.index.name = "t"
    return res_df


X_pre_feed = df.iloc[1]["x"] * df.iloc[1]["V_post_sample"]
X_post_feed = df.iloc[2]["x"] * df.iloc[2]["V_pre_sample"]

delta_P_feed = (
    df.iloc[2]["p"] * df.iloc[2]["V_pre_sample"]
    - df.iloc[1]["p"] * df.iloc[1]["V_post_sample"]
)

# linear feed for approx. 4 hours -> assume linear growth
feed_duration = df.index[2] - df.index[1]
k = (X_post_feed - X_pre_feed) / feed_duration
t_feed = np.linspace(0, feed_duration, 101)

plot_df(predict_linear(0.1, 0.1, t_feed, X_pre_feed, 0, k))
# %% ###################################################################################
# fit pi_0 and pi_1 for glycerol (i.e. the batch and stage 1)


def residuals(params):
    pi_0, pi_1 = params
    pred_batch = predict_batch(pi_0, pi_1, t_batch, X_pre_batch, 0, mu_batch)
    pred_feed = predict_linear(
        pi_0, pi_1, t_feed, X_pre_feed, pred_batch["P"].iloc[-1], k
    )
    return np.array(
        [
            delta_P_batch - (pred_batch["P"].iloc[-1] - pred_batch["P"].iloc[0]),
            delta_P_feed - (pred_feed["P"].iloc[-1] - pred_feed["P"].iloc[0]),
        ]
    )


res = scipy.optimize.least_squares(
    residuals,
    x0=[0, 0],
    bounds=([0, 0], [np.inf, np.inf]),
    # method="trf",
)
s1_pi_0, s1_pi_1 = res.x

# %% ###################################################################################
predict_batch(s1_pi_0, s1_pi_1, t_batch, X_pre_batch, 0, mu_batch)
# %% ###################################################################################
predict_linear(s1_pi_0, s1_pi_1, t_feed, X_pre_feed, 0, k)
# %% ###################################################################################
# analytical solution for the batch
X_batch_int = (X_pre_batch * np.exp(mu_batch * batch_duration) - X_pre_batch) / mu_batch
X_batch_int * (s1_pi_0 + s1_pi_1 * mu_batch)
# %% ###################################################################################
# analytical solution for the feed
X_feed_int = np.mean(X_post_feed - X_pre_feed) * feed_duration
X_feed_int * s1_pi_0 + k * s1_pi_1 * feed_duration

# %% ###################################################################################
# for the second stage the total volume doesn't change much and there is no growth.
# Therefore we simply take the average biomass concentration and calculate delta P that
# way
df_s2 = df.loc[22:]
delta_P_s2 = df_s2["p"].values[1:] - df_s2["p"].values[:-1]
x_mean_s2 = df_s2['x'].rolling(window=2).mean().values[1:]
delta_t_s2 = df_s2.index[1:] - df_s2.index[:-1]

pi_0_s2 = delta_P_s2 / (x_mean_s2 * delta_t_s2)

# %% ###################################################################################