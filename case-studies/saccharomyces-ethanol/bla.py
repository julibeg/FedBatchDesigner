# %% ###################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate

from julsy.plots import plot_df

# %% ###################################################################################
f2_df_all = pd.read_csv("fig2.csv", index_col=0)
f2_df = f2_df_all.loc[:7].copy()
f2_df.columns = ["x", "p", "s"]
# convert mM to g/L
f2_df["s"] = f2_df["s"] / 1000 * 180
f2_df["p"] = f2_df["p"] / 1000 * 46
f2_df

fig = plot_df(f2_df, return_fig=True)

for ax in fig.axes:
    ax.set_ylim(0, ax.get_ylim()[1])
# %% ###################################################################################
# we try to fit exponential growth to the data but see that there is a lag phase
f2_t = f2_df.index.values
f2_t_interp = np.linspace(f2_df.index[0], f2_df.index[-1], 101)


def fit_x_exp(t, x0, mu):
    return x0 * np.exp(mu * t)


(x0_fit, mu_fit), _ = scipy.optimize.curve_fit(
    fit_x_exp,
    f2_t,
    f2_df["x"],
    p0=[0.3, 0.1],
)

x_fit = fit_x_exp(f2_t_interp, x0_fit, mu_fit)

plt.plot(f2_t, f2_df["x"], "ko--")
plt.plot(f2_t_interp, x_fit, "r--")
# %% ###################################################################################
# therefore, we simply interpolate growth with cubic splines
f2_x_cs = scipy.interpolate.CubicSpline(f2_df.index, f2_df["x"], bc_type="natural")
f2_x_cs_d1 = f2_x_cs.derivative(1)


# Calculate the interpolated y values using the spline function
f2_x_interp = f2_x_cs(f2_t_interp)

plt.plot(f2_df.index, f2_df["x"], "ko--")
plt.plot(f2_t_interp, f2_x_interp, "r--")
# %% ###################################################################################
# now that we got ways to interpolate x and mu, use them to integrate the ODE for p in
# order to fit pi_0 and pi_1 (dP/dt = x (pi_0 + mu * pi_1))


def predict_p(t, pi_0, pi_1, p0=0):
    def ODE(t, _p):
        x = f2_x_cs(t)
        mu = f2_x_cs_d1(t) / x
        return x * (pi_0 + mu * pi_1)

    # solve the ODE
    res = scipy.integrate.solve_ivp(
        fun=ODE,
        t_span=(t[0], t[-1]),
        y0=[p0],
        t_eval=t,
    )
    return pd.Series(res.y[0], index=res.t, name="EtOH [g/L]")


def get_residuals(params):
    pi_0, pi_1 = params
    p_pred = predict_p(f2_t, pi_0, pi_1)
    return f2_df.loc[f2_t, "p"] - p_pred


res = scipy.optimize.least_squares(
    get_residuals,
    x0=[0.1, 0.1],
    bounds=([0, 0], [np.inf, np.inf]),
)

f2_pi_0, f2_pi_1 = res.x
print(f"{f2_pi_0=}, {f2_pi_1=}")
# %% ###################################################################################
# check the fit of product concentration visually
plt.plot(f2_df.index, f2_df["p"], "ko--")
plt.plot(f2_t_interp, predict_p(f2_t_interp, *res.x), "r--")
# %% ###################################################################################
# they reported a r_EtOH for the exponential phase between 2.5 and 7 hours in Table 2
# (28.8 mmol/(g CDW h)). We can use our `pi` values and the `mu` from the table (0.235
# /h) to see if we get a similar value
r_Eth_table = 28.8 / 1000 * 46
r_Eth_ours = f2_pi_0 + 0.235 * f2_pi_1
print(f"{r_Eth_table=:.3f}, {r_Eth_ours=:.3f}")
# %% ###################################################################################
