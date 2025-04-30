# %% ###################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate

from julsy.plots import plot_df

# %% ###################################################################################
f2_x = pd.read_csv("fig2-bm.csv", index_col=0).squeeze()
f2_p = pd.read_csv("fig2-p.csv", index_col=0).squeeze()
f2_s = pd.read_csv("fig2-glc.csv", index_col=0).squeeze()
f2_p.index = f2_x.index
f2_s.index = f2_x.index

# convert mM to g/L
f2_s = f2_s * 180 / 1000
f2_s.name = "g [g/L]"
f2_p = f2_p * 46 / 1000
f2_p.name = "EtOH [g/L]"
# %% ###################################################################################
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(f2_x, "ko--")
ax2.plot(f2_p, "ro--")
ax2.plot(f2_s, "bo--")
# %% ###################################################################################
f2_df_all = pd.DataFrame({"x": f2_x, "p": f2_p, "s": f2_s})
f2_df = f2_df_all.loc[:7]

fig = plot_df(f2_df, return_fig=True)

for ax in fig.axes:
    ax.set_ylim(0, ax.get_ylim()[1])
# %% ###################################################################################
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
# fit logistic growth to the data


def fit_logistic(t, L, k, x0):
    return L / (1 + np.exp(-k * (t - x0)))


(L_fit, k_fit, x0_fit), _ = scipy.optimize.curve_fit(
    fit_logistic,
    f2_t,
    f2_df["x"],
    p0=[1, 1, 0],
    maxfev=10000,
)

x_fit = fit_logistic(f2_t, L_fit, k_fit, x0_fit)
plt.plot(f2_t, f2_df["x"], "ko--")
plt.plot(f2_t, x_fit, "r--")

# %% ###################################################################################
# fit gompertz growth to the data


def fit_gompertz(t, a, b, c):
    return a * np.exp(-b * np.exp(-c * t))


(a_fit, b_fit, c_fit), _ = scipy.optimize.curve_fit(
    fit_gompertz,
    f2_t,
    f2_df["x"],
    p0=[1, 1, 1],
    maxfev=1000000,
)
x_fit = fit_gompertz(f2_t, a_fit, b_fit, c_fit)
plt.plot(f2_t, f2_df["x"], "ko--")
plt.plot(f2_t, x_fit, "r--")

# %% ###################################################################################
# interpolate growth with splines
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
    return f2_p.loc[f2_t] - p_pred


res = scipy.optimize.least_squares(
    get_residuals,
    x0=[0.1, 0.1],
    bounds=([0, 0], [np.inf, np.inf]),
)

f2_pi_0, f2_pi_1 = res.x
print(f"{f2_pi_0=}, {f2_pi_1=}")
# %% ###################################################################################
plt.plot(f2_df.index, f2_df["p"], "ko--")
plt.plot(f2_t_interp, predict_p(f2_t_interp, *res.x), "r--")
# %% ###################################################################################
28.8 / 1000 * 46
# %% ###################################################################################
f2_pi_0 + 0.235 * f2_pi_1
