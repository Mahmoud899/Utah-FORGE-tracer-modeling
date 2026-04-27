import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint

from reactive_transport.metrics_options import calc_whole_curve_nrmse
from reactive_transport.simulation_options_2 import simulateDualPorosity


GLOBAL_BOUNDS = {
    "MRT1": (1.0, 60.0),
    "MRT2": (1.0, 60.0),
    "Pe1": (0.1, 40.0),
    "Pe2": (0.1, 40.0),
    "fr1": (0.0, 1.0),
    "fr2": (0.0, 1.0),
}


def load_data(csv_path="data/data.csv"):
    df = pd.read_csv(csv_path)
    weights = df["weight"].to_numpy()
    mask = weights.astype(bool)
    t_true = df["Time_hr"].to_numpy()[mask]
    c_true = df["C_corr_mgL"].to_numpy()[mask]
    return t_true, c_true


def modelRELAP(t, mrt, pec):
    frac_retard = 1.0
    bckgrnd_conc = 0.0
    inj_concs = np.array([7.0, 0.0], dtype=float)
    inj_durs = np.cumsum(np.array([1.5], dtype=float))
    dpParam = 0
    matrix_retardation = 1.0
    recRatio = 0.0
    wsCoef = 0.0
    return simulateDualPorosity(
        mrt,
        pec,
        frac_retard,
        t,
        bckgrnd_conc,
        inj_concs,
        inj_durs,
        dpParam,
        matrix_retardation,
        recRatio=recRatio,
        wsCoef=wsCoef,
        delay_time=0,
    )


def modelRELAP2(t, mrt1, mrt2, pec1, pec2, fr1, fr2):
    c1 = modelRELAP(t, mrt1, pec1)
    c2 = modelRELAP(t, mrt2, pec2)
    return fr1 * c1 + fr2 * c2


def objective_vector(x, t_true, c_true):
    mrt1, mrt2, pec1, pec2, fr1, fr2 = x
    y_hat = modelRELAP2(t_true, mrt1, mrt2, pec1, pec2, fr1, fr2)
    return calc_whole_curve_nrmse(y_hat, c_true, t_true)


def build_fraction_constraint():
    a = np.zeros((1, 6), dtype=float)
    a[0, 4] = 1.0
    a[0, 5] = 1.0
    return LinearConstraint(a, -np.inf, 1.0)


def build_mrt_order_constraint():
    a = np.zeros((1, 6), dtype=float)
    a[0, 0] = -1.0
    a[0, 1] = 1.0
    return LinearConstraint(a, 0.0, np.inf)


def build_constraints():
    return (build_fraction_constraint(), build_mrt_order_constraint())

