import numpy as np
import pandas as pd

from reactive_transport.metrics_options import calc_whole_curve_nrmse
from reactive_transport.simulation_options_2 import simulateDualPorosity


GLOBAL_BOUNDS = {
    "R": (0.0, 1.0),
    "MRT": (4.0, 60.0),
    "Pe": (1.0, 30.0),
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


def objective_vector(x, t_true, c_true):
    R, mrt, pec = x
    y_hat = R * modelRELAP(t_true, mrt, pec)
    return calc_whole_curve_nrmse(y_hat, c_true, t_true)

