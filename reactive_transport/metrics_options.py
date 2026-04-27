import numpy as np


def _to_1d_float(a):
    return np.asarray(a, dtype=float).ravel()


def least_squares_error(y_pred, y_true, mean: bool = True, weights=None):
    y_pred = _to_1d_float(y_pred)
    y_true = _to_1d_float(y_true)
    err2 = (y_pred - y_true) ** 2
    if weights is None:
        return float(err2.mean() if mean else err2.sum())
    w = _to_1d_float(weights)
    wsum = float(np.sum(w))
    if mean:
        return float(np.sum(w * err2) / wsum)
    return float(np.sum(w * err2))


def calcR2(y_pred, y_true, weights=None):
    y_pred = _to_1d_float(y_pred)
    y_true = _to_1d_float(y_true)
    if weights is None:
        ybar = float(np.mean(y_true))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - ybar) ** 2))
    else:
        w = _to_1d_float(weights)
        wsum = float(np.sum(w))
        ybar = float(np.sum(w * y_true) / wsum)
        ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
        ss_tot = float(np.sum(w * (y_true - ybar) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def corr_R2(y_pred, y_true):
    x = _to_1d_float(y_pred)
    y = _to_1d_float(y_true)
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x0 * y0) / denom)


def weighted_corr(y_pred, y_true, weights):
    x = _to_1d_float(y_pred)
    y = _to_1d_float(y_true)
    w = _to_1d_float(weights)
    wsum = float(np.sum(w))
    mx = float(np.sum(w * x) / wsum)
    my = float(np.sum(w * y) / wsum)
    x0 = x - mx
    y0 = y - my
    cov = float(np.sum(w * x0 * y0) / wsum)
    vx = float(np.sum(w * x0 * x0) / wsum)
    vy = float(np.sum(w * y0 * y0) / wsum)
    denom = np.sqrt(vx * vy)
    if denom <= 0.0:
        return 0.0
    return cov / denom


def integrated_squared_error(y_pred, y_true, time_points):
    y_pred = _to_1d_float(y_pred)
    y_true = _to_1d_float(y_true)
    t = _to_1d_float(time_points)
    dt = t[-1] - t[0]
    return float(np.trapezoid((y_pred - y_true) ** 2, t) / dt)


def integrated_squared_error_asinh(y_pred, y_true, time_points, c_star):
    y_pred = _to_1d_float(y_pred)
    y_true = _to_1d_float(y_true)
    t = _to_1d_float(time_points)
    diff = np.arcsinh(y_pred / c_star) - np.arcsinh(y_true / c_star)
    dt = t[-1] - t[0]
    return float(np.trapezoid(diff ** 2, t) / dt)


def calc_whole_curve_nrmse(y_pred, y_true, time_points):
    y_pred = _to_1d_float(y_pred)
    y_true = _to_1d_float(y_true)
    t = _to_1d_float(time_points)
    num = float(np.trapezoid((y_pred - y_true) ** 2, t))
    den = float(np.trapezoid(y_true ** 2, t))
    if den <= 0.0:
        return np.inf
    return float(np.sqrt(num / den))


def calc_peak_time_rel_error(y_pred, y_true, time_points):
    y_pred = _to_1d_float(y_pred)
    y_true = _to_1d_float(y_true)
    t = _to_1d_float(time_points)
    t_peak_obs = t[np.argmax(y_true)]
    t_peak_pred = t[np.argmax(y_pred)]
    if t_peak_obs <= 0.0:
        return np.inf
    return float(abs(t_peak_pred - t_peak_obs) / t_peak_obs)


def calc_peak_conc_rel_error(y_pred, y_true):
    y_pred = _to_1d_float(y_pred)
    y_true = _to_1d_float(y_true)
    c_peak_obs = float(np.max(y_true))
    c_peak_pred = float(np.max(y_pred))
    if c_peak_obs <= 0.0:
        return np.inf
    return float(abs(c_peak_pred - c_peak_obs) / c_peak_obs)


def calc_post_peak_nrmse(y_pred, y_true, time_points):
    y_pred = _to_1d_float(y_pred)
    y_true = _to_1d_float(y_true)
    t = _to_1d_float(time_points)
    i_pk = int(np.argmax(y_true))
    t_post = t[i_pk:]
    y_pred_post = y_pred[i_pk:]
    y_true_post = y_true[i_pk:]
    if t_post.size < 2:
        return np.inf
    num = float(np.trapezoid((y_pred_post - y_true_post) ** 2, t_post))
    den = float(np.trapezoid(y_true_post ** 2, t_post))
    if den <= 0.0:
        return np.inf
    return float(np.sqrt(num / den))

