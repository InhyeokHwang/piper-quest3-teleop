import numpy as np

def joints123_near_zero(q, tol_deg=5.0):
    tol = np.deg2rad(tol_deg)
    q = np.asarray(q).reshape(-1)
    return np.all(np.abs(q[:3]) <= tol)