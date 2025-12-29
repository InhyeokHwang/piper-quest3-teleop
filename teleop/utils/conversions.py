# teleop/utils/conversions.py
from typing import List
import numpy as np

def rad6_to_piper_int6(q_rad: np.ndarray, factor: float) -> List[int]:
    q_rad = np.asarray(q_rad, dtype=float).reshape(6)
    return [round(float(q_rad[i]) * factor) for i in range(6)]
