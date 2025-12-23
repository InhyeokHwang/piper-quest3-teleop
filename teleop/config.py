# config.py
import numpy as np
from pathlib import Path

# config.py가 있는 폴더 = .../TeleVision/teleop
_TELEOP_DIR = Path(__file__).resolve().parent


UDP_IP = "127.0.0.1"
UDP_PORT = 15000

START_POSITION = [0, 0, 0, 0, 0, 0, 0]

RAD_TO_PIPER = 57324.840764  # 1000*180/pi

####### MINK ############
PIPER_MJCF_PATH = str(_TELEOP_DIR / "piper" / "agilex_piper" / "piper.xml")
MINK_EE_SITE = "attachment_site"  # 네 MJCF에서 end-effector site 이름
MINK_SOLVER = "daqp"             
MINK_DT = 0.01                    # IK 적분 dt (config.SLEEP와 맞춰도 됨)
MINK_LM_DAMPING = 1e-6
MINK_POSTURE_COST = 1e-3
#######MINK##############

IK_CONFIG = dict(
    max_iterations=100,
    position_tolerance=5e-2,
    orientation_tolerance=5e-2,
    damping_factor=0.5,
    use_analytical_jacobian=False,

    # --- nullspace posture bias ---
    enable_nullspace=True,
    q_rest = np.array([
        0.004430889,
        0.572840667,
    -0.696521778,
    -0.035883222,
        0.705470778,
    -0.007658111,
    ], dtype=float),
    nullspace_gain=0.08,
    nullspace_weights=[1.0, 1.0, 1.5, 1.0, 2.0, 2.0],  # 손목쪽 더 강하게
    sigma_min_threshold=0.02,
    nullspace_max_step=0.05,
)

SLEEP = 0.01
