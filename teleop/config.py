# config.py
import numpy as np
from pathlib import Path

# config.py가 있는 폴더 = .../TeleVision/teleop
_TELEOP_DIR = Path(__file__).resolve().parent


UDP_IP = "127.0.0.1"
UDP_PORT = 15000

START_POSITION = [0, 0, 0, 0, 0, 0, 0, 0]

RAD_TO_PIPER = 57324.840764  # 1000*180/pi

####### MINK ############
PIPER_MJCF_PATH = str(_TELEOP_DIR / "piper" / "agilex_piper" / "piper.xml")
MINK_EE_SITE = "attachment_site"  # 네 MJCF에서 end-effector site 이름
MINK_SOLVER = "daqp"             
MINK_DT = 0.01                    # IK 적분 dt (config.SLEEP와 맞춰도 됨)
MINK_LM_DAMPING = 1e-6
MINK_POSTURE_COST = 1e-3
#######MINK##############

# === Gripper mapping ===
GRIPPER_MAX_MM = 70.0          # 매뉴얼: 0~70mm
GRIPPER_UNIT_MM = 0.001        # SDK: 0.001mm 단위
GRIPPER_MAX_UM = int(GRIPPER_MAX_MM / GRIPPER_UNIT_MM)  # 70000

SIM_GRIPPER_RANGE = 0.035      # MuJoCo에서 joint7/8 벌리는 범위(너가 쓰던 값)
