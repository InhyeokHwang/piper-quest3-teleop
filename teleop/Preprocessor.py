import math
import numpy as np

from .constants_vuer import grd_yup2grd_zup, hand2inspire
from .motion_utils import mat_update, fast_mat_inv
from .TeleVision import OpenTeleVision

# VR로부터 넘어온 데이터를 4x4 행렬 세트로 정리
class VuerPreprocessor:
    def __init__(self): 
        # vuer를 키고 시작하는 원점은 불확식하며, 오른쪽 컨트롤러의 절대 좌표 시작점을 기준으로 한 상대좌표로 하는 것도 생각해볼 수 있음.
        # 그러나 디버깅의 효율성을 위해 머리를 기준으로 하고 오른쪽 컨트롤러를 머리 기준 상대좌표로 함.
        self.vuer_head_mat = np.eye(4)
        self.vuer_right_ctrl_mat = np.eye(4)
        # self.vuer_left_ctrl_mat = np.eye(4)

    # 좌표들을 살펴보면 Y축이 머리 위로 향하는 좌표계임을 알 수 있음. 여기서는 Y UP, -Z forward 시스템을 쓴다고 함. 
    def process(self, tv : OpenTeleVision):
        # mat_update의 역할은 행렬의 determinant가 0일 경우에 이전 값을 유지하고, 아니면 새 행렬로 업데이트하는 함수
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_ctrl_mat = mat_update(self.vuer_right_ctrl_mat, tv.right_controller.copy())

        # Y up 시스템(VR에서 쓰는 좌표계)을 Z up 시스템(시뮬레이션/제어 에서 쓰는 좌표계)으로 바꾸어야 함.
        # 새로운 행렬 = (변환행렬) x (기존 자세) x (변환행렬^-1) -> 기저변환
        head_mat    = grd_yup2grd_zup @ self.vuer_head_mat       @ fast_mat_inv(grd_yup2grd_zup)
        right_ctrl  = grd_yup2grd_zup @ self.vuer_right_ctrl_mat @ fast_mat_inv(grd_yup2grd_zup)

        ## 사람의 머리를 (0, 0, 0)으로 놓고 그 기준에서 손이 어딨는지 확인하기 위해 head_mat을 빼줌
        rel_right_ctrl = right_ctrl.copy()
        # rel_left_ctrl  = left_ctrl.copy()

        rel_right_ctrl[0:3, 3] -= head_mat[0:3, 3]
        # rel_left_ctrl[0:3, 3]  -= head_mat[0:3, 3]

        return head_mat, rel_right_ctrl
