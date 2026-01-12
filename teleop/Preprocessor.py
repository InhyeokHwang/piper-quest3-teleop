import numpy as np

from .constants_vuer import grd_yup2grd_zup
from .motion_utils import mat_update, fast_mat_inv
from .TeleVision import OpenTeleVision

# VR로부터 넘어온 데이터를 4x4 행렬 세트로 정리
class VuerPreprocessor:
    def __init__(self): 
        # vuer를 키고 시작하는 오른쪽 컨트롤러의 절대 좌표 시작점
        self.vuer_right_ctrl_mat = np.eye(4)
        self._T = grd_yup2grd_zup
        self._Tinv = fast_mat_inv(grd_yup2grd_zup)
        # [[1 0 0 0]
        #  [0 1 0 0]
        #  [0 0 1 0]
        #  [0 0 0 1]] 의 의미는 초기값을 pos = [0 0 0], quat_xyzw = [0 0 0 1]로 만듦

    # 좌표들을 살펴보면 Y축이 머리 위로 향하는 좌표계임을 알 수 있음. 여기서는 Y UP, -Z forward 시스템을 쓴다고 함. 
    def process(self, tv : OpenTeleVision):
        # mat_update의 역할은 행렬의 determinant가 0일 경우에 이전 값을 유지하고, 아니면 새 행렬로 업데이트하는 함수
        self.vuer_right_ctrl_mat = mat_update(self.vuer_right_ctrl_mat, tv.right_controller.copy())

        # Y up 시스템(VR에서 쓰는 좌표계)을 Z up 시스템(시뮬레이션/제어 에서 쓰는 좌표계)으로 바꾸어야 함.
        # 새로운 행렬 = (변환행렬) x (기존 자세) x (변환행렬^-1) -> 기저변환
        right_ctrl = self._T @ self.vuer_right_ctrl_mat @ self._Tinv

        return right_ctrl