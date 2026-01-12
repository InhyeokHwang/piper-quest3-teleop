# vuer_teleop.py
from __future__ import annotations

from pathlib import Path
from multiprocessing import Event, Queue, shared_memory
from typing import Tuple

import numpy as np
from pytransform3d import rotations

from .TeleVision import OpenTeleVision
from .Preprocessor import VuerPreprocessor

class VuerTeleop:
    def __init__(self, config_file_path: str):
        # Vuer/Quest3에서 쓸 이미지 해상도 (H, W)
        self.resolution = (720, 1280)

        # crop 설정 (현재는 crop 없음)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        # (H, 2W, 3) : 왼쪽눈/오른쪽눈 이미지를 가로로 붙인 버퍼
        self.img_shape = (
            self.resolution_cropped[0],
            2 * self.resolution_cropped[1],
            3,
        )
        self.img_height, self.img_width = self.resolution_cropped[:2]

        # 공유 메모리 생성 (uint8 버퍼)
        nbytes = int(np.prod(self.img_shape) * np.dtype(np.uint8).itemsize)
        self.shm = shared_memory.SharedMemory(create=True, size=nbytes)

        # OS shared memory에 직접 매핑된 numpy view
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )

        # Quest3 네트워크 통신 담당 (서버)
        # ex) https://192.168.x.x:PORT?ws=wss://192.168.x.x:PORT
        self.tv = OpenTeleVision(
            self.resolution_cropped,
            self.shm.name,
        )

        # raw VR 데이터를 가공해주는 전처리기
        self.processor = VuerPreprocessor()


    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns:
          right_pose:(7,)  [x,y,z,qx,qy,qz,qw]  (quat is xyzw)
        """
        # raw VR 데이터를 가공 (y-up -> z-up, 등)
        right_wrist_mat = self.processor.process(self.tv)

        # 오른손 pose [x, y, z, qx, qy, qz, qw]
        right_quat_wxyz = rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])
        right_quat_xyzw = right_quat_wxyz[[1, 2, 3, 0]]
        right_pose = np.concatenate([right_wrist_mat[:3, 3], right_quat_xyzw])

        return right_pose

    @property
    def right_state(self) -> np.ndarray:
        return self.tv.right_state
    @property
    def left_state(self) -> np.ndarray:
        return self.tv.left_state
    
    def close(self) -> None:
        shm = getattr(self, "shm", None)
        if shm is None:
            return
        try:
            shm.close()
        except Exception:
            pass

        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

        self.shm = None