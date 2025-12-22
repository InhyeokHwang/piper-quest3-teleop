# io/camera.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2


@dataclass
class CameraStreamerConfig:
    camera_index: int = 0
    rgb: bool = True                 # convert BGR->RGB
    duplicate_stereo: bool = True    # left=right=frame
    warn_every_n: int = 60           # warn print period on read failure

# openCV 카메라로부터 읽고 shared stereo RGB 버퍼에 기록
class OpenCVCameraStreamer:
    """
    Intended usage:
      - teleoperator.img_array is (H, 2W, 3)
      - camera provides (H, W, 3)
      - we hstack(frame, frame) -> (H, 2W, 3)
      - np.copyto(dst, stereo)
    """

    def __init__(self, dst_stereo_array: np.ndarray, config: CameraStreamerConfig | None = None):
        
        self.cfg = config if config is not None else CameraStreamerConfig()

        self.dst = dst_stereo_array
        if self.dst.ndim != 3 or self.dst.shape[2] != 3:
            raise ValueError(f"dst_stereo_array must be (H, 2W, 3), got {self.dst.shape}")

        self.img_h = int(self.dst.shape[0])
        self.img_w = int(self.dst.shape[1])
        self.single_w = self.img_w // 2

        if self.img_w % 2 != 0:
            raise ValueError(f"dst width must be even (2W). got W={self.img_w}")

        self.cap: Optional[cv2.VideoCapture] = None
        self._fail_count = 0
        self._tick = 0

    def open(self) -> bool:
        if self.cap is not None:
            return True

        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            print(f"[Camera] Failed to open camera index {self.cfg.camera_index}.")
            self.cap = None
            return False

        # Request size close to target
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.single_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_h)

        print(f"[Camera] Opened camera {self.cfg.camera_index}. Target: {self.img_h} x {self.single_w}")
        return True

    def close(self):
        if self.cap is None:
            return
        try:
            self.cap.release()
            print("[Camera] Released camera.")
        finally:
            self.cap = None

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] != self.img_h or frame.shape[1] != self.single_w:
            frame = cv2.resize(frame, (self.single_w, self.img_h))
        return frame

    # 한 프레임 읽고 dst stereo buffer에 기록
    def step(self) -> bool:
        
        self._tick += 1

        if self.cap is None:
            if not self.open():
                return False

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._fail_count += 1
            if self.cfg.warn_every_n > 0 and (self._tick % self.cfg.warn_every_n == 0):
                print("[Camera] Failed to read frame from camera.")
            return False

        # OpenCV 는 BGR 형태이므로..
        if self.cfg.rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = self._resize_if_needed(frame)

        if self.cfg.duplicate_stereo:
            stereo = np.hstack((frame, frame))
        else:
            # If later you have true stereo source, you can override this
            stereo = np.hstack((frame, frame))

        np.copyto(self.dst, stereo)
        return True
