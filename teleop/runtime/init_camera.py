# teleop/runtime/init_camera.py
from typing import Optional
from ..io.camera import OpenCVCameraStreamer, CameraStreamerConfig

def init_camera(teleoperator, camera_index: Optional[int]):
    if camera_index is None:
        print("[Camera] Disabled (no --camera).")
        return None
    return OpenCVCameraStreamer(
        teleoperator.img_array,
        CameraStreamerConfig(camera_index=camera_index),
    )
