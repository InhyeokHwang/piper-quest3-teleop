# config.py

UDP_IP = "127.0.0.1"
UDP_PORT = 15000

START_POSITION = [0, 0, 0, 0, 0, 0, 0]

RAD_TO_PIPER = 57324.840764  # 1000*180/pi

IK_CONFIG = dict(
    max_iterations=100,
    position_tolerance=5e-2,
    orientation_tolerance=5e-2,
    damping_factor=0.5,
    use_analytical_jacobian=False,
)

CAMERA_SLEEP = 0.01
