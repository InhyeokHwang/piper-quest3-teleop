from ..piper.driver import PiperDriver
from ..piper.safety import enable_and_wait

def init_driver(args):
    if args.dry_run:
        print("[DRY RUN] No hardware commands will be sent")
        return None

    driver = PiperDriver(args.can)
    driver.connect()
    enable_and_wait(driver, timeout_s=5.0, fail_hard=True, also_open_gripper=True)

    driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=50, is_mit_mode=0x00)  # joint mode    
    driver.set_gripper(position=20000, effort=2000, enable=True, clear_error=True)
    print("[Piper] Ready.")
    return driver