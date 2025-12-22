#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np

from .VuerTeleop import VuerTeleop

from .kinematics.piper_forward_kinematics import PiperForwardKinematics, DHType
from .kinematics.piper_jacobian_ik import PiperJacobianIK

from .piper.driver import PiperDriver
from .piper.safety import enable_and_wait, move_to_start_pose

from .mapping.vr_mapper import VRToRobotMapper, VRMapperConfig

from .io.udp_sender import UDPSender, UDPSenderConfig
from .io.camera import OpenCVCameraStreamer, CameraStreamerConfig

from . import config
from typing import List



def rad6_to_piper_int6(q_rad: np.ndarray, factor: float) -> List[int]:
    q_rad = np.asarray(q_rad, dtype=float).reshape(6)
    return [round(float(q_rad[i]) * factor) for i in range(6)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--can", type=str, default="can0", help="CAN port name (default: can0)")
    p.add_argument("--dry-run", action="store_true", help="Do not send commands to real robot (print only)")
    p.add_argument("--config", type=str, default="inspire_hand.yml", help="VuerTeleop config yaml")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--no-udp", action="store_true", help="Disable UDP sending")
    p.add_argument("--print-freq", action="store_true", help="Print loop frequency")
    p.add_argument("--debug-mapper", action="store_true", help="Print VR mapping debug logs")
    return p.parse_args()


def main():
    args = parse_args()

    teleoperator = VuerTeleop(args.config)

    # Camera streamer -> teleoperator.img_array (H,2W,3)
    cam = OpenCVCameraStreamer(
        teleoperator.img_array,
        CameraStreamerConfig(camera_index=args.camera),
    )

    # UDP sender - ros2용 
    udp = None
    if not args.no_udp:
        udp = UDPSender(UDPSenderConfig(ip=config.UDP_IP, port=config.UDP_PORT))

    # FK/IK
    fk = PiperForwardKinematics(DHType.STANDARD)
    ik = PiperJacobianIK(fk=fk, **config.IK_CONFIG)

    # FK at q=0 -> EE reference
    q_zero = np.zeros(6, dtype=float) ## 조인트가 모두 0
    T_ee0 = fk.compute_fk(q_zero) # 0일 때의 EE 위치, 오리엔테이션
    EE_START = T_ee0[:3, 3].copy() # 위치
    R_ee0 = T_ee0[:3, :3].copy() # 오리엔테이션
    print("[INIT] EE_START from FK(q=0):", EE_START)

    # VR mapper
    mapper = VRToRobotMapper( # yup -z forward를 zup x forward로
        ee_start_pos=EE_START,
        ee_start_rot=R_ee0,
        config=VRMapperConfig(position_scale=1.0),
        debug=args.debug_mapper,
    )

    # Robot driver + safety
    driver = None
    if not args.dry_run:
        driver = PiperDriver(args.can)
        print("[Piper] Connecting:", args.can)
        driver.connect()

        enable_and_wait(driver, timeout_s=5.0, fail_hard=True)
        driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=100, acc=0x00)
        driver.set_gripper(position=0, speed=1000, enable=True)

        print("[Piper] Ready.")
    else:
        print("[DRY RUN] No hardware commands will be sent.")

    # Loop state
    last_q = q_zero.copy()
    gripper_pos = 0  # TODO: VR 버튼으로 바꾸기

    try:
        while True:
            loop_t0 = time.time()

            # get VR right pose
            _, right_pose = teleoperator.step()

            # build target EE pose
            target_T, _info = mapper.compute_target_T(right_pose)
            if target_T is None:
                cam.step()
                time.sleep(config.CAMERA_SLEEP)
                continue

            # IK solve
            try:
                q_sol, _ = ik.compute_ik(
                    initial_guess=last_q,
                    target_pose=target_T,
                    verbose=False,
                    return_final_error=True,
                )
                last_q = q_sol
            except RuntimeError:
                print("[IK] Failed to converge -> keep last_q")
                q_sol = last_q

            # UDP send (rad)
            if udp is not None:
                udp.send_floats(q_sol.tolist())

            # send to robot
            joint_int = rad6_to_piper_int6(q_sol, config.RAD_TO_PIPER)

            if args.dry_run:
                print(f"[DRY RUN] JointCtrl{tuple(joint_int)}")
                print(f"[DRY RUN] GripperCtrl(abs({gripper_pos}), 1000, 0x01, 0)")
            else:
                driver.send_joints(joint_int)
                driver.set_gripper(position=gripper_pos, speed=1000, enable=True)

            # camera -> stereo buffer
            cam.step()

            if args.print_freq:
                dt = max(time.time() - loop_t0, 1e-9)
                print("[Loop] freq:", 1.0 / dt)

            time.sleep(config.CAMERA_SLEEP)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted")

    finally:
        # camera
        try:
            cam.close()
        except Exception as e:
            print("[WARN] cam.close failed:", e)

        # safe shutdown
        if not args.dry_run and driver is not None:
            try:
                ok = move_to_start_pose(
                    driver,
                    config.START_POSITION,
                    config.RAD_TO_PIPER,
                    check_reached=True,
                )
                if ok:
                    driver.disable()
                    driver.set_gripper(position=0, speed=1000, enable=False)
                    print("[Piper] Motors disabled. Safe to power off.")
                else:
                    print("[Piper] NOT reached safe pose. Keeping motors enabled.")
            except Exception as e:
                print("[WARN] Safety shutdown failed:", e)

        # udp
        if udp is not None:
            try:
                udp.close()
            except Exception as e:
                print("[WARN] udp.close failed:", e)

        # teleoperator
        try:
            teleoperator.close()
        except Exception as e:
            print("[WARN] teleoperator.close failed:", e)


if __name__ == "__main__":
    main()
