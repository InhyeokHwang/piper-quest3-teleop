#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

import mink

from loop_rate_limiters import RateLimiter

from .VuerTeleop import VuerTeleop

from .kinematics.piper_forward_kinematics import PiperForwardKinematics, DHType

from .piper.driver import PiperDriver
from .piper.safety import enable_and_wait, move_to_start_pose

from .mapping.vr_mapper import VRToRobotMapper, VRMapperConfig

from .io.camera import OpenCVCameraStreamer, CameraStreamerConfig

from .controllers.gripper import GripperController, GripperConfig

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

    # FK
    fk = PiperForwardKinematics(DHType.STANDARD)
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

    # =========================
    # mink IK 초기화 (여기가 핵심)
    # =========================
    # config.py에 아래 두 개는 반드시 있어야 함:
    #   MINK_XML_PATH = "/abs/path/to/piper_scene.xml"
    #   MINK_EE_SITE  = "attachment_site"  # mjcf 내 site 이름
    model = mujoco.MjModel.from_xml_path(config.PIPER_MJCF_PATH)

    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    if args.dry_run:
        print("=== MuJoCo joints ===")
        for j in range(model.njnt):
            print(j, model.joint(j).name)


    limits = [mink.ConfigurationLimit(model=model)]

    max_vel = float(getattr(config, "MINK_MAX_VEL", np.pi))  # 기본 pi rad/s
    max_velocities = {
        "joint1": max_vel,
        "joint2": max_vel,
        "joint3": max_vel,
        "joint4": max_vel,
        "joint5": max_vel,
        "joint6": max_vel,
    }
    limits.append(mink.VelocityLimit(model, max_velocities))

    ## mujoco viewer - dry run에서만 실행할 것임
    viewer = None
    if args.dry_run:
        # MuJoCo viewer는 디버깅용: dry-run에서만 띄움
        viewer = mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        )
        # 보기 편하게 free camera 기본값
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        print("[DRY RUN] MuJoCo viewer launched.")

    # solver / dt
    solver = getattr(config, "MINK_SOLVER", "daqp")

    # IK/control loop rate (예: 200Hz)
    rate_hz = float(getattr(config, "MINK_RATE_HZ", 200.0))
    rate = RateLimiter(frequency=rate_hz, warn=False)

    # Tasks
    end_effector_task = mink.FrameTask(
        frame_name=config.MINK_EE_SITE,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.3,
        lm_damping=float(getattr(config, "MINK_LM_DAMPING", 1e-6)),
    )
    posture_task = mink.PostureTask(model, cost=float(getattr(config, "MINK_POSTURE_COST", 1e-3)))
    tasks = [end_effector_task, posture_task]

    # posture target(q_rest) 기본값: q_zero를 기반으로 전체 q 차원에 맞춰 세팅
    # mink posture_task는 model nq 길이의 q 타겟을 받는 게 안전함
    q_rest_full = configuration.q.copy()
    q_rest_full[:6] = q_zero
    posture_task.set_target(q_rest_full)


    # Robot driver + safety
    driver = None
    if not args.dry_run:
        driver = PiperDriver(args.can)
        print("[Piper] Connecting:", args.can)
        driver.connect()

        enable_and_wait(driver, timeout_s=5.0, fail_hard=True)
        driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=100, acc=0x00)
        driver.set_gripper(position=0, enable=True)

        print("[Piper] Ready.")
    else:
        print("[DRY RUN] No hardware commands will be sent.")

    ## gripper
    gripper_ctl = GripperController(GripperConfig(
        # === control mode ===
        mode="analog",              # 누르는 동안만 활성화
        analog_source="trigger",    # "trigger" or "squeeze"

        # === right_state index mapping ===
        idx_a_button=4,             # A 버튼 (analog 모드에서는 사실상 미사용)
        idx_trigger_value=6,        # triggerValue (0.0 ~ 1.0)
        idx_squeeze_value=7,        # squeezeValue (0.0 ~ 1.0)

        # === output scale (Piper gripper) ===
        out_min=0,
        out_max=1000,

        # === behavior tuning ===
        alpha=0.35,                 # smoothing (0~1)
        deadzone_low=0.05,          # 미세 떨림 제거
        deadzone_high=0.95,
        close_when_high=True,       # trigger 많이 누를수록 닫힘
    ))

    # Loop state
    last_q = q_zero.copy()
    T_filt = None  # filtered target_T (4x4)
    gripper_pos = 0

    send_hz = float(getattr(config, "SEND_RATE_HZ", 60.0))
    send_period = 1.0 / max(send_hz, 1e-6)
    next_send = time.monotonic()  # 다음 전송 시각

    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break

            loop_t0 = time.time()

            _, right_pose = teleoperator.step()

            # target EE pose (from VR)
            target_T, _info = mapper.compute_target_T(right_pose)

            # --- guard: mapper not ready yet ---
            if target_T is None:
                # 그래도 viewer에는 현재 last_q를 계속 반영 (멈춘 것처럼 안 보이게)
                if viewer is not None:
                    data.qpos[:6] = last_q
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                rate.sleep()
                continue
            

            alpha = float(getattr(config, "EE_FILTER_ALPHA", 0.2))  # 0.1~0.3 추천
            pos_deadband = float(getattr(config, "EE_POS_DEADBAND", 0.001))  # 1mm

            if T_filt is None:
                T_filt = target_T.copy()
            else:
                # position EMA + deadband
                p = target_T[:3, 3]
                p_prev = T_filt[:3, 3]
                dp = p - p_prev
                if np.linalg.norm(dp) >= pos_deadband:
                    T_filt[:3, 3] = (1 - alpha) * p_prev + alpha * p

                # rotation: 일단 가장 쉬운 버전(회전은 그대로 두거나, alpha를 더 작게)
                # (진짜 slerp까지 넣으면 더 좋아지는데, 우선 이 단계에서 체감 확인)
                T_filt[:3, :3] = target_T[:3, :3]

            target_T_use = T_filt

            T_wt = mink.SE3.from_matrix(target_T_use)
            end_effector_task.set_target(T_wt)

            # =========================
            # mink IK solve (기존 ik.compute_ik(...) 블록 교체)
            # =========================
            try:
                # 1) configuration에 현재 q 반영 (모델 q 전체 중 앞 6개만 arm이라고 가정)
                q_full = configuration.q.copy()
                q_full[:6] = last_q
                configuration.q[:] = q_full

                # 2) mujoco forward로 kinematics 업데이트 (안전하게)
                #    (mink 내부 update 메서드가 있더라도 mj_forward가 확실함)
                data.qpos[:] = configuration.q

                mujoco.mj_forward(model, data)

                # 3) target pose를 mink.SE3로 변환해서 task 타겟 업데이트
                T_wt = mink.SE3.from_matrix(target_T_use)
                end_effector_task.set_target(T_wt)

                # 4) solve_ik -> velocity, integrate
                dt = rate.dt
                vel = mink.solve_ik(configuration, tasks, dt, solver, limits=limits)
                configuration.integrate_inplace(vel, dt)

                # ✅ MuJoCo data에 qpos 반영 (viewer는 data.qpos를 봄)
                data.qpos[:] = configuration.q
                mujoco.mj_forward(model, data)

                # 5) 결과 q 회수 (arm 6축만)
                last_q = np.asarray(configuration.q[:6], dtype=float).copy()
            
            except Exception as e:
                print("[mink IK] Failed -> keep last_q:", repr(e))

            # --- Skeleton render (use last_q) ---
            joints_xyz = fk.fk_all_joint_positions(last_q)
            teleoperator.tv.set_robot_joints(joints_xyz)
            # teleoperator.tv.set_robot_joints(test_joints)

            # --- Gripper ---
            gripper_pos = gripper_ctl.update(teleoperator)
            t = gripper_pos / 1000.0
            open_ratio = 1.0 - t
            joint7 =  0.035 * open_ratio
            joint8 = -0.035 * open_ratio

            # send to robot
            now = time.monotonic()
            if now >= next_send:
                # (드리프트 방지) 늦었으면 catch-up 하되 너무 누적되지 않게 한 스텝만
                while next_send <= now:
                    next_send += send_period

                # send to robot (60Hz)
                joint_int = rad6_to_piper_int6(last_q, config.RAD_TO_PIPER)

                if args.dry_run:
                    print(f"[DRY RUN] JointCtrl{tuple(joint_int)}")
                else:
                    driver.send_joints(joint_int)
                    driver.set_gripper(position=gripper_pos, effort=2000, enable=True)
            
            cam.step()

            if args.print_freq:
                dt = max(time.time() - loop_t0, 1e-9)
                print("[Loop] freq:", 1.0 / dt)

            if viewer is not None:
                viewer.sync()

            rate.sleep()


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
                    driver.set_gripper(position=0, effort=0, enable=False)
                    print("[Piper] Motors disabled. Safe to power off.")
                else:
                    print("[Piper] NOT reached safe pose. Keeping motors enabled.")
            except Exception as e:
                print("[WARN] Safety shutdown failed:", e)

        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

        # teleoperator
        try:
            teleoperator.close()
        except Exception as e:
            print("[WARN] teleoperator.close failed:", e)


if __name__ == "__main__":
    main()
