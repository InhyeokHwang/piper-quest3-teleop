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

    # FK - EE 초기 위치 계산
    fk = PiperForwardKinematics(DHType.STANDARD) 
    # FK at q=0 -> EE reference
    q_zero = np.zeros(6, dtype=float) ## 조인트가 모두 0
    T_ee0 = fk.compute_fk(q_zero) # 0일 때의 EE 위치, 오리엔테이션
    EE_START = T_ee0[:3, 3].copy() # 위치
    R_ee0 = T_ee0[:3, :3].copy() # 오리엔테이션
    print("[INIT] EE_START from FK(q=0):", EE_START)

    # VR mapper 4x4 homogeneous matrix
    mapper = VRToRobotMapper( # yup -z forward를 zup x forward로
        ee_start_pos=EE_START,
        ee_start_rot=R_ee0,
        config=VRMapperConfig(position_scale=1.0),
        debug=args.debug_mapper,
    )

    # mink IK 초기화 
    model = mujoco.MjModel.from_xml_path(config.PIPER_MJCF_PATH) # MJCF 경로
    data = mujoco.MjData(model)  
    configuration = mink.Configuration(model)

    # Dry run - MuJoCo Viewer로
    if args.dry_run:
        print("=== MuJoCo joints ===")
        for j in range(model.njnt):
            print(j, model.joint(j).name)

    # MJCF로부터 joint의 range, limit이 켜져있는지 여부, 등등 IK에서 QP(quadratic programming) 부등식 제약을 만들기 위한 준비.
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
    ## 관절 속도 리밋을 주입
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
    ## mink ik는 직접 closed form으로 풀지 않고 EE목표를 만족하는 joint velocity/step을 찾는 문제를 QP형태로 구성하는 것임.
    ## 그 qp 문제를 풀어주는 엔진이 daqp임

    # IK/control loop rate - 이건 실제 로봇에 명령을 보내는 주기가 아님. IK 계산, 상태 갱신 등 처리되는 주기임
    rate_hz = float(getattr(config, "MINK_RATE_HZ", 200.0))
    rate = RateLimiter(frequency=rate_hz, warn=False)

    # Tasks - MINK가 풀어야할 목표를 정의하는 부분. IK를 QP로 풀 때 무엇을 만족시키고 무엇을 최소화할지 구성하는 재료.
    end_effector_task = mink.FrameTask(
        frame_name=config.MINK_EE_SITE,
        frame_type="site", 
        position_cost=1.0, # EE 위치 오차를 줄이는데 주는 가중치 -> orientation보다 position을 우선시(안정적)
        orientation_cost=0.3, # EE 회전 오차를 줄이는데 주는 가중치 
        lm_damping=float(getattr(config, "MINK_LM_DAMPING", 1e-6)),
    )
    posture_task = mink.PostureTask(model, cost=float(getattr(config, "MINK_POSTURE_COST", 1e-3)))
    tasks = [end_effector_task, posture_task] 

    # posture target(q_rest) 기본값: q_zero를 기반으로 전체 q 차원에 맞춰 세팅
    # mink posture_task는 model nq 길이의 q 타겟을 받는 게 안전함
    q_rest_full = configuration.q.copy()
    q_rest_full[:6] = q_zero # [0, 0, 0, 0, 0, 0]
    posture_task.set_target(q_rest_full) # 현재 qpos가 q_rest_full에서 너무 멀어지지 않도록. 

    # Robot driver + safety
    driver = None
    if not args.dry_run: # 실제 파이퍼
        driver = PiperDriver(args.can)
        print("[Piper] Connecting:", args.can)
        driver.connect()

        enable_and_wait(driver, timeout_s=5.0, fail_hard=True, also_open_gripper=True)
        driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=100, acc=0x00)
        driver.set_gripper(position=20000, effort=2000, enable=True, clear_error=True)

        print("[Piper] Ready.")
    else: # Dry Run 
        print("[DRY RUN] No hardware commands will be sent.")

    ## gripper 세팅
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

    send_hz = float(getattr(config, "SEND_RATE_HZ", 60.0)) # 실제 로봇에 각도를 보내는 주기
    send_period = 1.0 / max(send_hz, 1e-6)
    next_send = time.monotonic()  # 다음 전송 시각

    try:
        # MuJoCo에서 data.qpos[]는 모든 관절의 qpos가 들어있는 배열
        j7 = model.joint("joint7")
        j8 = model.joint("joint8")

        # .item()으로 int 변환 (MuJoCo qpos 슬롯 주소)
        q_idx7 = int(np.asarray(j7.qposadr).item())
        q_idx8 = int(np.asarray(j8.qposadr).item())

        # 루프
        while True:
            ## Dry run에서 MuJoCo viewer 창을 닫으면 루프 빠져나감
            if viewer is not None and not viewer.is_running():
                break

            loop_t0 = time.time() # print_freq 용도

            _, right_pose = teleoperator.step() # vr 컨트롤러로부터 오른손 포즈 받기

            # Gripper 먼저 계산 
            gripper_pos = gripper_ctl.update(teleoperator)  # 0..1000
            t = gripper_pos / 1000.0
            open_ratio = 1.0 - t  # 1=open, 0=close

            # (A) MuJoCo용 joint7/8 (dry-run viewer용)
            joint7 =  config.SIM_GRIPPER_RANGE * open_ratio
            joint8 = -config.SIM_GRIPPER_RANGE * open_ratio

            # (B) 실로봇용 (0.001mm 단위)
            grip_um = int(round(config.GRIPPER_MAX_UM * open_ratio))


            # target EE pose (from VR)
            target_T, _info = mapper.compute_target_T(right_pose) # vr -> 로봇 변환행렬

            # --- guard: mapper not ready yet ---
            if target_T is None:
                # 그래도 viewer에는 현재 last_q를 계속 반영 (멈춘 것처럼 안 보이게)
                if viewer is not None:
                    data.qpos[:6] = last_q
                    data.qpos[q_idx7] = joint7   
                    data.qpos[q_idx8] = joint8   
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                rate.sleep()
                continue

            alpha = float(getattr(config, "EE_FILTER_ALPHA", 0.2))  # 0.1~0.3 추천 EMA(Exponential Moving Average) 스무딩 계수
            pos_deadband = float(getattr(config, "EE_POS_DEADBAND", 0.001))  # 1mm

            # T_filt는 4x4변환행렬을 저장하는 변수, target_T를 EMA와 섞어서 부드럽게 만듦
            if T_filt is None: # 4x4 변환행렬
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

            T_wt = mink.SE3.from_matrix(target_T_use) # 4x4 행렬로부터 R, p 추출하여 Mink.SE3객체 생성
            end_effector_task.set_target(T_wt) 

            ############# mink IK solve ################# 
            try:
                # configuration에 현재 q 반영 (모델 q 전체 중 앞 6개만 arm이라고 가정)
                q_full = configuration.q.copy()
                q_full[:6] = last_q
                configuration.q[:] = q_full

                # mujoco forward로 kinematics 업데이트 (안전하게)
                # (mink 내부 update 메서드가 있더라도 mj_forward가 확실함)
                data.qpos[:] = configuration.q
                mujoco.mj_forward(model, data)
                print(f"[QPOS CHECK] q7={float(data.qpos[q_idx7]):.5f} q8={float(data.qpos[q_idx8]):.5f}")

                # target pose를 mink.SE3로 변환해서 task 타겟 업데이트
                T_wt = mink.SE3.from_matrix(target_T_use)
                end_effector_task.set_target(T_wt)

                # solve_ik -> velocity, integrate
                dt = rate.dt
                vel = mink.solve_ik(configuration, tasks, dt, solver, limits=limits)
                configuration.integrate_inplace(vel, dt)

                # MuJoCo data에 qpos 반영 (viewer는 data.qpos를 봄)
                data.qpos[:] = configuration.q

                # 그리퍼는 "data.qpos"에 마지막으로 강제 주입 (여기가 핵심!)
                data.qpos[q_idx7] = float(joint7)
                data.qpos[q_idx8] = float(joint8)

                configuration.q[:] = data.qpos

                mujoco.mj_forward(model, data)

                # 결과 q 회수 (arm 6축만)
                last_q = np.asarray(configuration.q[:6], dtype=float).copy()
            
            except Exception as e:
                print("[mink IK] Failed -> keep last_q:", repr(e))

            # --- Skeleton render (use last_q) ---
            joints_xyz = fk.fk_all_joint_positions(last_q)
            teleoperator.tv.set_robot_joints(joints_xyz)
            # teleoperator.tv.set_robot_joints(test_joints)

            
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
                    print(f"[SEND GRIP] pos={gripper_pos}")
                    driver.set_gripper(position=grip_um, effort=2000, enable=True)
            
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
