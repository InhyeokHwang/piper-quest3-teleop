# teleop/app.py
import time
import numpy as np
import mujoco

from .VuerTeleop import VuerTeleop
from . import config
from .runtime.context import RuntimeContext
from .runtime.init_camera import init_camera
from .runtime.init_fk_mapper import init_fk_and_start, init_mapper
from .runtime.init_mink import init_mink
from .runtime.init_viewer import init_viewer
from .runtime.init_driver import init_driver
from .runtime.init_ros2_bridge import init_ros2_bridge
from .runtime.init_mujoco import init_T_zero
from .runtime.init_mujoco import init_mujoco_state_zero
from .runtime.init_mujoco import init_gripper_indices
from .runtime.init_left_controller import LeftController, LeftControllerConfig
from .runtime.init_right_controller import RightController, RightControllerConfig

from .utils.joint123_near_zero import joints123_near_zero
from .utils.runtime_reset import reset_to_zero_like_init
from .control.ik_stepper import ik_step

from .control.sender import piper_send_jointctrl


## 반환 값은 run_loop로 들어감 -> 실행에 필요한 모든 객체를 초기화
def build_runtime(args) -> RuntimeContext:
    teleoperator = VuerTeleop()
    left = LeftController(LeftControllerConfig(
        deadband=0.15,
        max_v=0.6,
        max_w=1.2,
        pub_hz=20.0,
    ))
    right = RightController(RightControllerConfig(
        gripper_mode="analog",
        gripper_out_min=0,
        gripper_out_max=1000,
        gripper_close_when_high=True,
        gripper_alpha=0.35,
        idx_squeeze_pressed=1,
        idx_return_pressed=4,
    ))
    right.reset(open_gripper=True)

    vision_rclpy, vision_node = init_ros2_bridge(args.vision60)
    cam = init_camera(teleoperator, args.camera) # camera

    fk, q_zero, EE_START, R_ee0 = init_fk_and_start() # EE_START는 zero pos에 대한 x y z, R_ee0는 zero pos에 대한 rotation mat
    mapper = init_mapper(EE_START, R_ee0, debug=args.debug_mapper)
    model, data, configuration, tasks, limits, solver, rate = init_mink(q_zero)
    viewer = init_viewer(model, data, args.dry_run)

    T_zero = init_T_zero(EE_START, R_ee0)
    init_mujoco_state_zero(model, data)
    mujoco.mj_forward(model, data)

    # gripper indices
    q_idx7, q_idx8 = init_gripper_indices(model)
    
    driver = init_driver()

    # store driver inside teleoperator or return separately
    rt = RuntimeContext(
        teleoperator=teleoperator, 
        cam=cam,
        fk=fk, mapper=mapper,
        model=model, data=data, configuration=configuration,
        tasks=tasks, limits=limits, solver=solver, rate=rate,
        viewer=viewer, 
        T_zero=T_zero,
        last_q=q_zero.copy(), T_filt=None, vel_filt=None,
        q_idx7=q_idx7, q_idx8=q_idx8,
        driver=driver,
        startup_sent_zero=False, # 시작할 때 0 쏘기
        sent_joint_zero=False, # returning 이후에 0 쏘기
        hold_target=None,
        mode="RETURNING",
        vision_rclpy=vision_rclpy,
        vision_node=vision_node,
        left=left,
        right=right,
    )
    return rt


def run_loop(args, rt: RuntimeContext):
    send_hz = float(getattr(config, "SEND_RATE_HZ", 100.0))
    send_period = 1.0 / max(send_hz, 1e-6)
    next_send = time.monotonic()

    try:
        while True:
            ## viewer가 살아있는지 체크
            if rt.viewer is not None and not rt.viewer.is_running():
                break

            if not rt.startup_sent_zero:
                rt.last_q[:6] = 0.0

                # viewer 상태도 유지
                for k in range(6):
                    j = rt.model.joint(f"joint{k+1}")
                    adr = int(np.asarray(j.qposadr).item())
                    rt.data.qpos[adr] = 0.0
                    vadr = int(np.asarray(j.dofadr).item())
                    rt.data.qvel[vadr] = 0.0
                mujoco.mj_forward(rt.model, rt.data)

                # 바로 송신 (dry-run이면 출력)
                next_send = piper_send_jointctrl(
                    getattr(rt, "driver", None),
                    args.dry_run,
                    rt.last_q,
                    0,  # grip은 지금 값 써도 됨
                    next_send,
                    send_period,
                    config.RAD_TO_PIPER
                )

                rt.startup_sent_zero = True
                rt.rate.sleep()
                continue  # 첫 프레임은 IK/모드로직 안 탐

            loop_t0 = time.time()

            # vr로부터 오른손 컨트롤러 정보 수신
            right_pose_T = rt.teleoperator.step()


            # LEFT Controller -> vision60 command + publish + toggle까지 내부에서 처리
            rt.left.update(
                rt.teleoperator.left_state,
                vision_node=rt.vision_node,
                vision_rclpy=rt.vision_rclpy,
            )
            
            # RIGHT Controller
            r = rt.right.update(rt.teleoperator)
            grip_um = r.grip_out
            squeeze_holding = r.holding
            squeeze_just_pressed = r.just_pressed

            ##################### 네 종류 MODE ######################
            ## RETURNING - zero position으로 돌아가는 상태
            ## AT_ZERO - zero position에 도착한 상태
            ## HOLD - 자세 유지
            ## TELEOP - squeeze 버튼을 누르고 있는 상태 (TELEOP중인 상태)

            # 현재 로봇 EE pose 얻기
            T_cur = rt.fk.compute_fk(rt.last_q)

            if rt.mode == "RETURNING":
                if joints123_near_zero(rt.last_q, tol_deg=5.0):
                    rt.mode = "AT_ZERO"
                    rt.mapper.reset_state(keep_neutral_target=False)

                    if not rt.sent_joint_zero:
                        q_zero6 = np.zeros(6, dtype=float)
                        reset_to_zero_like_init(rt, q_zero6)
                        rt.sent_joint_zero = True
                target_T = rt.T_zero

            elif rt.mode == "AT_ZERO":
                target_T = rt.T_zero

                # squeeze 누름
                if squeeze_just_pressed:
                    # quest 쪽 스켈레톤 anchor는 지금 컨트롤러 위치로 리셋
                    controller_anchor = rt.teleoperator.tv.right_controller[:3, 3].copy()
                    rt.teleoperator.tv.enable_skeleton(controller_anchor)

                    q_zero6 = np.zeros(6, dtype=float)
                    reset_to_zero_like_init(rt, q_zero6)

                    # neutral 잡기
                    rt.mapper.set_neutral(rt.T_zero, right_pose_T)

                    rt.mode = "TELEOP"
                    rt.rate.sleep()
                    continue

            elif rt.mode == "HOLD":
                target_T = rt.hold_target if rt.hold_target is not None else T_cur

                # squeeze 다시 누르면 TELEOP 재진입
                if squeeze_just_pressed:
                    controller_anchor = rt.teleoperator.tv.right_controller[:3, 3].copy()
                    rt.teleoperator.tv.enable_skeleton(controller_anchor)

                    rt.mapper.set_neutral(target_T, right_pose_T)
                    rt.T_filt = None
                    rt.vel_filt = None
                    rt.mode = "TELEOP"
                    rt.rate.sleep()
                    continue

                # RETURN 버튼 눌렀을 때만 복귀
                if r.go_to_zero:
                    rt.mode = "RETURNING"
                    rt.T_filt = None
                    rt.vel_filt = None
                    rt.sent_joint_zero = False
                    if hasattr(rt.mapper, "neutral_target_T"):
                        rt.mapper.neutral_target_T = None

            elif rt.mode == "TELEOP":
                # TELEOP일 때만 컨트롤러 매핑
                target_T = rt.mapper.compute_target_T(right_pose_T)
                # squeeze release 감지되면 HOLD로 전환
                if not squeeze_holding:
                    rt.teleoperator.tv.clear_robot_joints()
                    rt.hold_target = T_cur.copy()
                    rt.mode = "HOLD"
                    rt.T_filt = None
                    rt.vel_filt = None
            ###################################################

            
            ############### IK step #####################
            if rt.mode != "AT_ZERO":
                try:
                    dt = float(rt.rate.dt)

                    rt.last_q = ik_step(
                        rt.model, rt.data, rt.configuration,
                        rt.tasks, rt.limits, rt.solver, dt,
                        rt.last_q, target_T,
                        grip_um, rt.q_idx7, rt.q_idx8,
                        debug_qpos_check=False
                    )
                except Exception as e:
                    print("[mink IK] Failed -> keep last_q:", repr(e))
            ##############################################


            ############ skeleton render #################
            if rt.mode == "TELEOP":
                joints_xyz = rt.fk.fk_all_joint_positions(rt.last_q)
                rt.teleoperator.tv.set_robot_joints(joints_xyz)
            ##############################################

            ############## send to robot ##################
            next_send = piper_send_jointctrl(
                getattr(rt, "driver", None),
                args.dry_run,
                rt.last_q,
                grip_um,
                next_send,
                send_period,
                config.RAD_TO_PIPER
            )

            # 카메라
            if rt.cam is not None:
                rt.cam.step()

            if args.print_freq:
                dtp = max(time.time() - loop_t0, 1e-9)
                print("[Loop] freq:", 1.0 / dtp)

            if rt.viewer is not None:
                rt.viewer.sync()

            rt.rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        try:
            rt.close()
        except Exception:
            pass