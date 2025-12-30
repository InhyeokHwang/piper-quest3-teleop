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
from .runtime.init_gripper import init_gripper
from .runtime.init_squeeze import init_squeeze
from .runtime.return_zero_pos_init import init_return_zero_pos


from .control.gripper_stepper import step_gripper
from .control.smoothing import make_pose_params, make_vel_params, smooth_target
from .control.ik_stepper import ik_step
from .control.sender import piper_send

from .piper.driver import PiperDriver
from .piper.safety import enable_and_wait

def build_runtime(args) -> RuntimeContext:
    teleoperator = VuerTeleop(args.config)
    cam = init_camera(teleoperator, args.camera) #카메라 켜기, 없다면 None

    fk, q_zero, EE_START, R_ee0 = init_fk_and_start() # EE_START는 zero pos에 대한 x y z, R_ee0는 zero pos에 대한 rotation mat
    # build T_zero (4x4)
    T_zero = np.eye(4, dtype=float)
    T_zero[:3, :3] = R_ee0
    T_zero[:3, 3]  = np.asarray(EE_START, dtype=float).reshape(3,)

    squeeze_ctl = init_squeeze()
    ret_zero_ctl = init_return_zero_pos()

    mapper = init_mapper(EE_START, R_ee0, debug=args.debug_mapper)

    model, data, configuration, tasks, limits, solver, rate = init_mink(q_zero)
    viewer = init_viewer(model, data, args.dry_run)

    # gripper indices
    j7 = model.joint("joint7")
    j8 = model.joint("joint8")
    q_idx7 = int(np.asarray(j7.qposadr).item())
    q_idx8 = int(np.asarray(j8.qposadr).item())

    gripper_ctl = init_gripper()

    # driver
    driver = None
    if not args.dry_run:
        driver = PiperDriver(args.can)
        driver.connect()
        enable_and_wait(driver, timeout_s=5.0, fail_hard=True, also_open_gripper=True)
        driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=100, acc=0x00)
        driver.set_gripper(position=20000, effort=2000, enable=True, clear_error=True)
        print("[Piper] Ready.")
    else:
        print("[DRY RUN] No hardware commands will be sent.")

    # store driver inside teleoperator or return separately
    rt = RuntimeContext(
        teleoperator=teleoperator, cam=cam,
        fk=fk, mapper=mapper,
        model=model, data=data, configuration=configuration,
        tasks=tasks, limits=limits, solver=solver, rate=rate,
        viewer=viewer, gripper_ctl=gripper_ctl, squeeze_ctl=squeeze_ctl,ret_zero_ctl=ret_zero_ctl,
        T_zero=T_zero,
        last_q=q_zero.copy(), T_filt=None, vel_filt=None,
        q_idx7=q_idx7, q_idx8=q_idx8,
        driver=driver,
        target_T=None,
    )
    return rt

def pose_reached(T_cur, T_goal, pos_tol=0.03):
    return np.linalg.norm(T_cur[:3,3] - T_goal[:3,3]) <= pos_tol

def run_loop(args, rt: RuntimeContext):
    pose_params = make_pose_params()
    vel_params = make_vel_params()

    send_hz = float(getattr(config, "SEND_RATE_HZ", 60.0))
    send_period = 1.0 / max(send_hz, 1e-6)
    next_send = time.monotonic()

    try:
        while True:
            if rt.viewer is not None and not rt.viewer.is_running():
                break

            loop_t0 = time.time()

            # vr로부터 오른손 컨트롤러 정보 수신
            right_pose = rt.teleoperator.step()
            # 그리퍼
            g = step_gripper(rt.gripper_ctl, rt.teleoperator)   
            
            ##################### 네 종류 MODE ######################
            ## RETURNING - zero position으로 돌아가는 상태
            ## AT_ZERO - zero position에 도착한 상태
            ## HOLD - 자세 유지
            ## TELEOP - squeeze 버튼을 누르고 있는 상태 (TELEOP중인 상태)
            if not hasattr(rt, "mode"):
                rt.mode = "RETURNING"
            if not hasattr(rt, "hold_target"):
                rt.hold_target = None

            # squeeze (teleop)
            sq_out = rt.squeeze_ctl.update(rt.teleoperator)

            # ret_zero
            ret_out = rt.ret_zero_ctl.update(rt.teleoperator)


            # 현재 로봇 EE pose 얻기
            T_cur = rt.fk.compute_fk(rt.last_q)

            if rt.mode == "RETURNING":
                if pose_reached(T_cur, rt.T_zero, pos_tol=0.03):
                    rt.mode = "AT_ZERO"
                    rt.T_filt = None
                    rt.vel_filt = None
                target_T = rt.T_zero

            elif rt.mode == "AT_ZERO":
                target_T = rt.T_zero

                if sq_out.just_pressed:
                    controller_anchor = rt.teleoperator.tv.right_controller[:3, 3].copy()
                    rt.teleoperator.tv.enable_skeleton(controller_anchor)

                    rt.mapper.set_neutral(rt.T_zero, right_pose)
                    rt.T_filt = None
                    rt.vel_filt = None
                    rt.mode = "TELEOP"
                    rt.rate.sleep()
                    continue

            elif rt.mode == "HOLD":
                target_T = rt.hold_target if rt.hold_target is not None else T_cur

                # squeeze 다시 누르면 TELEOP 재진입
                if sq_out.just_pressed:
                    controller_anchor = rt.teleoperator.tv.right_controller[:3, 3].copy()
                    rt.teleoperator.tv.enable_skeleton(controller_anchor)

                    rt.mapper.set_neutral(target_T, right_pose)
                    rt.T_filt = None
                    rt.vel_filt = None
                    rt.mode = "TELEOP"
                    rt.rate.sleep()
                    continue

                # RETURN 버튼 눌렀을 때만 복귀
                if ret_out.go_to_zero:
                    rt.mode = "RETURNING"
                    rt.T_filt = None
                    rt.vel_filt = None
                    if hasattr(rt.mapper, "neutral_target_T"):
                        rt.mapper.neutral_target_T = None

            elif rt.mode == "TELEOP":
                # TELEOP일 때만 컨트롤러 매핑
                target_T = rt.mapper.compute_target_T(right_pose)
                # squeeze release 감지되면 HOLD로 전환
                if not sq_out.holding:
                    rt.teleoperator.tv.clear_robot_joints()
                    rt.hold_target = T_cur.copy()
                    rt.mode = "HOLD"
                    rt.T_filt = None
                    rt.vel_filt = None
            ###################################################

            rt.T_filt = smooth_target(rt.T_filt, target_T, pose_params)
            target_T_use = rt.T_filt
            
            ############### IK step #####################
            try:
                dt = rt.rate.dt
                def _smooth_vel(vel, vel_filt, dt):
                    # 여기서 vel_params를 닫아둠
                    from .control.smoothing import smooth_vel
                    return smooth_vel(vel, vel_filt, dt, vel_params)

                rt.last_q, rt.vel_filt = ik_step(
                    rt.model, rt.data, rt.configuration,
                    rt.tasks, rt.limits, rt.solver, dt,
                    rt.last_q, target_T_use,
                    g.joint7, g.joint8, rt.q_idx7, rt.q_idx8,
                    rt.vel_filt, _smooth_vel
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
            next_send = piper_send(
                getattr(rt, "driver", None),
                args.dry_run,
                rt.last_q,
                g.grip_um,
                next_send,
                send_period,
                config.RAD_TO_PIPER
            )
            ###############################################

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
        raise
