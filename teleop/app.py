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

from .control.gripper_stepper import step_gripper
from .control.smoothing import make_pose_params, make_vel_params, smooth_target
from .control.ik_stepper import ik_step
from .control.sender import maybe_send

from .piper.driver import PiperDriver
from .piper.safety import enable_and_wait, move_to_start_pose

def build_runtime(args) -> RuntimeContext:
    teleoperator = VuerTeleop(args.config)
    cam = init_camera(teleoperator, args.camera) #카메라 켜기, 없다면 None

    fk, q_zero, EE_START, R_ee0 = init_fk_and_start() # EE_START는 zero pos에 대한 x y z, R_ee0는 zero pos에 대한 rotation mat
    # build T_zero (4x4)
    T_zero = np.eye(4, dtype=float)
    T_zero[:3, :3] = R_ee0
    T_zero[:3, 3]  = np.asarray(EE_START, dtype=float).reshape(3,)

    squeeze_ctl = init_squeeze()

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
        viewer=viewer, gripper_ctl=gripper_ctl, squeeze_ctl=squeeze_ctl,
        T_zero=T_zero,
        last_q=q_zero.copy(), T_filt=None, vel_filt=None,
        q_idx7=q_idx7, q_idx8=q_idx8,
        driver=driver,
        target_T=None,
    )
    return rt

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

            right_pose = rt.teleoperator.step()           # 여기서 pose7 (x,y,z,qx,qy,qz,qw)
            g = step_gripper(rt.gripper_ctl, rt.teleoperator)  
            sq = rt.squeeze_ctl.update(rt.teleoperator)

                
            # 1) squeeze press: neutral 재설정 (필요하면 1프레임 스킵)
            if sq.just_pressed:
                rt.mapper.set_neutral_from_pose7(right_pose)
                # 기준 잡는 프레임은 스킵(안정적). 원치 않으면 continue 제거해도 됨.
                rt.rate.sleep()
                continue

            # 2) 기본 target 계산
            target_T = rt.mapper.compute_target_T(right_pose)

            # 3) squeeze release: zero pose가 최우선 (이 아래 다른 분기보다 먼저!)
            if sq.go_to_zero:
                target_T = rt.T_zero
            elif sq.target_T is not None:
                target_T = sq.target_T
            else:
                if getattr(rt.squeeze_ctl.cfg, "hold_to_teleop", False) and not sq.holding:
                    target_T = None

            if target_T is None:
                if rt.viewer is not None:
                    rt.data.qpos[:6] = rt.last_q
                    rt.data.qpos[rt.q_idx7] = g.joint7
                    rt.data.qpos[rt.q_idx8] = g.joint8
                    mujoco.mj_forward(rt.model, rt.data)
                    rt.viewer.sync()
                rt.rate.sleep()
                continue

            rt.T_filt = smooth_target(rt.T_filt, target_T, pose_params)
            target_T_use = rt.T_filt

            # IK step
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

            # skeleton render
            joints_xyz = rt.fk.fk_all_joint_positions(rt.last_q)
            rt.teleoperator.tv.set_robot_joints(joints_xyz)

            # send to robot
            next_send = maybe_send(
                getattr(rt, "driver", None),
                args.dry_run,
                rt.last_q,
                g.grip_um,
                next_send,
                send_period,
                config.RAD_TO_PIPER
            )

            if rt.cam is not None:
                rt.cam.step()

            if args.print_freq:
                dtp = max(time.time() - loop_t0, 1e-9)
                print("[Loop] freq:", 1.0 / dtp)

            if rt.viewer is not None:
                rt.viewer.sync()

            rt.rate.sleep()

    except KeyboardInterrupt:
        print("\n[Main] Interrupted")
        raise
