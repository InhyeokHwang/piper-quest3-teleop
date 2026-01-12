# teleop/app.py
import time
import numpy as np
import mujoco
import mink

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
from .control.ik_stepper import ik_step

from .control.sender import piper_send_jointctrl
# from .control.sender import piper_send_mit  # (MIT 테스트용)

from .piper.driver import PiperDriver
from .piper.safety import enable_and_wait

## 반환 값은 run_loop로 들어감 -> 실행에 필요한 모든 객체를 초기화
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
    
    # 시작 조인트 0 
    for k in range(6):
        j = model.joint(f"joint{k+1}")
        adr = int(np.asarray(j.qposadr).item())
        data.qpos[adr] = 0.0
        vadr = int(np.asarray(j.dofadr).item())
        data.qvel[vadr] = 0.0
    mujoco.mj_forward(model, data)

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

        driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=50, is_mit_mode=0x00)  # joint mode
        # driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x04, speed=100, is_mit_mode=0xAD) # mit mode
        
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
        startup_sent_zero=False, # 시작할 때 0 쏘기
        sent_joint_zero=False, # returning 이후에 0 쏘기
        hold_target=None,
        mode="RETURNING"
    )
    return rt


def joints123_near_zero(q, tol_deg=5.0):
    tol = np.deg2rad(tol_deg)
    q = np.asarray(q).reshape(-1)
    return np.all(np.abs(q[:3]) <= tol)

def reset_to_zero_like_init(rt, q_zero: np.ndarray):
    q_zero = np.asarray(q_zero, dtype=float).reshape(6,)

    rt.last_q[:6] = q_zero.copy()

    # mujoco state를 먼저 0으로 (FK/Jacobian 기준)
    for k in range(6):
        j = rt.model.joint(f"joint{k+1}")
        adr = int(np.asarray(j.qposadr).item())
        rt.data.qpos[adr] = float(q_zero[k])
        vadr = int(np.asarray(j.dofadr).item())
        rt.data.qvel[vadr] = 0.0
    mujoco.mj_forward(rt.model, rt.data)

    # mink configuration을 새로 생성 (과거 상태 싹 제거)
    rt.configuration = mink.Configuration(rt.model)

    q_full = rt.configuration.q.copy()
    q_full[:6] = q_zero
    rt.configuration.q[:] = q_full

    # tasks도 새로 생성
    ee_task = mink.FrameTask(
        frame_name=config.MINK_EE_SITE,  
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.3,
        lm_damping=float(getattr(config, "MINK_LM_DAMPING", 1e-6)),
    )
    posture_task = mink.PostureTask(
        rt.model,
        cost=float(getattr(config, "MINK_POSTURE_COST", 1e-3))
    )
    rt.tasks = [ee_task, posture_task]

    # posture target = q_zero로 확정
    q_rest_full = rt.configuration.q.copy()
    q_rest_full[:6] = q_zero
    posture_task.set_target(q_rest_full)

    # 필터도 같이 초기화
    rt.T_filt = None
    rt.vel_filt = None

def run_loop(args, rt: RuntimeContext):

    send_hz = float(getattr(config, "SEND_RATE_HZ", 60.0))
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

                # ---- MIT로 바꿔보고 싶으면 아래로 교체 ----
                # next_send = piper_send_mit(
                #     getattr(rt, "driver", None),
                #     args.dry_run,
                #     rt.last_q,
                #     0,
                #     next_send,
                #     send_period,
                #     kp=5.0, kd=0.1, tau_ref=0.0
                # )

                rt.startup_sent_zero = True
                rt.rate.sleep()
                continue  # 첫 프레임은 IK/모드로직 안 탐

            loop_t0 = time.time()

            # vr로부터 오른손 컨트롤러 정보 수신
            right_pose = rt.teleoperator.step()

            # ---------- LEFT input debug (thumbstick + trigger + squeeze + X/Y) ----------
            ls = rt.teleoperator.left_state  # shape (14,)

            # raw thumbstick (Quest 기준)
            lx_raw = float(ls[10])   # +right
            ly_raw = float(ls[11])   # +down

            # trigger / squeeze / buttons (bool + analog)
            tr_btn = bool(ls[0])
            sq_btn = bool(ls[1])
            x_btn  = bool(ls[4])
            y_btn  = bool(ls[5])

            tr_val = float(ls[6])    # 0..1
            sq_val = float(ls[7])    # 0..1

            # deadband on stick
            deadband = 0.15
            lx = 0.0 if abs(lx_raw) < deadband else lx_raw
            ly = 0.0 if abs(ly_raw) < deadband else ly_raw

            # Vision60 command mapping
            v_forward = -ly
            w_yaw     = -lx

            # scale
            MAX_V = 0.6   # m/s
            MAX_W = 1.2   # rad/s

            cmd_v = MAX_V * v_forward
            cmd_w = MAX_W * w_yaw

            # 출력 조건:
            #  - 이동 명령이 있거나
            #  - trigger / squeeze / X / Y 중 하나라도 눌렸을 때
            active = (
                abs(cmd_v) > 1e-3 or
                abs(cmd_w) > 1e-3 or
                tr_val > 0.05 or
                sq_val > 0.05 or
                x_btn or
                y_btn
            )

            # 10Hz 출력 제한
            now = time.monotonic()
            if active and (now - getattr(rt, "_last_left_print_t", 0.0) > 0.10):
                print(
                    f"[LEFT] "
                    f"stick(x={lx:+.2f}, y={ly:+.2f}) | "
                    f"cmd(v={cmd_v:+.2f} m/s, w={cmd_w:+.2f} rad/s) | "
                    f"trigger={int(tr_btn)}({tr_val:.2f}) "
                    f"squeeze={int(sq_btn)}({sq_val:.2f}) "
                    f"X={int(x_btn)} Y={int(y_btn)}"
                )
                rt._last_left_print_t = now
            # ---------------------------------------------------------------------------


            # 그리퍼
            g = step_gripper(rt.gripper_ctl, rt.teleoperator)   
            
            ##################### 네 종류 MODE ######################
            ## RETURNING - zero position으로 돌아가는 상태
            ## AT_ZERO - zero position에 도착한 상태
            ## HOLD - 자세 유지
            ## TELEOP - squeeze 버튼을 누르고 있는 상태 (TELEOP중인 상태)

            # squeeze (teleop)
            sq_out = rt.squeeze_ctl.update(rt.teleoperator)

            # ret_zero
            ret_out = rt.ret_zero_ctl.update(rt.teleoperator)

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
                if sq_out.just_pressed:
                    # quest 쪽 스켈레톤 anchor는 지금 컨트롤러 위치로 리셋
                    controller_anchor = rt.teleoperator.tv.right_controller[:3, 3].copy()
                    rt.teleoperator.tv.enable_skeleton(controller_anchor)

                    q_zero6 = np.zeros(6, dtype=float)
                    reset_to_zero_like_init(rt, q_zero6)

                    # neutral 잡기
                    rt.mapper.set_neutral(rt.T_zero, right_pose)

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
                    rt.sent_joint_zero = False
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

            
            ############### IK step #####################
            if rt.mode != "AT_ZERO":
                try:
                    dt = float(rt.rate.dt)

                    rt.last_q = ik_step(
                        rt.model, rt.data, rt.configuration,
                        rt.tasks, rt.limits, rt.solver, dt,
                        rt.last_q, target_T,
                        g.joint7, g.joint8, rt.q_idx7, rt.q_idx8,
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
                g.grip_um,
                next_send,
                send_period,
                config.RAD_TO_PIPER
            )

            # ---- MIT로 바꿔보고 싶으면 아래로 교체 ----
            # next_send = piper_send_mit(
            #     getattr(rt, "driver", None),
            #     args.dry_run,
            #     rt.last_q,
            #     g.grip_um,
            #     next_send,
            #     send_period,
            #     kp=5.0, kd=0.1, tau_ref=0.0
            # )
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
