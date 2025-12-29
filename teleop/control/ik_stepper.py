# teleop/control/ik_stepper.py
import numpy as np
import mujoco
import mink

def ik_step(
    model, data, configuration,
    tasks, limits, solver, dt,
    last_q: np.ndarray,
    target_T_use,
    joint7: float, joint8: float,
    q_idx7: int, q_idx8: int,
    vel_filt, smooth_vel_fn,
    debug_qpos_check: bool = False,
):
    # (A) configuration에 현재 q 반영 (앞 6개만 arm)
    q_full = configuration.q.copy()
    q_full[:6] = last_q
    configuration.q[:] = q_full

    # (B) mujoco forward로 kinematics 업데이트
    data.qpos[:] = configuration.q
    mujoco.mj_forward(model, data)

    if debug_qpos_check:
        print(f"[QPOS CHECK] q7={float(data.qpos[q_idx7]):.5f} q8={float(data.qpos[q_idx8]):.5f}")

    # (C) target pose를 mink.SE3로 변환해서 task 타겟 업데이트
    T_wt = mink.SE3.from_matrix(target_T_use)
    tasks[0].set_target(T_wt)  # end_effector_task

    # (D) solve_ik -> velocity, integrate
    vel = mink.solve_ik(configuration, tasks, dt, solver, limits=limits)

    # ✅ 지금은 분리 안정화가 목표라 smoothing은 "끄는" 걸 추천
    # vel_filt = smooth_vel_fn(vel, vel_filt, dt)
    # configuration.integrate_inplace(vel_filt if vel_filt is not None else vel, dt)
    vel_filt = None
    configuration.integrate_inplace(vel, dt)

    # (E) MuJoCo data에 qpos 반영
    data.qpos[:] = configuration.q

    # (F) 그리퍼는 data.qpos에 마지막으로 강제 주입 (핵심!)
    data.qpos[q_idx7] = float(joint7)
    data.qpos[q_idx8] = float(joint8)

    # (G) 다시 configuration에 동기화
    configuration.q[:] = data.qpos
    mujoco.mj_forward(model, data)

    new_last_q = np.asarray(configuration.q[:6], dtype=float).copy()
    return new_last_q, vel_filt
