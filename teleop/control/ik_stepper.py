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
    # configuration에 현재 q 반영 (앞 6개만 arm)
    q_full = configuration.q.copy()
    q_full[:6] = last_q
    configuration.q[:] = q_full

    # 매 스탭마다 fk/jacobian 갱신하고 configuration 다시 반영
    data.qpos[:] = configuration.q
    mujoco.mj_forward(model, data)

    if debug_qpos_check:
        print(f"[QPOS CHECK] q4={float(data.qpos[3]):.5f} q7={float(data.qpos[q_idx7]):.5f} q8={float(data.qpos[q_idx8]):.5f}")

    # target pose를 mink.SE3로 변환해서 task 타겟 업데이트
    T_wt = mink.SE3.from_matrix(target_T_use)
    tasks[0].set_target(T_wt)  # end_effector_task

    # solve_ik -> velocity, integrate
    vel = mink.solve_ik(configuration, tasks, dt, solver, limits=limits)

    vel_filt = None
    configuration.integrate_inplace(vel, dt)

    # MuJoCo data에 qpos 반영
    data.qpos[:] = configuration.q

    # 그리퍼는 data.qpos에 마지막으로 강제 주입 (핵심!)
    data.qpos[q_idx7] = float(joint7)
    data.qpos[q_idx8] = float(joint8)

    # 다시 configuration에 동기화
    configuration.q[:] = data.qpos
    mujoco.mj_forward(model, data)

    new_last_q = np.asarray(configuration.q[:6], dtype=float).copy()
    return new_last_q, vel_filt
