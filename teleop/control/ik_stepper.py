# teleop/control/ik_stepper.py
import numpy as np
import mujoco
import mink

def ik_step(
    model, data, configuration,
    tasks, limits, solver, dt,
    last_q: np.ndarray,
    target_T_use,
    grip_um: int,
    q_idx7: int, q_idx8: int,
    debug_qpos_check: bool = False,
    dt_min: float = 1e-4,
    dt_max: float = 1/30,   # 33ms 이상 튀면 불안정해지기 쉬워서 클램프
):
    # dt 안정화
    dt = float(dt)
    if not np.isfinite(dt):
        dt = dt_max
    dt = max(dt_min, min(dt, dt_max))

    # configuration에 현재 arm q 반영
    last_q = np.asarray(last_q, dtype=float).reshape(6,)
    q_full = configuration.q.copy()
    q_full[:6] = last_q
    configuration.q[:] = q_full

    # MuJoCo 상태 동기화 + forward (FK/Jacobian 갱신) 
    data.qpos[:] = configuration.q
    # gripper도 여기서 같이 넣어두면, FK/Jacobian이 그리퍼 포함 모델일 때 일관성이 좋아짐
    joint7 = float(grip_um)
    joint8 = float(grip_um)
    data.qpos[q_idx7] = float(joint7)
    data.qpos[q_idx8] = float(joint8)
    mujoco.mj_forward(model, data)

    if debug_qpos_check:
        print(f"[QPOS CHECK] q4={float(data.qpos[3]):.5f} q7={float(data.qpos[q_idx7]):.5f} q8={float(data.qpos[q_idx8]):.5f}")

    # task 타겟 업데이트
    T_wt = mink.SE3.from_matrix(target_T_use)
    tasks[0].set_target(T_wt)

    # 속도 스무딩 적용
    vel = mink.solve_ik(configuration, tasks, dt, solver, limits=limits)
    configuration.integrate_inplace(vel, dt)

    #  MuJoCo qpos에 최종 반영 + forward
    data.qpos[:] = configuration.q
    data.qpos[q_idx7] = float(joint7)
    data.qpos[q_idx8] = float(joint8)

    # configuration에도 최종 반영(일관성 유지)
    configuration.q[:] = data.qpos

    mujoco.mj_forward(model, data)

    new_last_q = np.asarray(configuration.q[:6], dtype=float).copy()
    return new_last_q
