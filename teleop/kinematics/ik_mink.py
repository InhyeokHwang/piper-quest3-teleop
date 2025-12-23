# ik_mink.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mujoco
import mink

@dataclass
class MinkIKConfig:
    xml_path: str
    ee_frame_name: str = "attachment_site"   # site or body name
    ee_frame_type: str = "site"             # "site" or "body"
    solver: str = "daqp"
    dt: float = 0.01
    lm_damping: float = 1e-6
    posture_cost: float = 1e-3

class MinkIK:
    def __init__(self, cfg: MinkIKConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data

        # Tasks
        self.ee_task = mink.FrameTask(
            frame_name=cfg.ee_frame_name,
            frame_type=cfg.ee_frame_type,
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=cfg.lm_damping,
        )
        self.posture_task = mink.PostureTask(self.model, cost=cfg.posture_cost)
        self.tasks = [self.ee_task, self.posture_task]

        # Limits (필요하면 추가)
        self.limits = [mink.ConfigurationLimit(model=self.model)]

        # 초기 자세(q_rest)를 home에서 잡거나, 네가 원하는 q_rest를 넣어도 됨
        # 예: keyframe이 있으면
        try:
            self.configuration.update_from_keyframe("home")
        except Exception:
            pass
        self.posture_task.set_target(self.configuration.q.copy())

    def set_q(self, q: np.ndarray):
        """외부(실로봇) q를 시뮬 상태로 반영"""
        self.configuration.q[:] = q
        self.configuration.update()

    def step(self, target_T_w: mink.SE3):
        """target SE3를 향해 한 스텝 IK -> q 업데이트"""
        self.ee_task.set_target(target_T_w)
        vel = mink.solve_ik(
            self.configuration,
            self.tasks,
            self.cfg.dt,
            self.cfg.solver,
            limits=self.limits,
        )
        self.configuration.integrate_inplace(vel, self.cfg.dt)
        return self.configuration.q.copy()
