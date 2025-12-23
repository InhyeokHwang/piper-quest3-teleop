# robot_skeleton_vuer.py
import numpy as np
from vuer.schemas import group, Sphere, Cylinder

# ---------------------------
# math helpers
# ---------------------------
## 로드리게스 회전
def _rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = np.cos(angle); s = np.sin(angle); C = 1.0 - c
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C  ],
    ], dtype=float)

def _cylinder_pose(p0, p1, local_axis=np.array([0.0, 1.0, 0.0])):
    """
    두 점 p0->p1을 잇는 실린더의 (T, length) 반환.
    - 실린더의 로컬축(local_axis)을 링크 방향으로 회전
    - 위치는 중점(mid)
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    T = np.eye(4, dtype=float)

    if L < 1e-9:
        T[:3, 3] = p0
        return T, 0.0

    d = v / L
    a = np.asarray(local_axis, float)
    a /= (np.linalg.norm(a) + 1e-12)

    dot = float(np.clip(np.dot(a, d), -1.0, 1.0))
    if dot > 0.999999:
        R = np.eye(3, dtype=float)
    elif dot < -0.999999:
        # 180도 회전: a와 직교하는 축 하나 잡기
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, tmp)
        R = _rodrigues(axis, np.pi)
    else:
        axis = np.cross(a, d)
        angle = np.arccos(dot)
        R = _rodrigues(axis, angle)

    mid = (p0 + p1) * 0.5
    T[:3, :3] = R
    T[:3, 3] = mid
    return T, L


# ---------------------------
# renderer
# ---------------------------

class VuerRobotSkeleton:
    """
    Vuer 씬에 로봇 스켈레톤(관절=Sphere, 링크=Cylinder)을 업서트하는 유틸.
    - joints_xyz: (N,3) in Vuer/world coordinates
    - edges: [(parent, child), ...]
    """
    def __init__(
        self,
        edges,
        key="robot-skel",
        joint_radius=0.015,
        link_radius=0.008,
        cylinder_local_axis=(0.0, 1.0, 0.0),
        layers=0,
        offset=(0.0, 0.0, 0.0)
    ):
        self.edges = list(edges)
        self.key = key
        self.joint_radius = float(joint_radius)
        self.link_radius = float(link_radius)
        self.cyl_axis = np.array(cylinder_local_axis, dtype=float)
        self.layers = layers
        self.offset = np.array(offset, dtype=float)


    def build_elements(self, joints_xyz):
        joints_xyz = np.asarray(joints_xyz, dtype=float)
        joints_xyz = joints_xyz + self.offset
        
        elems = []

        # joints
        for i, p in enumerate(joints_xyz):
            elems.append(
                Sphere(
                    args=(self.joint_radius, 16, 12),
                    position=p.tolist(),
                    key=f"{self.key}:joint:{i}",
                    layers=self.layers,
                )
            )

        # links
        for (i, j) in self.edges:
            T, L = _cylinder_pose(joints_xyz[i], joints_xyz[j], local_axis=self.cyl_axis)
            if L <= 1e-9:
                continue
            # Cylinder args: (radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength)
            elems.append(
                Cylinder(
                    args=(self.link_radius, self.link_radius, float(L), 12, 1, False, 0.0, 6.28318),
                    matrix=T.flatten(order="F").tolist(),
                    key=f"{self.key}:link:{i}-{j}",
                    layers=self.layers,
                )
            )

        return elems

    def upsert(self, session, joints_xyz):
        elems = self.build_elements(joints_xyz)
        session.upsert @ group(children=elems, key=self.key)
