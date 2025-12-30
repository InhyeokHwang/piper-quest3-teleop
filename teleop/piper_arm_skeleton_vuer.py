# robot_skeleton_vuer.py
import numpy as np
from vuer.schemas import group, Sphere, Cylinder

def _quat_from_two_vec(a, b):
    # a,b: (3,) unit vectors
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    w = 1.0 + float(np.dot(a, b))
    if w < 1e-8:
        # 180도: a와 직교축 하나 잡기
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, tmp)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.array([v[0], v[1], v[2], 0.0], float)  # (x,y,z,w)
    q = np.array([v[0], v[1], v[2], w], float)
    q = q / (np.linalg.norm(q) + 1e-12)
    return q  # (x,y,z,w)


# ---------------------------
# renderer
# ---------------------------

class VuerRobotSkeleton:
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

    ## quest3 상에 그려줌
    def build_elements(self, joints_xyz): # joints_xyz는 각 관절에 대한 좌표
        joints_xyz = np.asarray(joints_xyz, dtype=float)
        joints_xyz = joints_xyz + self.offset # 각 관절 좌표 + offset
        
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
            p0 = joints_xyz[i]
            p1 = joints_xyz[j]
            v = p1 - p0
            L = float(np.linalg.norm(v))
            if L <= 1e-9:
                continue

            mid = (p0 + p1) * 0.5
            d = v / L

            # Cylinder 기본 축(Y) -> 링크 방향(d)
            q = _quat_from_two_vec(np.array([0.0, 1.0, 0.0]), d)
            elems.append(
                Cylinder(
                    # (radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength)
                    args=(self.link_radius, self.link_radius, L, 12, 1, False, 0.0, 6.28318),
                    position=mid.tolist(),
                    quaternion=q.tolist(),          
                    key=f"{self.key}:link:{i}-{j}",
                    layers=self.layers,
                    materialType="phong",
                    material=dict(color="#888888"),
                )
            )

        return elems

    def upsert(self, session, joints_xyz):
        elems = self.build_elements(joints_xyz)
        session.upsert @ group(children=elems, key=self.key)

    def clear(self, session):
        session.upsert @ group(children=[], key=self.key)