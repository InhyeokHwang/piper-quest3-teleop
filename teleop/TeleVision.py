import time
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, DefaultScene
from vuer.schemas import MotionControllers
from multiprocessing import Array, Value, Process, shared_memory
import numpy as np
import asyncio
from pathlib import Path
from .piper_arm_skeleton_vuer import VuerRobotSkeleton  
from vuer.schemas import Sphere

def robot_to_vuer_pos(p_r):
    x, y, z = p_r
    return np.array([x, z, -y], dtype=float)


class OpenTeleVision:
    def __init__(self, img_shape, shm_name, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=False):

        base_dir = Path(__file__).resolve().parent
        cert_path = (base_dir / cert_file).resolve() if not Path(cert_file).is_absolute() else Path(cert_file)
        key_path  = (base_dir / key_file).resolve()  if not Path(key_file).is_absolute()  else Path(key_file)

        cert_file = str(cert_path)
        key_file  = str(key_path)

        # self.app=Vuer()
        self.img_shape = (img_shape[0], 2*img_shape[1], 3) ## 한 눈(left or right) 기준의 해상도로 들어옴
        self.img_height, self.img_width = img_shape[:2] ## 한 눈(left or right) 기준의 height/width 

        # ngrok은 로컬에서 돌아가는 서버를 인터넷 어디서나 접속할 수 있게 해주는 터널링 서비스
        if ngrok: ## 참이면 ngrok이 제공하는 https로 열기 
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3) ## queries dict(grid=False)는 Vuer 기본 UI 그리드 표시를 끄는 것. queue_len=3은 이벤트 큐 길이를 제한하는 것(지연 방지)
        else: ## 인증서 직접 사용
            self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        
        # 기존에는 손을 트래킹하는 방식이지만 나는 컨트롤러 트래킹으로 바꿀 것임
        # 컨트롤러 이벤트 핸들러
        self.app.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        # 공유메모리
        if stream_mode == "image": # OpenTeleVision -> 브라우저 
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf) 
            self.app.spawn(start=False)(self.main_image) ## vuer 세션마다 실행할 루틴으로 main_image 등록
        else:
            raise ValueError("stream_mode must be 'image'")

        
        # 컨트롤러
        self.right_controller_shared = Array('d', 16, lock=True) ## 4x4 (오른손)
        # 0 trigger_pressed, 1 squeeze_pressed, 2 touchpad_pressed, 3 thumbstick_pressed, 4 a_pressed, 5 b_pressed,
        # 6 triggerValue, 7 squeezeValue, 8 touchpadX, 9 touchpadY, 10 thumbX, 11 thumbY
        self.right_state_shared = Array('d', 12, lock=True)
        
        # -------------------------
        # Robot skeleton shared memory (joints xyz)
        # -------------------------
        self.max_joints = 8  # 넉넉히. piper면 보통 7~8이면 충분
        self.robot_n_joints = Value('i', 0, lock=True)                 # 현재 유효 조인트 개수
        self.robot_joints_shared = Array('d', 3 * self.max_joints, lock=True)  # xyz flat

        # (예시) base->...->ee 연결. 너 FK 조인트 순서에 맞게 바꿔야 함
        # N개 조인트면 edges는 [(0,1),(1,2)...]
        self.robot_edges = []

        self.skel = VuerRobotSkeleton(
            edges=self.robot_edges,
            key="robot-skel",
            joint_radius=0.015,
            link_radius=0.008,
            offset=(0.0, 0.0, 0.0),  # 또는 offset 파라미터 제거
            layers=0,                # (있다면) 레이어도 안전하게
        )
        ##########################################

        # --- EE-anchor calibration state ---
        self._world_offset = None  # np.array shape (3,) or None

        # EE를 Vuer에서 어디에 놓을지 (원점에 두려면 [0,0,0])
        self._anchor_in_vuer = np.array([0.0, 0.8, -1.5], dtype=float)

        # joints_xyz에서 EE 인덱스 (보통 마지막이면 -1)
        self._ee_index = -1

        # 머리
        self.head_matrix_shared = Array('d', 16, lock=True) ## 4x4 (머리) 
        # 카메라 aspect
        self.aspect_shared = Value('d', 1.0, lock=True) ## 1x1 (카메라 aspect)

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    
    def run(self):
        self.app.run()

    ## vr에서 손이 움직일 때 호출됨 (안씀)
    async def on_hand_move(self, event, session, fps=60):
        try:
            self.left_hand_shared[:] = event.value["leftHand"] 
            self.right_hand_shared[:] = event.value["rightHand"]
        except: 
            pass

    ## 컨트롤러를 트래킹 
    async def on_controller_move(self, event, session, fps=60):
        data = event.value
        try:
            # RIGHT
            right = data.get("right")
            if isinstance(right, (list, tuple)) and len(right) == 16:
                self.right_controller_shared[:] = right

            # RIGHT state
            rs = data.get("rightState") or {}
            if isinstance(rs, dict):
                self.right_state_shared[:] = [
                    1.0 if rs.get("trigger", False) else 0.0,
                    1.0 if rs.get("squeeze", False) else 0.0,
                    1.0 if rs.get("touchpad", False) else 0.0,
                    1.0 if rs.get("thumbstick", False) else 0.0,
                    1.0 if rs.get("aButton", False) else 0.0,
                    1.0 if rs.get("bButton", False) else 0.0,
                    float(rs.get("triggerValue", 0.0) or 0.0),
                    float(rs.get("squeezeValue", 0.0) or 0.0),
                    float((rs.get("touchpadValue", [0.0, 0.0]) or [0.0, 0.0])[0]),
                    float((rs.get("touchpadValue", [0.0, 0.0]) or [0.0, 0.0])[1]),
                    float((rs.get("thumbstickValue", [0.0, 0.0]) or [0.0, 0.0])[0]),
                    float((rs.get("thumbstickValue", [0.0, 0.0]) or [0.0, 0.0])[1]),
                ]

        except Exception as e:
            print("[CONTROLLER_MOVE] error:", e)

    def set_robot_joints(self, joints_xyz: np.ndarray):
        arr_r = np.asarray(joints_xyz, dtype=float).reshape(-1, 3)

        arr_v = np.stack([robot_to_vuer_pos(p) for p in arr_r], axis=0)

        # ✅ 최초 1회: EE 기준으로 오프셋 캘리브레이션
        if self._world_offset is None and arr_v.shape[0] >= 1:
            ee0 = arr_v[self._ee_index].copy()
            self._world_offset = self._anchor_in_vuer - ee0
            print(f"[CALIB] ee0={ee0}, world_offset={self._world_offset}")

        # ✅ 오프셋 적용
        if self._world_offset is not None:
            arr_v = arr_v + self._world_offset

        # (선택) 추가로 y를 조금 올리고 싶으면:
        # arr_v[:, 1] += 0.10

        n = int(min(arr_v.shape[0], self.max_joints))
        self.robot_edges = [(i, i + 1) for i in range(max(0, n - 1))]

        with self.robot_n_joints.get_lock():
            self.robot_n_joints.value = n

        with self.robot_joints_shared.get_lock():
            flat = self.robot_joints_shared
            for k in range(3 * self.max_joints):
                flat[k] = 0.0
            for i in range(n):
                base = 3 * i
                flat[base + 0] = float(arr_v[i, 0])
                flat[base + 1] = float(arr_v[i, 1])
                flat[base + 2] = float(arr_v[i, 2])


    async def main_image(self, session, fps=60):
        # 그리드 끄기
        session.set @ DefaultScene(grid=False, frameloop="always")
        
        # 컨트롤러
        session.upsert @ MotionControllers(stream=True, key="motion-controller", left=True, right=True,)
            
        while True:
            display_image = self.img_array

            session.upsert(
            [ImageBackground(
                # Can scale the images down.
                # display_image[::2, :self.img_width],
                # display_image[:self.img_height:2, ::2],
                display_image[::2, :self.img_width:2],
                # 'jpg' encoding is significantly faster than 'png'.
                format="jpeg",
                quality=80,
                key="left-image",
                interpolate=True,
                # fixed=True,
                aspect=1.66667,
                # distanceToCamera=0.5,
                height = 2,
                position=[0, 1, 3],
                # rotation=[0, 0, 0],
                layers=1, 
            ),
            ImageBackground(
                # Can scale the images down.
                # display_image[::2, self.img_width:],
                # display_image[self.img_height::2, ::2],
                display_image[::2, self.img_width::2],
                # 'jpg' encoding is significantly faster than 'png'.
                format="jpeg",
                quality=80,
                key="right-image",
                interpolate=True,
                # fixed=True,
                aspect=1.66667,
                # distanceToCamera=0.5,
                height = 2,
                position=[0, 1, 3],
                # rotation=[0, 0, 0],
                layers=2, 
            )],
            to="bgChildren",
            )

            # -------------------------
            # Robot skeleton draw
            # -------------------------
            # n과 버퍼를 같은 락 구역에서 읽어서 레이스 제거
            with self.robot_n_joints.get_lock(), self.robot_joints_shared.get_lock():
                n = int(self.robot_n_joints.value)
                if n >= 2:
                    buf = np.array(self.robot_joints_shared[: 3 * n], dtype=float)
                else:
                    buf = None

            if n >= 2 and buf is not None:
                joints = buf.reshape(n, 3).copy()

                # edges 동기화 (체인 형태)
                self.skel.edges = [(i, i + 1) for i in range(n - 1)]

                self.skel.upsert(session, joints)

            await asyncio.sleep(0.03)
            
        
    @property
    def right_controller(self):
        return np.array(self.right_controller_shared[:]).reshape(4, 4, order="F")
    
    @property
    def right_state(self) -> np.ndarray:
        """
        right_state shape: (12,)
        index:
          0 trigger, 1 squeeze, 2 touchpad, 3 thumbstick, 4 a, 5 b,
          6 triggerValue, 7 squeezeValue, 8 touchX, 9 touchY, 10 thumbX, 11 thumbY
        """
        return np.array(self.right_state_shared[:], dtype=float)

    @property
    def head_matrix(self):
        # with self.head_matrix_shared.get_lock():
        #     return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        # with self.aspect_shared.get_lock():
            # return float(self.aspect_shared.value)
        return float(self.aspect_shared.value)


### VR 기기 쪽에서 발생하는 이벤트들이 서버쪽 공유메모리에 올바르게 잘 기록이 되는지 검증하는 테스트    
if __name__ == "__main__":
    import time
    import numpy as np
    from multiprocessing import shared_memory, Queue, Event

    # VR에서 사용할 해상도 설정 (한 눈 기준)
    resolution = (720, 1280)
    crop_size_w = 340
    crop_size_h = 270
    resolution_cropped = (
        resolution[0] - crop_size_h,      # height
        resolution[1] - 2 * crop_size_w   # width
    )  # 예: (450, 600)

    # 공유 메모리용 전체 이미지 shape (H=2*height, W=width, 3채널)
    img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)
    img_height, img_width = resolution_cropped[:2]

    # 공유 메모리 생성
    shm = shared_memory.SharedMemory(
        create=True,
        size=np.prod(img_shape) * np.uint8().itemsize
    )
    shm_name = shm.name
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
    img_array[:] = 0  # 초기 화면은 검정색


    # OpenTeleVision 생성 (image 모드)
    tv = OpenTeleVision(
        resolution_cropped,
        shm_name,
        stream_mode="image",
        cert_file="./cert.pem",
        key_file="./key.pem",
    )

    # 테스트: 컨트롤러 기준 오른손 pose 출력
    try:
        while True:
            print("right_controller:\n", tv.right_controller)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Exit.")
    finally:
        shm.close()
        shm.unlink()