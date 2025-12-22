import time
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, DefaultScene
from vuer.schemas import MotionControllers
from multiprocessing import Array, Value, Process, shared_memory
import numpy as np
import asyncio
from pathlib import Path


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
        if ngrok: ## 참이면 ngrok이 제공하는 https로 열기 -> 인증서 없이 (vuer는 로봇공학에서 많이 쓰는 오픈 소스 시각화 도구임)
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3) ## queries dict(grid=False)는 Vuer 기본 UI 그리드 표시를 끄는 것. queue_len=3은 이벤트 큐 길이를 제한하는 것(지연 방지)
        else: ## 인증서 직접 사용
            self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        
        # 기존에는 손을 트래킹하는 방식이지만 나는 컨트롤러 트래킹으로 바꿀 것임
        #self.app.add_handler("HAND_MOVE")(self.on_hand_move) ## 만약 Vuer 클라이언트에서 "HAND_MOVE"가 오면 on_hand_move 루틴 호출
        # 컨트롤러 이벤트 핸들러
        self.app.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        # 공유메모리
        if stream_mode == "image": # OpenTeleVision -> 브라우저 
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf) 
            self.app.spawn(start=False)(self.main_image) ## vuer 세션마다 실행할 루틴으로 main_image 등록
        else:
            raise ValueError("stream_mode must be 'image'")

        # 손
        self.left_hand_shared = Array('d', 16, lock=True) ## 4x4 (왼손)
        self.right_hand_shared = Array('d', 16, lock=True) ## 4x4 (오른손)

        # 컨트롤러
        self.left_controller_shared = Array('d', 16, lock=True) ## 4x4 (왼손)
        self.right_controller_shared = Array('d', 16, lock=True) ## 4x4 (오른손)
        
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

            # LEFT
            left = data.get("left")
            if isinstance(left, (list, tuple)) and len(left) == 16:
                self.left_controller_shared[:] = left

        except Exception as e:
            print("[CONTROLLER_MOVE] error:", e)


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
            await asyncio.sleep(0.03)

        
    @property
    def left_controller(self):
        return np.array(self.left_controller_shared[:]).reshape(4, 4, order="F")

    @property
    def right_controller(self):
        return np.array(self.right_controller_shared[:]).reshape(4, 4, order="F")

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