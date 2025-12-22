#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
from teleop.VuerTeleop import VuerTeleop  # 경로는 네 프로젝트 구조에 맞게

def main():
    teleop = VuerTeleop("inspire_hand.yml")  # 지금 쓰는 config와 동일하게

    try:
        while True:
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleop.step()

            # 1) 한 번만 형식/shape 로그
            print("head_rmat shape:", np.shape(head_rmat))
            print("left_pose shape:", np.shape(left_pose))
            print("right_pose shape:", np.shape(right_pose))
            print("left_qpos shape:", np.shape(left_qpos))
            print("right_qpos shape:", np.shape(right_qpos))

            # 2) 실제 값 몇 개만 찍어보기
            print("head_rmat[0]:", head_rmat[0])
            print("right_pose (xyz):", right_pose[:3])
            print("right_pose (quat):", right_pose[3:])

            # 3) 값이 변하는지 보기 위해 간단히 sleep
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[TEST] Interrupted. Bye!")

if __name__ == "__main__":
    main()
