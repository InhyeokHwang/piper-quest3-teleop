# teleop/teleop_real_arm.py
from .cli import parse_args
from .app import build_runtime, run_loop
from .piper.safety import move_to_start_pose   
from . import config


def main():
    args = parse_args()
    rt = None
    try:
        rt = build_runtime(args)
        run_loop(args, rt)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted")
    finally:
        try:
            if rt is not None and (not args.dry_run) and getattr(rt, "driver", None) is not None:
                move_to_start_pose(
                    driver=rt.driver,
                    start_position_rad=config.START_POSITION[:6],  
                    factor=config.RAD_TO_PIPER,
                    steps=200,
                    step_dt_s=0.01,
                    motion_speed=20,
                    check_reached=True,
                )
        except Exception as e:
            print("[safety] move_to_start_pose on exit failed:", repr(e))

        if rt is not None:
            rt.close()

if __name__ == "__main__":
    main()
