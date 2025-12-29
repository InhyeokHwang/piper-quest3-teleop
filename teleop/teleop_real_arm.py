# teleop/teleop_real_arm.py
from .cli import parse_args
from .app import build_runtime, run_loop

def main():
    args = parse_args()
    rt = None
    try:
        rt = build_runtime(args)
        run_loop(args, rt)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted")
    finally:
        if rt is not None:
            rt.close()

if __name__ == "__main__":
    main()
