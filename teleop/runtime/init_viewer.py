# teleop/runtime/init_viewer.py
import mujoco
import mujoco.viewer

def init_viewer(model, data, dry_run: bool):
    if not dry_run:
        return None
    viewer = mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    )
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    print("[DRY RUN] MuJoCo viewer launched.")
    return viewer
