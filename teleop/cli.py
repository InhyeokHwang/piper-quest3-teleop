# teleop/cli.py
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--can", type=str, default="can0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config", type=str, default="inspire_hand.yml")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--print-freq", action="store_true")
    p.add_argument("--debug-mapper", action="store_true")
    
    return p.parse_args()
