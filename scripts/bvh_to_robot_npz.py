#!/usr/bin/env python3
"""Deprecated shim. Merged into bvh_to_robot.py.

Kept so existing commands keep working. Prefer calling bvh_to_robot.py directly
and selecting NPZ output via a .npz --save_path (or --output_format npz).
Note: --target_fps is accepted as an alias of --motion_fps.
"""
import sys
import warnings

from bvh_to_robot import main

if __name__ == "__main__":
    warnings.warn(
        "bvh_to_robot_npz.py is deprecated; use bvh_to_robot.py "
        "(save to a .npz path or pass --output_format npz).",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.exit(main())
