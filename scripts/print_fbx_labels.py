#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Print FBX timing labels and optional curve-key timing range."
    )
    parser.add_argument("--fbx_file", required=True, help="Path to .fbx file")
    parser.add_argument(
        "--print_curve_range",
        action="store_true",
        default=False,
        help="Also print longest joint curve key time range.",
    )
    args = parser.parse_args()

    fbx_path = Path(args.fbx_file).expanduser().resolve()
    if not fbx_path.exists():
        raise FileNotFoundError(f"FBX file not found: {fbx_path}")

    try:
        import fbx  # noqa: F401
        import FbxCommon
    except Exception as e:
        raise RuntimeError(
            "Cannot import fbx/FbxCommon. Please install Autodesk FBX Python SDK."
        ) from e

    sdk_manager, scene = FbxCommon.InitializeSdkObjects()
    FbxCommon.LoadScene(sdk_manager, scene, str(fbx_path))

    num_anim_stacks = scene.GetSrcObjectCount(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId)
    )
    if num_anim_stacks <= 0:
        print("No animation stack found.")
        return

    # Keep same selection logic as project parser.
    stack_index = 1 if num_anim_stacks > 1 else 0
    anim_stack = scene.GetSrcObject(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId), stack_index
    )
    num_anim_layers = anim_stack.GetSrcObjectCount(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimLayer.ClassId)
    )
    anim_layer = (
        anim_stack.GetSrcObject(
            FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimLayer.ClassId), 0
        )
        if num_anim_layers > 0
        else None
    )

    anim_range = anim_stack.GetLocalTimeSpan()
    duration = anim_range.GetDuration()
    time_mode = duration.GetGlobalTimeMode()
    fps_meta = duration.GetFrameRate(time_mode)
    try:
        frame_count = duration.GetFrameCount(True)
    except TypeError:
        frame_count = duration.GetFrameCount()

    start_sec = anim_range.GetStart().GetSecondDouble()
    stop_sec = anim_range.GetStop().GetSecondDouble()
    time_range_sec = stop_sec - start_sec
    fps_calc = frame_count / time_range_sec if time_range_sec > 0 else float("nan")

    print("FBX_LABELS")
    print(f"file={fbx_path}")
    print(f"num_anim_stacks={num_anim_stacks}")
    print(f"selected_anim_stack_index={stack_index}")
    print(f"anim_stack_name={anim_stack.GetName()}")
    print(f"num_anim_layers={num_anim_layers}")
    print(f"time_mode={time_mode}")
    print(f"fps_meta_from_time_mode={fps_meta}")
    print(f"frame_count={frame_count}")
    print(f"start_sec={start_sec}")
    print(f"stop_sec={stop_sec}")
    print(f"time_range_sec={time_range_sec}")
    print(f"fps_calc_framecount_div_timerange={fps_calc}")

    if args.print_curve_range and anim_layer is not None:
        max_keys = -1
        best_joint = None
        best_curve = None
        queue = [scene.GetRootNode()]
        while queue:
            joint = queue.pop(0)
            longest_curve = None
            longest_keys = -1

            for channel in ("X", "Y", "Z"):
                curve_t = joint.LclTranslation.GetCurve(anim_layer, channel)
                if curve_t and curve_t.KeyGetCount() > longest_keys:
                    longest_curve = curve_t
                    longest_keys = curve_t.KeyGetCount()

            curve_r = joint.LclRotation.GetCurve(anim_layer, "X")
            if curve_r and curve_r.KeyGetCount() > longest_keys:
                longest_curve = curve_r
                longest_keys = curve_r.KeyGetCount()

            if longest_curve and longest_keys > max_keys:
                max_keys = longest_keys
                best_joint = joint
                best_curve = longest_curve

            for i in range(joint.GetChildCount()):
                queue.append(joint.GetChild(i))

        if best_curve and best_curve.KeyGetCount() > 0:
            t0 = best_curve.KeyGetTime(0).GetSecondDouble()
            t1 = best_curve.KeyGetTime(best_curve.KeyGetCount() - 1).GetSecondDouble()
            print("CURVE_LABELS")
            print(f"best_joint_name={best_joint.GetName() if best_joint else 'None'}")
            print(f"best_curve_key_count={best_curve.KeyGetCount()}")
            print(f"best_curve_start_sec={t0}")
            print(f"best_curve_stop_sec={t1}")
            print(f"best_curve_time_range_sec={t1 - t0}")
            if best_curve.KeyGetCount() > 1:
                dt = (
                    best_curve.KeyGetTime(1).GetSecondDouble()
                    - best_curve.KeyGetTime(0).GetSecondDouble()
                )
                print(f"first_key_dt_sec={dt}")
                if dt > 0:
                    print(f"first_key_instant_fps={1.0 / dt}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
