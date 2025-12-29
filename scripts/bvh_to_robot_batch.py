import argparse
import pathlib
import os
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def retarget_single_file(args_tuple):
    """Worker function for multiprocessing — must be top-level."""
    bvh_file, robot, format_name, motion_fps, override = args_tuple

    try:
        bvh_path = pathlib.Path(bvh_file)
        bvh_basename = bvh_path.stem
        save_path = pathlib.Path("retarget") / robot / format_name / f"{bvh_basename}.pkl"

        # Skip if exists and not override
        if save_path.exists() and not override:
            return {
                "status": "skipped",
                "bvh": bvh_path.name,
                "save_path": str(save_path),
                "error": None
            }

        # Ensure output dir
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load BVH
        lafan1_data_frames, actual_human_height = load_bvh_file(str(bvh_path), format=format_name)

        # Retarget
        retargeter = GMR(
            src_human=f"bvh_{format_name}",
            tgt_robot=robot,
            actual_human_height=actual_human_height,
        )

        qpos_list = []
        for smplx_data in lafan1_data_frames:
            qpos = retargeter.retarget(smplx_data)
            qpos_list.append(qpos)

        # Extract & convert to standard format
        root_pos = np.array([q[:3] for q in qpos_list])
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz → xyzw
        dof_pos = np.array([q[7:] for q in qpos_list])

        dof_names = [
            name for name, idx in sorted(retargeter.robot_dof_names.items(), key=lambda x: x[1])
            if name != "floating_joint"
        ]

        labels = (
            ["root_pos/x", "root_pos/y", "root_pos/z"] +
            ["root_quat/x", "root_quat/y", "root_quat/z", "root_quat/w"] +
            [f"dof_pos/{name}" for name in dof_names]
        )

        frames = [
            np.concatenate([root_pos[i], root_rot[i], dof_pos[i]]).tolist()
            for i in range(len(qpos_list))
        ]

        motion_data = {
            "LoopMode": "Once",
            "LoopNum": 1,
            "FrameDuration": 1.0 / motion_fps,
            "EnableCycleOffsetPosition": True,
            "EnableCycleOffsetRotation": True,
            "MotionWeight": 1.0,
            "Labels": labels,
            "Frames": frames,
        }

        with open(save_path, "wb") as f:
            pickle.dump(motion_data, f)

        return {
            "status": "success",
            "bvh": bvh_path.name,
            "save_path": str(save_path),
            "error": None
        }

    except Exception as e:
        return {
            "status": "failed",
            "bvh": pathlib.Path(bvh_file).name,
            "save_path": None,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Source folder containing BVH files to retarget.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "sfu"],
        default="lafan1",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy",
            "fourier_n1", "engineai_pm01", "pal_talos", "adam_sp_pro", "adam_inspire"
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel processes. Default: os.cpu_count()",
    )
    args = parser.parse_args()

    src_folder = pathlib.Path(args.src_folder).resolve()
    if not src_folder.is_dir():
        raise ValueError(f"❌ src_folder '{src_folder}' is not a directory.")

    bvh_files = list(src_folder.rglob("*.bvh"))
    if not bvh_files:
        print(f"⚠️  No .bvh files found in {src_folder}")
        return

    print(f"🔍 Found {len(bvh_files)} BVH files. Retargeting to robot: [bold]{args.robot}[/bold] | format: [bold]{args.format}[/bold]")

    # Prepare args for workers
    tasks = [
        (str(f), args.robot, args.format, args.motion_fps, args.override)
        for f in bvh_files
    ]

    num_workers = args.num_workers or os.cpu_count()
    print(f"👷 Using {num_workers} worker processes...")

    success, skipped, failed = 0, 0, 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(retarget_single_file, task): task[0] for task in tasks}

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(tasks), desc="Retargeting"):
            result = future.result()
            status = result["status"]

            if status == "success":
                success += 1
                print(f"✅ {result['bvh']} → {result['save_path']}")
            elif status == "skipped":
                skipped += 1
                # Uncomment below if you want to see skipped files:
                # print(f"⏭️  Skipped: {result['bvh']}")
            else:  # failed
                failed += 1
                print(f"❌ {result['bvh']}: {result['error']}")

    # Summary
    print("\n" + "="*50)
    print(f"🎉 Batch Retargeting Finished!")
    print(f"  ✅ Success: {success}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Failed:  {failed}")
    print(f"📁 Outputs saved under: ./retarget/{args.robot}/{args.format}/")


if __name__ == "__main__":
    # Required for Windows/macOS multiprocessing
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()