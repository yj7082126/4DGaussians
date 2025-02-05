import argparse
from pathlib import Path
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str)
    args = parser.parse_args()

    path1 = Path("/data/3D_dataset/datasets/kubric4d/test/") / args.workdir
    path2 = Path("/data/3D_dataset/MVS/4DGaussians/data/multipleview/") / args.workdir

    cameras = sorted(list(path1.glob("frames*/")), key=lambda x: int(x.stem.split("_v")[-1]))
    for c in cameras:
        cam_ind = int(c.stem.split("_v")[-1]) + 1
        (path2 / f"cam{cam_ind:02d}").mkdir(parents=True, exist_ok=True)

        frames = sorted(list(c.glob("rgba_*.png")))
        for f in frames:
            frame_ind = int(f.stem.split("_")[-1]) + 1
            shutil.move(
                f, 
                path2 / f"cam{cam_ind:02d}/frame_{frame_ind:05d}.jpg"
            )