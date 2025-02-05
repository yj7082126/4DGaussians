import os, sys
sys.path.append("..")
from pathlib import Path
import json, shutil
import numpy as np
np.set_printoptions(suppress=True)
import torch

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def getw2c(qvec, T):
    c2w = np.eye(4, dtype=np.float32)
    c2w[0:3, 0:3] = qvec2rotmat(qvec)
    c2w[0:3, 3] = T
    c2w[0:3, 1] *= -1.0
    c2w[0:3, 2] *= -1.0

    w2c = np.linalg.inv(c2w)
    R, T = w2c[:3,:3], w2c[:3,3]
    qvec = rotmat2qvec(R)
    return qvec, T

root_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

(out_dir / "images").mkdir(parents=True, exist_ok=True)
(out_dir / "colmap").mkdir(parents=True, exist_ok=True)
(out_dir / "colmap/images").mkdir(parents=True, exist_ok=True)
(out_dir / "colmap/sparse_custom").mkdir(parents=True, exist_ok=True)
(out_dir / "colmap/sparse/0").mkdir(parents=True, exist_ok=True)
(out_dir / "colmap/dense/workspace").mkdir(parents=True, exist_ok=True)

object_images_txt = ""
object_cameras_txt = ""
idx = 0
cameras = sorted(list(root_dir.glob("*.json")), key=lambda x: int(x.stem.split("_v")[-1]))
for cam_path in cameras:
    name = cam_path.stem.replace(root_dir.stem, "frames")
    metadata = json.loads(cam_path.read_bytes())
    W, H = metadata['scene']['resolution']

    K = np.abs(metadata['camera']['K'])
    K = torch.tensor(K, dtype=torch.float32)
    K[0, :] *= W
    K[1, :] *= H
    focal = K[0,0]
    princp = K[0,2], K[1,2]
    for img_path in sorted(list(root_dir.glob(f"{name}/rgba_*.png"))):
        new_img_name = f"{name}-{img_path.stem}.png"
        t = int(img_path.stem.split("_")[-1])
        shutil.copy(img_path, out_dir / f"images/{new_img_name}")
        if t == 0:
            qvec = np.array(metadata['camera']['quaternions'][t])
            T = np.array(metadata['camera']['positions'][t])
            qvec, T = getw2c(qvec, T)

            shutil.copy(img_path, out_dir / f"colmap/images/{new_img_name}")
            object_images_txt += f"{idx+1} {' '.join([str(x) for x in qvec])} {' '.join([str(x) for x in T])} 1 {new_img_name}\n\n"
            object_cameras_txt += f"{idx+1} SIMPLE_PINHOLE {W} {H} {focal} {princp[0]} {princp[1]}\n"
            idx += 1

(out_dir / "colmap/sparse_custom/images.txt").write_text(object_images_txt)
(out_dir / "colmap/sparse_custom/cameras.txt").write_text(object_cameras_txt)
(out_dir / "colmap/sparse_custom/points3D.txt").write_text("")