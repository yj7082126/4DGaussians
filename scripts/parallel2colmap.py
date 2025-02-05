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
W, H = 640, 480
idx = 0
metadata = json.loads(list(root_dir.glob("calibration/*.json"))[0].read_bytes())
for (name, K, RT) in zip(metadata['names'][1:], metadata['intrinsics'][1:], metadata['extrinsics'][1:]):
    qvec = [RT['rotation']['qw'], RT['rotation']['qx'], RT['rotation']['qy'], RT['rotation']['qz']]
    T = [RT['translation']['x'], RT['translation']['y'], RT['translation']['z']]
    qvec, T = getw2c(qvec, T)

    focal = K['fx']
    princp = K['cx'], K['cy']

    for img_path in sorted(list(root_dir.glob(f"rgb/{name}/*.png"))):
        # print(img_path)
        t = int(img_path.stem)
        new_img_name = f"{name}-rgba_{((t-5)//10):05d}.png"
        shutil.copy(img_path, out_dir / f"images/{new_img_name}")
        if t == 5:
            shutil.copy(img_path, out_dir / f"colmap/images/{new_img_name}")
            object_images_txt += f"{idx+1} {' '.join([str(x) for x in qvec])} {' '.join([str(x) for x in T])} 1 {new_img_name}\n\n"
            object_cameras_txt += f"{idx+1} SIMPLE_PINHOLE {W} {H} {focal} {princp[0]} {princp[1]}\n"
            idx += 1

(out_dir / "colmap/sparse_custom/images.txt").write_text(object_images_txt)
(out_dir / "colmap/sparse_custom/cameras.txt").write_text(object_cameras_txt)
(out_dir / "colmap/sparse_custom/points3D.txt").write_text("")