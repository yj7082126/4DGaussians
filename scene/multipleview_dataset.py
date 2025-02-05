import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split
    ):
        key = list(cam_intrinsics.keys())[0]
        self.focal = [cam_intrinsics[key].params[0], cam_intrinsics[key].params[0]]
        height=cam_intrinsics[key].height
        width=cam_intrinsics[key].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)
        
    
    def load_images_path(self, cam_folder, cam_extrinsics,cam_intrinsics,split):
        image_length = len(os.listdir(os.path.join(cam_folder,"cam01")))
        #len_cam=len(cam_extrinsics)
        image_paths=[]
        image_poses=[]
        image_times=[]
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            number = int(os.path.basename(extr.name)[5:-4])
            images_folder=os.path.join(cam_folder,f"cam{number:02d}")

            image_range=range(image_length)
            if split=="test":
                image_range = [image_range[0],image_range[int(image_length/3)],image_range[int(image_length*2/3)]]

            for i in image_range:    
                num=i+1
                image_path=os.path.join(images_folder,"frame_"+str(num).zfill(5)+".jpg")
                image_paths.append(image_path)
                image_poses.append((R,T))
                image_times.append(float(i/image_length))

        return image_paths, image_poses,image_times
    
    def get_video_cam_infos(self,datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]
    

class multipleview_dataset_kubric(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        image_folder,
        split,
        cam_ids = [],
        image_length = 60, 
        factor=4
    ):
        key = list(cam_intrinsics.keys())[0]
        self.focal = [cam_intrinsics[key].params[0], cam_intrinsics[key].params[0]]
        height=cam_intrinsics[key].height
        width=cam_intrinsics[key].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(
            image_folder, cam_extrinsics, split, cam_ids=cam_ids, 
            image_length=image_length, factor=factor)

    def load_images_path(self, image_folder, cam_extrinsics, split="train", cam_ids=[], image_length = 60, factor=4):
        image_range = list(range(0, image_length, factor)) if split == "train" else list(range(0, image_length, factor*5))

        image_paths, image_poses, image_times = [], [], []
        if len(cam_ids) > 0:
            cam_extrinsics = {k:v for k,v in cam_extrinsics.items() if k in cam_ids}
            cam_names = ', '.join([v.name for k,v in cam_extrinsics.items()])
            print(f"Using Camera {cam_names}") 
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            image_path_0 = os.path.join(image_folder, extr.name)
            for time in image_range:
                image_path = os.path.join(
                    os.path.dirname(image_path_0), 
                    os.path.basename(image_path_0).replace(
                        "rgba_00000.png", f"rgba_{time:05d}.png"
                    )
                )
                image_paths.append(image_path)
                image_poses.append((R,T))
                image_times.append(float(time/image_length))

        return image_paths, image_poses, image_times
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    
    def load_pose(self,index):
        return self.image_poses[index]