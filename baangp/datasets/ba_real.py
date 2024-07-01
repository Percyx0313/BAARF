"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import imageio.v2 as imageio
import json
import numpy as np
import os
from PIL import Image
import sys
import torch
import torch.nn.functional as torch_F

sys.path.append("..")

from camera_utils import img2cam
from lie_utils import se3_to_SE3,SE3_to_se3
from pose_utils import to_hom, construct_pose, compose_poses, invert_pose
import kornia
from icecream import ic
ic.configureOutput(includeContext=True)
from LightGlue.lightglue import LightGlue, SuperPoint
from einops import rearrange
def parse_raw_camera(pose_raw):
    """Convert pose from camera_to_world to world_to_camera and follow the right, down, forward coordinate convention."""
    pose_flip = construct_pose(R=torch.diag(torch.tensor([1,-1,-1]))) # right, up, backward --> right down, forward
    pose = compose_poses([pose_flip, pose_raw[:3]])
    pose = invert_pose(pose) # world_from_camera --> camera_from_world
    pose = compose_poses([pose_flip,pose])
    return pose
def center_camera_poses(poses):
    # compute average pose
    center = poses[...,3].mean(dim=0)
    v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
    v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
    v0 = v1.cross(v2)
    pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
    # apply inverse of averaged pose
    poses =compose_poses([poses,invert_pose(pose_avg)])
    return poses
def parse_cameras_and_bounds(path):
    fname = "{}/poses_bounds.npy".format(path)
    data = torch.tensor(np.load(fname),dtype=torch.float32)
    # parse cameras (intrinsics and poses)
    cam_data = data[:,:-2].view([-1,3,5]) # [N,3,5]
    poses_raw = cam_data[...,:4] # [N,3,4]
    poses_raw[...,0],poses_raw[...,1] = poses_raw[...,1],-poses_raw[...,0]
    raw_H,raw_W,focal = cam_data[0,:,-1]
    assert(raw_H==raw_H and raw_W==raw_W)
    # parse depth bounds
    bounds = data[:,-2:] # [N,2]
    scale = 1./(bounds.min()*0.75) # not sure how this was determined
    poses_raw[...,3] *= scale
    bounds *= scale
    # roughly center camera poses
    poses_raw = center_camera_poses(poses_raw)
    return focal,poses_raw,bounds

def _load_renderings(root_fp: str, subject_id: str, split: str, factor: float,scale: float):
    """"return images, camfromworld, focal"""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )
    
    data_dir = os.path.join(root_fp, subject_id)
    path_image = "{}/images_{}".format(data_dir,scale)
    path_depth_image = "{}/depth_{}".format(data_dir,scale)
    image_fnames = sorted(os.listdir(path_image))
    # depth_image_fnames = sorted(os.listdir(path_depth_image))
    
    images = []
    camfromworld = []
    depths=[]
    for img in image_fnames:
        name = "{}/{}".format(path_image,img)
        # print(name)
        rgba=imageio.imread(name)
        if rgba.shape[-1] == 3:
            alpha = np.ones_like(rgba[..., :1]) * 255
            rgba = np.concatenate([rgba, alpha], axis=-1)
        if factor != 1:
            h, w = rgba.shape[:2]
            image = Image.fromarray(rgba)
            resized_image = image.resize((int(w/factor), int(h/factor)))
            rgba = np.array(resized_image)
        images.append(rgba)
        
        depths.append(rgba)
    # for img in depth_image_fnames:
    #     name = "{}/{}".format(path_depth_image,img)
    #     # print(name)
    #     rgba=imageio.imread(name)
        
    #     if factor != 1:
    #         h, w = rgba.shape[:2]
    #         image = Image.fromarray(rgba)
    #         resized_image = image.resize((int(w/factor), int(h/factor)))
    #         rgba = np.array(resized_image)
    #     gray_frames = np.dot(rgba, [0.2989, 0.5870, 0.1140])
        
    #     depths.append(gray_frames)
        
    focal,poses_raw,bounds = parse_cameras_and_bounds(data_dir)
    
    # parse the camera pose
    for i in range(len(poses_raw)):
        camfromworld.append(parse_raw_camera(poses_raw[i]))
    # split the train val
    num_val_split = int(len(poses_raw)*0.1)
    images = images[:-num_val_split] if split=="train" else images[-num_val_split:]
    poses = camfromworld[:-num_val_split] if split=="train" else camfromworld[-num_val_split:]
    bounds = bounds[:-num_val_split] if split=="train" else bounds[-num_val_split:]

    images = torch.from_numpy(np.stack(images, axis=0)).to(torch.uint8)
    depths = torch.from_numpy(np.stack(depths, axis=0)).to(torch.uint8)
    poses = torch.from_numpy(np.stack(poses, axis=0)).to(torch.float32)

    
    max_bound=(bounds[:,1]-bounds[:,0]).max()
    
    
    return images,depths,poses,focal,max_bound

   


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "fern",
        "flower",
        "fern",
        "horns",
        "leaves",
        "orchids",
        "room",
        "trex",
        "fortress",
        "garden"
    ]


    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        factor: float = 4,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
        dof: int = 6,
        noise: float = 0.15
    ):
        self.RAW_WIDTH, self.RAW_HEIGHT = 4032, 3024
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.load_image_scale=8
        self.split = split
        self.factor = factor
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.noise = noise
        if split == "trainval":
            _images_train,_depths_train, _camfromworld_train, _focal_train ,self.max_bound= _load_renderings(
                root_fp, subject_id, "train", factor=factor,scale=self.load_image_scale
            )
            _images_val,_depths_val, _camfromworld_val, _ ,self.max_bound= _load_renderings(
                root_fp, subject_id, "val", factor=factor,scale=self.load_image_scale
            )
            images = torch.cat([_images_train, _images_val])
            depths= torch.cat([_depths_train, _depths_val])
            camfromworld = torch.cat(
                [_camfromworld_train, _camfromworld_val]
            )
            self.focal = _focal_train            
        else:
            images, depths,camfromworld, self.focal, self.max_bound = _load_renderings(
                root_fp, subject_id, split, factor=factor,scale=self.load_image_scale
            )
            
        self.pyramid_level=4
        self.images_parimid=kornia.geometry.transform.build_pyramid(images.permute(0,3,1,2)/255., self.pyramid_level, border_type='replicate',align_corners=True)
        
        # assert images.shape[1:3] == (self.RAW_HEIGHT//factor, self.RAW_WIDTH//factor)
        self.height,self.width = images.shape[1:3]
        K = torch.tensor(
            [
                [self.focal/self.load_image_scale/self.factor, 0, self.width / 2.0],
                [0, self.focal/self.load_image_scale/self.factor, self.height / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        self.origin_images = images.to(device).to(torch.float32)
        self.images = images.to(device).to(torch.float32)
        self.depths=depths.to(device)
        
        self.camfromworld = camfromworld.to(device)
        self.K = K.to(device)
        self.num_rays = num_rays
        self.training = (self.num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.hypothesis_test=False
        self.hypothesis_cam_num=64
        
        self.progress=None
        self.outlier_idx=None
        
        
        # for matching point 
        with torch.no_grad():
            self.macthing_k=3
            self.k_matching_image_list=[[0 for _ in range(self.macthing_k)] for _ in range(len(self.images))]
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
            self.matcher = LightGlue(features="superpoint").eval().to(device)
            feats=[self.extractor.extract(rearrange(self.images[i,:,:,:3],'h w c -> c h w')/255.0) for i in range(len(self.images))]
        # align the feature number
        
        minimum_feature_num=min([f['keypoints'].shape[1] for f in feats])
        self.feats={}
        
        
        for key in feats[0].keys():
            # ic(key)
            # ic(feats[0][key].shape)
            for f in feats:
                f[key]=f[key][:,:minimum_feature_num]
            self.feats[key]=torch.cat([f[key] for f in feats],dim=0)
            
            # ic(self.feats[key].shape)
        # self.feats=torch.cat([f['keypoints'][:,:minimum_feature_num] for f in feats],dim=0)
        # ic("matching feat tensor : ", self.feats.shape)
    def get_gaussian_kernel(self,kernel_size=127, sigma=10):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*np.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
        return gaussian_kernel / gaussian_kernel.sum()
    def set_smooth_image(self):
        
        # linear interpolation from 10 to 0
        alpha=10*(1-np.clip(((self.progress)/0.15),0,1))
        print(alpha)
        if(alpha<=1e-4):
            return
        kernel=self.get_gaussian_kernel(31,alpha).to(self.origin_images.device)
        self.images=torch_F.conv2d(self.origin_images.permute(0,3,1,2)/255.0, kernel.repeat(4,1,1,1),
                                        bias=None, stride=1, padding="same", dilation=1, groups=4).permute(0,2,3,1)*255.0
        
        
    def smooth_img_scheduler(self):
        # linear decrease the sigma from 10 to 1
        if self.progress is not None:
            sigma=10*(1-self.progress)
            self.images=self.get_smooth_image(self.images,127,sigma)
        self.images=self.get_smooth_image(self.images,127,10)
        
    def change_pyramid_img(self,level):
        '''
        level : [0,1,2,3]
        '''
        self.progress=0
        if level>0:
            upscale_image=kornia.geometry.transform.resize(torch.FloatTensor(self.images_parimid[level]),size=(self.images_parimid[0].shape[2],\
                self.images_parimid[0].shape[3]),align_corners=True)
        else:
            upscale_image=self.images_parimid[level]
        upper_level=level+1 if level<self.pyramid_level-1 else level
        upper_upscale_image=kornia.geometry.transform.resize(torch.FloatTensor(self.images_parimid[upper_level]),size=(self.images_parimid[0].shape[2],\
                self.images_parimid[0].shape[3]),align_corners=True)
        
        device=self.images.device
        self.images=self.images.to('cpu')
        self.images=upscale_image.to(device).permute(0,2,3,1)*255.
        self.upper_level_images=upper_upscale_image.to(device).permute(0,2,3,1)*255.
        imageio.imwrite(f'resize_test_{0}.png', self.images[0,:,:,:3].squeeze().cpu().numpy().astype(np.uint8))
        imageio.imwrite(f'resize_test_{1}.png', self.images[1,:,:,:3].squeeze().cpu().numpy().astype(np.uint8))
    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba = data["rgba"]
        def rgba_to_pixels(rgba):
            pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

            if self.training:
                if self.color_bkgd_aug == "random":
                    color_bkgd = torch.rand(3, device=self.images.device)
                elif self.color_bkgd_aug == "white":
                    color_bkgd = torch.ones(3, device=self.images.device)
                elif self.color_bkgd_aug == "black":
                    color_bkgd = torch.zeros(3, device=self.images.device)
            else:
                # just use white during inference
                color_bkgd = torch.ones(3, device=self.images.device)

            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
            return pixels, color_bkgd
        pixels, color_bkgd = rgba_to_pixels(rgba)
        if self.training:
            return {
                "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
                "color_bkgd": color_bkgd,  # [3,]
                **{k: v for k, v in data.items() if k not in ["rgba"]},
            }

        # pre-calculate camera centers in camera coordinate to 
        # save time during training.
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # Compute image coordinate grid.
        if self.hypothesis_test==True:
            assert self.num_rays is not None, "self.training is True, must pass in a num_rays."
            image_id = [index]
            x = torch.randint(
                0, self.width, size=(self.num_rays//self.hypothesis_cam_num,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(self.num_rays//self.hypothesis_cam_num,), device=self.images.device
            )
        elif self.training:
            # Randomly select num_ray images and sample one ray per image.
            # allow duplicates, so one image may have more rays.
            if self.batch_over_images:
                assert self.num_rays is not None, "self.training is True, must pass in a num_rays."
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(self.num_rays,),
                    device=self.images.device,
                )
                if self.outlier_idx!=None:
                    image_id[image_id==self.outlier_idx]=(self.outlier_idx+1)%len(self.images)
            else:
                image_id = [index]
            x = torch.randint(
                0, self.width, size=(self.num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(self.num_rays,), device=self.images.device
            )
            
            
        
            
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            
                
        # adds 0.5 here.
        xy_grid = torch.stack([x,y],dim=-1).view(-1, 2) + 0.5 # [HW,2] or [B, N_rays, 2]
        # self.K is of shape [3,3]
        grid_3D = img2cam(to_hom(xy_grid), self.K) # [B, 2], [3, 3] -> [B, 3]
        images = self.images
        
        
        rgba = images[image_id, y, x] / 255.0
        
        # if self.training and self.progress is not None:
        #     # ic(self.progress)
        #     rgba= self.progress*rgba+(1- self.progress)*self.upper_level_images[image_id, y, x] / 255.0

        w2c = torch.reshape(self.camfromworld[image_id], (-1, 3, 4)) # [3, 4] or (num_rays, 3, 4)
        if self.training:
            rgba = torch.reshape(rgba, (-1, 4))
            grid_3D = torch.reshape(grid_3D, (-1, 1, 3)) # extra dimension is needed for query_rays.
            return {
                "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
                "grid_3D": grid_3D,  # [h, w, 3] or [num_rays, 3]
                "gt_w2c": w2c, # [num_images, 3, 4]
                "image_id": image_id, # [num_images]]
            }
        else:
            rgba = torch.reshape(rgba, (self.height, self.width, 4))
            grid_3D = torch.reshape(grid_3D, (self.height, self.width, 3))
            return {
                "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
                "grid_3D": grid_3D,  # [h, w, 3] or [num_rays, 3]
                "gt_w2c": w2c, # [num_images, 3, 4]
                "image_id": image_id # [num_images]]
            }
    @torch.no_grad()
    def get_matching_point(self, noise_pose):
        '''
        intput :
        noise_pose : [N,3,4]
        
        output :
        matching_point : [B,N,2]   , N is determined by the least of matching points
        '''
        
        
        
        
        self.matcher.eval()
        # find the k nearest camera for each image
        noise_lie=SE3_to_se3(noise_pose)
        dist_matrix=torch.norm(noise_lie[:,None,...]-noise_lie[None,...],dim=-1)
        diag_idx=torch.arange(dist_matrix.shape[0])
        dist_matrix[diag_idx,diag_idx]=1e10
        values,indices=torch.topk(dist_matrix,self.macthing_k,dim=-1,largest=False)
    
        
        # extract the feature for each image
        with torch.no_grad():
            matches01 = self.matcher({"image0": self.feats, "image1": self.feats})
        # feats0, feats1, matches01 = [
        #     rbd(x) for x in [feats0, feats1, matches01]
        # ]  # remove batch dimension

        # kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        # axes = viz2d.plot_images([image0, image1])
        # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

        # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        # viz2d.plot_images([image0, image1])
        # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        # keypoints = self.superpoint(image)
        exit()
        return keypoints

