"""
Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License
import os
import torch
import numpy as np
from camera_utils import cam2world
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from icecream import ic


# New imports #############################################
from evaluation_utils import (
    evaluate_test_time_photometric_optim,
    prealign_cameras
)
from lie_utils import se3_to_SE3
from utils import render_image_with_occgrid
from pose_utils import compose_poses
import torchvision.transforms.functional as torchvision_F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def plot_images_blender(dataset, idx, models, path, step=0, render_step_size=5e-3, alpha_thre=0.0, cone_angle=0.0, optim_lr_pose=1.e-3, near_plane=0.0, far_plane=1.0e10):
    dataset.training = False
    models["radiance_field"].eval()
    models["estimator"].eval()
    models['radiance_field'].testing = True

    with torch.no_grad():
        print("Plotting final pose alignment.")
        pose_refine = se3_to_SE3(models["radiance_field"].se3_refine.weight)
        gt_poses = dataset.camfromworld
        pred_poses = compose_poses([pose_refine, models["radiance_field"].pose_noise, gt_poses])
        _, sim3 = prealign_cameras(pred_poses, gt_poses)
        # dump numbers
    
    # evaluate novel view synthesis
    data = dataset[idx]
    gt_poses, pose_refine_test = evaluate_test_time_photometric_optim(
        radiance_field=models["radiance_field"], estimator=models["estimator"],
        render_step_size=render_step_size,
        cone_angle=cone_angle,
        data=data, 
        sim3=sim3, lr_pose=optim_lr_pose, test_iter=100,
        alpha_thre=alpha_thre,
        device=device,
        near_plane=near_plane,
        far_plane=far_plane,
        )
    with torch.no_grad():
        rays = models["radiance_field"].query_rays( idx=None,
                                                    sim3=sim3, 
                                                    gt_poses=gt_poses, 
                                                    pose_refine_test=pose_refine_test, 
                                                    mode='eval',
                                                    test_photo=True,
                                                    grid_3D=data['grid_3D'])
        # rendering
        rgb, opacity, depth, _ = render_image_with_occgrid(
            # scene
            radiance_field=models["radiance_field"],
            estimator=models["estimator"],
            rays=rays,
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd = data["color_bkgd"],
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        # evaluate view synthesis
        invdepth = 1/ depth
        loaded_pixels = data["pixels"]
        h, w, c = loaded_pixels.shape
        pixels = loaded_pixels.permute(2, 0, 1)
        rgb_map = rgb.view(h, w, 3).permute(2, 0, 1)
        invdepth_map = invdepth.view(h, w)
        # dump novel views
        rgb_map_cpu = rgb_map.cpu()
        gt_map_cpu = pixels.cpu()
        depth_map_cpu = invdepth_map.cpu()
        path = os.path.join(path, "test_pred_view")
        torchvision_F.to_pil_image(rgb_map_cpu).save(f"{path}/rgb_{step}_{idx}.png")
        torchvision_F.to_pil_image(gt_map_cpu).save(f"{path}/rgb_GT_{step}_{idx}.png")
        torchvision_F.to_pil_image(depth_map_cpu).save(f"{path}/depth_{step}_{idx}.png")
    dataset.training = True
    models["radiance_field"].train()
    models["estimator"].train()
    models['radiance_field'].testing = False

def plot_save_poses_blender(fig, pose, pose_ref=None, path=None, ep=None, cam_depth=0.5,outlier=None,other_pose=None):
    # get the camera meshes
    _, _, cam = get_camera_mesh(pose, depth=cam_depth)
    cam = cam.numpy()
    
    if other_pose is not None:
        _, _, other_cam = get_camera_mesh(other_pose, depth=cam_depth)
        other_cam = other_cam.numpy()
    
    if pose_ref is not None:
        _, _, cam_ref = get_camera_mesh(pose_ref, depth=cam_depth)
        cam_ref = cam_ref.numpy()
    else:
      cam_ref = cam
    # set up plot window(s)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("epoch {}".format(ep),pad=0)
    setup_3D_plot(ax, elev=45, azim=35, lim=edict(x=(-3,3), y=(-3,3), z=(-3,2.4)))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0, hspace=0)
    plt.margins(tight=True, x=0, y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)
    outlier_color = (0,0.5,0)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i,:,0], cam_ref[i,:,1], cam_ref[i,:,2], color=ref_color, linewidth=0.5)
        ax.scatter(cam_ref[i,5,0], cam_ref[i,5,1], cam_ref[i,5,2], color=ref_color,s=20)
    if ep==0:
        png_fname = "{}/GT.png".format(path)
        plt.savefig(png_fname,dpi=75)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2, facecolor=pred_color))
    
    
    if other_pose is not None:
        ax.add_collection3d(Poly3DCollection([v[:4] for v in other_cam],alpha=0.2, facecolor=pred_color))
    for i in range(N):
        if outlier is not None and i in outlier:
            ax.plot(cam[i,:,0], cam[i,:,1], cam[i,:,2], color=outlier_color, linewidth=2)
            ax.scatter(cam[i,5,0], cam[i,5,1], cam[i,5,2], color=outlier_color, s=20)
        else:
            ax.plot(cam[i,:,0], cam[i,:,1], cam[i,:,2], color=pred_color, linewidth=1)
            ax.scatter(cam[i,5,0], cam[i,5,1], cam[i,5,2], color=pred_color, s=20)
    
            
    for i in range(N):
        ax.plot([cam[i,5,0],cam_ref[i,5,0]],
                [cam[i,5,1],cam_ref[i,5,1]],
                [cam[i,5,2],cam_ref[i,5,2]], color=(1,0,0), linewidth=3)
        
        
    if other_pose is not None:
        for i in range(len(other_cam)):
            ax.plot(other_cam[i,:,0], other_cam[i,:,1], other_cam[i,:,2], color=outlier_color, linewidth=2)
            ax.scatter(other_cam[i,5,0], other_cam[i,5,1], other_cam[i,5,2], color=outlier_color, s=20)
      
    png_fname = "{}/{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()
    return png_fname

def get_camera_mesh(pose,depth=1):
    vertices = torch.tensor([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])
    # vertices forms a camera facing z direction at origin.
    vertices = cam2world(vertices[None], pose)
    # converts vertices to world coordinate system.
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices, faces, wireframe


def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)