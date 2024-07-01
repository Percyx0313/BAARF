"""
Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch
import numpy as np
from camera_utils import cam2world
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def plot_save_poses_llff(fig,pose,pose_ref=None,path=None,ep=None):
    # get the camera meshes
    cam_depth=0.5
    _,_,cam = get_camera_mesh(pose,depth=cam_depth)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    plt.title("epoch {}".format(ep))
    ax1 = fig.add_subplot(121,projection="3d")
    ax2 = fig.add_subplot(122,projection="3d")
    setup_3D_plot(ax1,elev=-90,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))
    setup_3D_plot(ax2,elev=0,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))
    ax1.set_title("forward-facing view",pad=0)
    ax2.set_title("top-down view",pad=0)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    pred_color = (0,0.6,0.7)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):
        if pose_ref is not None:
            ax1.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax2.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax1.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
            ax2.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
        c = np.array(color(float(i)/N))*0.8
        ax1.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        
        # plot the oriantation of the camera
        # ax1.plot(cam[i,:2,0], cam[i,:2,1], cam[i,:2,2], color=(0,0,1.0), linewidth=2)
        # ax2.plot(cam[i,:2,0], cam[i,:2,1], cam[i,:2,2], color=(0,0,1.0), linewidth=2)
        
        ax1.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
        ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
    for i in range(N):
        ax1.plot([cam[i,5,0],cam_ref[i,5,0]],
                [cam[i,5,1],cam_ref[i,5,1]],
                [cam[i,5,2],cam_ref[i,5,2]], color=(1,0,0), linewidth=3)
        ax2.plot([cam[i,5,0],cam_ref[i,5,0]],
                [cam[i,5,1],cam_ref[i,5,1]],
                [cam[i,5,2],cam_ref[i,5,2]], color=(1,0,0), linewidth=3)
    png_fname = "{}/{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()

def plot_save_poses_blender(fig, pose, pose_ref=None, path=None,name=None, ep=None, cam_depth=0.5,outlier=None,other_pose=None):
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
            
            ax.plot(other_cam[i,:2,0], other_cam[i,:2,1], other_cam[i,:2,2], color=(0,0,1.0), linewidth=2)
            ax.scatter(other_cam[i,5,0], other_cam[i,5,1], other_cam[i,5,2], color=outlier_color, s=20)
    if name!=None:
        png_fname = "{}/{}_{}.png".format(path,name,ep)
    else:   
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