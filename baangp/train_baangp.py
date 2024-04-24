# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import argparse
from easydict import EasyDict as edict
from lpips import LPIPS
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
import torchvision.transforms.functional as torchvision_F
import tqdm
from datasets.ba_synthetic import SubjectLoader
from evaluation_utils import (
    evaluate_camera_alignment,
    evaluate_test_time_photometric_optim,
    prealign_cameras
)
from lie_utils import se3_to_SE3
from nerfacc.estimators.occ_grid import OccGridEstimator
from pose_utils import compose_poses
from radiance_fields.baangp import BAradianceField
from utils import (
    render_image_with_occgrid,
    set_random_seed,
    load_ckpt,
    save_ckpt
)
import visualization_utils as viz
from einops import rearrange

import wandb
from icecream import ic
from camera_utils import cam2world, rotation_distance, procrustes_analysis
from pose_utils import construct_pose
from utils import Rays
def save_camera_poses(args,train_dataset,outlier_ids : list =None,path="poses_outlier",hyp_se3=None):
    models["radiance_field"].eval()
    models["estimator"].eval()
    models['radiance_field'].testing = True
    with torch.no_grad():
        pose_refine = se3_to_SE3(models["radiance_field"].se3_refine.weight)
        gt_poses = train_dataset.camfromworld
        pred_poses = compose_poses([pose_refine, models["radiance_field"].pose_noise, gt_poses])
        pose_aligned, sim3 = prealign_cameras(pred_poses, gt_poses)
        error = evaluate_camera_alignment(pose_aligned, gt_poses)
        rot_error = np.rad2deg(error.R.mean().item())
        trans_error = error.t.mean().item()
        print("--------------------------")
        print("{} train rot error:   {:8.3f}".format(step, rot_error)) # to use numpy, the value needs to be on cpu first.
        print("{} train trans error: {:10.5f}".format(step, trans_error))
        print("--------------------------")
        # dump numbers

        pose_aligned_detached, gt_poses_detached = pose_aligned.detach().cpu(), gt_poses.detach().cpu()
        
        # # pertube pose
        if hyp_se3 is not None:
            
            # delta_translation=torch.linspace(-0.5,0.5,8).to('cuda')
            # tx,ty,tz=torch.meshgrid(delta_translation,delta_translation,delta_translation)
            # temp=torch.stack([tx,ty,tz],dim=-1).view(-1,3) # N,3
            # trans_terms=torch.cat([torch.zeros_like(temp,device=outlier_se3.device),temp],dim=-1)
            # pertubation_se3=outlier_se3.view(1,-1)+trans_terms
            pertubation_se3=hyp_se3
            
            pertubation_pose = se3_to_SE3(pertubation_se3.view(-1,6))
            pertubation_pose = compose_poses([pertubation_pose, models["radiance_field"].pose_noise[outlier_ids[0]].clone(), gt_poses[outlier_ids[0]].clone()])
            # align the pertubation pose
            center = torch.zeros(1, 1, 3,device=pred_poses.device)
            pertubation_centers = cam2world(center, pertubation_pose)[:,0] # [N,3]
            
            sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3,device=pred_poses.device))
            # align the camera poses
            pertubation_center_aligned = (pertubation_centers-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
            R_aligned = pertubation_pose[...,:3]@sim3.R.t()
            t_aligned = (-R_aligned@pertubation_center_aligned[...,None])[...,0]
            pertubation_pose_aligned = construct_pose(R=R_aligned, t=t_aligned)
        
        
        fig = plt.figure(figsize=(10, 10))
        cam_dir = os.path.join(args.save_dir, path)
        os.makedirs(cam_dir, exist_ok=True)
        png_fname = viz.plot_save_poses_blender(fig=fig,
                                                pose=pose_aligned_detached, 
                                                pose_ref=gt_poses_detached, 
                                                path=cam_dir, 
                                                ep=step,
                                                outlier=outlier_ids,other_pose=None if hyp_se3==None else pertubation_pose_aligned.detach().cpu()) 
   
    models["radiance_field"].train()
    models["estimator"].train()
    models['radiance_field'].testing = False
# def relocalize(args,models,img_idx):
#     # perform relocalization for the bad image
    
    
    

def validate(models , train_dataset, test_dataset):
    models["radiance_field"].eval()
    models["estimator"].eval()
    models['radiance_field'].testing = True

    with torch.no_grad():
        pose_refine = se3_to_SE3(models["radiance_field"].se3_refine.weight)
        gt_poses = train_dataset.camfromworld
        pred_poses = compose_poses([pose_refine, models["radiance_field"].pose_noise, gt_poses])
        pose_aligned, sim3 = prealign_cameras(pred_poses, gt_poses)
        error = evaluate_camera_alignment(pose_aligned, gt_poses)
        rot_error = np.rad2deg(error.R.mean().item())
        trans_error = error.t.mean().item()
        print("--------------------------")
        print("{} train rot error:   {:8.3f}".format(step, rot_error)) # to use numpy, the value needs to be on cpu first.
        print("{} train trans error: {:10.5f}".format(step, trans_error))
        print("--------------------------")
        # dump numbers
    
        pose_aligned_detached, gt_poses_detached = pose_aligned.detach().cpu(), gt_poses.detach().cpu()
        fig = plt.figure(figsize=(10, 10))
        cam_dir = os.path.join(args.save_dir, "poses")
        os.makedirs(cam_dir, exist_ok=True)
        png_fname = viz.plot_save_poses_blender(fig=fig,
                                                pose=pose_aligned_detached, 
                                                pose_ref=gt_poses_detached, 
                                                path=cam_dir, 
                                                ep=step)
    
    # evaluate novel view synthesis
    test_dir = os.path.join(args.save_dir, "test_pred_view")
    os.makedirs(test_dir,exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    res = []
    for i in tqdm.tqdm([27]):
        data = test_dataset[i]
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
            rays = models["radiance_field"].query_rays(idx=None,
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
            invdepth_map = invdepth.view(h, w)[None,:,:]
            # mse = F.mse_loss(rgb_map, pixels)
            # psnr = (-10.0 * torch.log(mse) / np.log(10.0)).item()
            # ssim_val = ssim(rgb_map[None, ...], pixels[None, ...]).item()
            # ms_ssim_val = ms_ssim(rgb_map[None, ...], pixels[None, ...]).item()
            # lpips_loss_val = lpips_fn(rgb, loaded_pixels).item()
            # res.append(edict(psnr=psnr, ssim=ssim_val, ms_ssim=ms_ssim_val, lpips=lpips_loss_val))
            # dump novel views
            rgb_map_cpu = rgb_map.cpu()
            # gt_map_cpu = pixels.cpu()
            depth_map_cpu = invdepth_map.cpu()
            rgb_map_cpu=torchvision_F.to_pil_image(rgb_map_cpu)#.save("{}/rgb_{}.png".format(test_dir,i))
            depth_map_cpu=torchvision_F.to_pil_image(depth_map_cpu)#.save("{}/depth_{}.png".format(test_dir,i))
    models["radiance_field"].train()
    models["estimator"].train()
    models['radiance_field'].testing = False
    return rot_error,trans_error,rgb_map_cpu,depth_map_cpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        help="the root dir of the dataset",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=SubjectLoader.SUBJECT_IDS,
        help="which scene to use",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="random seed"
    )

    parser.add_argument(
        "--c2f",
        type=float,
        nargs="+",
        action="extend"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
    )
    

    parser.add_argument("--save-dir", type=str,
        required=True,
        help="The output root directory for saving models.")
    args = parser.parse_args()

    device = "cuda:0"
    
    set_random_seed(args.seed)
    if os.path.exists(args.save_dir):
        print('%s exists!'%args.save_dir)
    else:
        print("Creating %s"%args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    # training parameters
    lr = 1.e-2
    lr_end = 1.e-4
    lr_pose = 1.e-3 #1.e-2
    lr_pose_end = 1.e-5 #1.e-3
    optim_lr_pose = 1.e-3
    max_steps = 40000 # 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 1e-6
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"factor": 2}
    test_dataset_kwargs = {"factor": 2}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

    #
    ema_alpha=0.7
    ema_ratio=(0.99/ema_alpha)**(1/(max_steps/4))
    # init wandb
    
    wandb_tag = []
    
    
    
    
    wandb.init(config=args,
               project="baangp",
               dir=args.save_dir,
               name=('/').join(args.save_dir.split('/')[-2:]),
               tags=args.scene,
               job_type="training",
               reinit=True,
               mode= "online" if args.wandb else "disabled")
    
    
    
    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=init_batch_size,
        batch_over_images=True,
        device=device,
        **train_dataset_kwargs,
    )
    
    
    # re-localization 
    re_localize_error_list=torch.zeros(len(train_dataset),device=device)
    
    
    print("Found %d train images"%len(train_dataset.images))
    print("Train image shape", train_dataset.images.shape)
    print("Setup the test dataset.")
    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="test",
        num_rays=None,
        device=device,
        batch_over_images=False,
        **test_dataset_kwargs,
    )
    print("Found %d test images."%len(test_dataset.images))
    print("Test image shape", test_dataset.images.shape)
    print(f"Setup Occupancy Grid. Grid resolution is {grid_resolution}")

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    ).to(device)


    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    
    radiance_field = BAradianceField(
        num_frame=len(train_dataset),
        aabb=estimator.aabbs[-1],
        device=device,
        c2f=args.c2f,
        ).to(device)
    
    print("Setting up optimizers...")
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=lr, eps=1e-15, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=(lr_end/lr)**(1./max_steps)
            )
    models = {"radiance_field": radiance_field, "estimator": estimator}
    schedulers={"scheduler": scheduler}
    optimizers={"optimizer": optimizer}

    pose_optimizer = torch.optim.AdamW(models['radiance_field'].se3_refine.parameters(), lr=lr_pose)
    pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        pose_optimizer,
        gamma=(lr_pose_end/lr_pose)**(1./max_steps)
    )
    schedulers["pose_scheduler"] = pose_scheduler
    optimizers["pose_optimizer"] = pose_optimizer

    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()


    models, optimizers, schedulers, epoch, iteration, has_checkpoint = load_ckpt(save_dir=args.save_dir, models=models, optimizers=optimizers, schedulers=schedulers)
    # training
    current_pyramid_level=3
    train_dataset.change_pyramid_img(level=current_pyramid_level)
    print(f"Update pyramid level to {current_pyramid_level}")
    
    if not has_checkpoint:
        tic = time.time()
        loader = tqdm.trange(max_steps + 1, desc="training", leave=False)
        for step in loader:
            models['radiance_field'].train()
            models["estimator"].train()

            
            if (step//2500)>(3-current_pyramid_level) and current_pyramid_level>0:
                current_pyramid_level-=1
                train_dataset.change_pyramid_img(level=current_pyramid_level)
                print(f"Update pyramid level to {current_pyramid_level}")
            train_dataset.progress = min(train_dataset.progress+1/2500,1)
            
            
            i = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            pixels = data["pixels"]
            grid_3D = data["grid_3D"]
            gt_poses = data["gt_w2c"] # [num_ray, 3, 4]
            image_ids = data["image_id"]
            
            
            if args.c2f is not None:
                models["radiance_field"].update_progress(step/max_steps)

            def occ_eval_fn(x):
                density = models["radiance_field"].query_density(x)
                return density * render_step_size

            # update occupancy grid
            models["estimator"].update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2,
            )

            for key in optimizers:
                # setting gradient to None to avoid extra computations.
                optimizers[key].zero_grad(set_to_none=True)

            # query rays

            rays = models["radiance_field"].query_rays(idx=image_ids, grid_3D=grid_3D, gt_poses=gt_poses, mode='train')
            # render
            rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                radiance_field=models["radiance_field"],
                estimator=models["estimator"],
                rays=rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )
            if n_rendering_samples == 0:
                loader.set_postfix(it=step, loss="skipped")
                continue

            if target_sample_batch_size > 0:
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(
                    num_rays * (target_sample_batch_size / float(n_rendering_samples))
                )
                train_dataset.update_num_rays(num_rays)

            # compute loss
            #consistency_loss = models["radiance_field"].nerf.consistency_loss
            # feat_reg= models["radiance_field"].nerf.hash_feat_loss
            
            
            
            
            
            loss = F.smooth_l1_loss(rgb, pixels)#+0.001*feat_reg#+consistency_loss
            with torch.no_grad():
                re_localize_error_list*=0.9
                re_localize_error_list.index_add_(0,image_ids,0.1*torch.mean((rgb.view(-1,3)-pixels.view(-1,3))**2.,dim=-1))
                # ic(re_localize_error_list)
                # re_localize_error_list[image_ids]=0.9*re_localize_error_list[image_ids]+0.1*torch.mean((rgb.view(-1,3)-pixels.view(-1,3))**2.,dim=-1).detach()
            old_pose=models["radiance_field"].se3_refine.weight.clone()
                
            # update the relocalization error 
            # ic(loss.item())
            
            
            

            
            
            # do not unscale it because we are using Adam.
            scaled_train_loss = grad_scaler.scale(loss)
            scaled_train_loss.backward()
            for key in optimizers:
                optimizers[key].step()
            for key in schedulers:
                schedulers[key].step()
            loader.set_postfix(it=step, loss="{:.4f}".format(scaled_train_loss.item()))

            # EMA pose
            
            with torch.no_grad():
                ema_alpha=min(ema_ratio*ema_alpha,0.99)
                # ic(ema_alpha)
                models["radiance_field"].se3_refine.weight.data=((1-ema_alpha)*models["radiance_field"].se3_refine.weight+ema_alpha*old_pose)
                
            if((step+2500)%5000==0 and step!=0):
                ic("the max error camera is : ",re_localize_error_list.argmax().item())
                ic(re_localize_error_list.max().item())
                old_re_localize_error=re_localize_error_list.max().item()
                outlier_id= re_localize_error_list.argmax().item()
                train_dataset.outlier_idx=outlier_id
            
            if(step%5000==0 and step!=0):
                train_dataset.outlier_idx=None
                viz.plot_images_blender(train_dataset, outlier_id, models, args.save_dir, step)
                # visualize the outlier pose
                save_camera_poses(args,train_dataset,outlier_ids=[outlier_id],path="poses_outlier_before")
                # parallel camera pose hypothesys
                filter_ratio=0.25
                outlier_pose_id=outlier_id
                outlier_se3=models["radiance_field"].se3_refine.weight[outlier_pose_id].clone()
                
                Stage_number=4
                stage_iter=512
                hyp_lr=1.e-3
                hyp_lr_end=1.e-4
                hyp_grad_scaler = torch.cuda.amp.GradScaler(2**10)
                
                num_hypothesis=128
                
                translation_range=0.5
                rotation_range=60
                # initial hypothesis pose
                delta_translation=translation_range*(torch.rand(num_hypothesis,3).to('cuda')-0.5)
                delta_rotation=torch.deg2rad(rotation_range*(torch.rand(num_hypothesis,3).to('cuda')-0.5))
                noise_term=torch.stack([delta_rotation,delta_translation],dim=-1).view(-1,6)
                pertubation_se3=torch.nn.Parameter(outlier_se3.view(1,-1)+noise_term)
                
                
                # multi_hypothesis training 
                hyp_pose_optimizer = torch.optim.SGD([pertubation_se3], lr=hyp_lr,momentum=0.9)
                # hyp_pose_optimizer = torch.optim.Adam([pertubation_se3], lr=hyp_lr)
                # hyp_pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                #     hyp_pose_optimizer,
                #     gamma=(hyp_lr_end/hyp_lr)**(1/stage_iter)
                    
                # )
                
                train_dataset.hypothesis_test=True
                ic("ray number is ", num_rays)
                train_dataset.hypothesis_cam_num=num_hypothesis
                soft_epoch=4096*train_dataset.hypothesis_cam_num//num_rays
                accumulated_loss=torch.zeros(train_dataset.hypothesis_cam_num).to('cuda')
                for epoch in range(Stage_number):
                    for it in tqdm.tqdm(range(stage_iter)):
                        if (it+1)%soft_epoch==0:
                            hyp_pose_optimizer.zero_grad(set_to_none=True)
                        # get ray
                        hyp_data = train_dataset[outlier_pose_id]
                        hyp_render_bkgd = hyp_data["color_bkgd"]
                        hyp_pixels = hyp_data["pixels"]
                        hyp_pixels=hyp_pixels.unsqueeze(0).repeat(train_dataset.hypothesis_cam_num,1,1)
                        hyp_grid_3D = hyp_data["grid_3D"]
                        hyp_gt_poses = hyp_data["gt_w2c"] # [num_ray, 3, 4]
                        hyp_image_ids = hyp_data["image_id"]
                        # query rays
                        pose_noises =  models["radiance_field"].pose_noise[outlier_pose_id]
                        init_poses = compose_poses([pose_noises, hyp_gt_poses])
                        poses_refine = se3_to_SE3(pertubation_se3) # [1, 3, 4]
                        # add learnable pose correction
                        poses = compose_poses([poses_refine, init_poses])
                        # given the intrinsic/extrinsic matrices, get the camera center and ray directions
                        center_3D = torch.zeros_like(hyp_grid_3D) # [B, N, 3]
                        # transform from camera to world coordinates
                        grid_3D = cam2world(hyp_grid_3D.squeeze().unsqueeze(0), poses) # [B, N, 3], [B, 3, 4] -> [B, 3]
                        center_3D = cam2world(center_3D.squeeze().unsqueeze(0), poses) # [B, N, 3]
                        directions = grid_3D - center_3D # [B, N, 3]
                        viewdirs = directions / torch.linalg.norm(
                            directions, dim=-1, keepdims=True
                        )
                        
                        center_3D = torch.reshape(center_3D, (-1, 3))
                        viewdirs = torch.reshape(viewdirs, (-1, 3))
                        rays=Rays(origins=center_3D, viewdirs=viewdirs)
                        # render
                        hyp_rgb, hyp_acc, hyp_depth, hyp_n_rendering_samples = render_image_with_occgrid(
                            radiance_field=models["radiance_field"],
                            estimator=models["estimator"],
                            rays=rays,
                            # rendering options
                            near_plane=near_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=cone_angle,
                            alpha_thre=alpha_thre,
                        )
                        
                        # compute the loss and record the loss
                        accumulated_loss=0.9*accumulated_loss+0.1*torch.mean(((hyp_rgb.view(-1,3)-hyp_pixels.view(-1,3))**2.).view(train_dataset.hypothesis_cam_num,-1),dim=-1).detach()
                        F1 = F.smooth_l1_loss(hyp_rgb.view(-1,3),hyp_pixels.view(-1,3))
                        scaled_train_loss = hyp_grad_scaler.scale(F1)
                        scaled_train_loss.backward()
                        if (it+1)%soft_epoch==0:
                            hyp_pose_optimizer.step()
                            # old_lr=hyp_pose_optimizer.param_groups[0]['lr']
                            # pertubation_se3=pertubation_se3+(old_lr*2)**0.5*torch.randn_like(pertubation_se3)
                       
                        if it%128==0:
                            save_camera_poses(args,train_dataset,outlier_ids=[outlier_id],path="temp",hyp_se3=pertubation_se3)
                    # remain the 25% best pose
                    if epoch<Stage_number-1:
                        with torch.no_grad():
                            remain_pose_num=int((filter_ratio**(epoch+1))*train_dataset.hypothesis_cam_num)
                            ic(remain_pose_num)
                            accumulated_loss,indices=torch.topk(accumulated_loss,remain_pose_num,largest=False)
                            pertubation_se3=pertubation_se3[indices].clone()
                            # resample arount the pose
                            num_resmple_hypothesis=(num_hypothesis-remain_pose_num)//remain_pose_num
                            ic(num_resmple_hypothesis)
                            translation_range/=2
                            rotation_range/=2
                            
                            
                            delta_translation=translation_range*(torch.rand(remain_pose_num,num_resmple_hypothesis,3).to('cuda')-0.5)
                            delta_rotation=torch.deg2rad(rotation_range*(torch.rand(remain_pose_num,num_resmple_hypothesis,3).to('cuda')-0.5))
                            noise_term=torch.stack([delta_rotation,delta_translation],dim=-1).view(-1,num_resmple_hypothesis,6)
                            
                            ic(noise_term.shape)
                            ic(pertubation_se3.shape)
                            new_pose=(pertubation_se3.view(-1,1,6)+noise_term).view(-1,6)
                            
                            ic(pertubation_se3.shape)

                            pertubation_se3=torch.cat([pertubation_se3,new_pose],dim=0).view(-1,6)
                            ic(pertubation_se3.shape)
                            pertubation_se3=torch.nn.Parameter(pertubation_se3).view(-1,6)
                            ic(pertubation_se3.shape)
                            
                            # reset the accumulated loss
                            accumulated_loss=torch.zeros(train_dataset.hypothesis_cam_num).to('cuda')
                            # reset the pose optimizer
                            old_lr=hyp_pose_optimizer.param_groups[0]['lr']
                            hyp_pose_optimizer = torch.optim.SGD([pertubation_se3], lr=hyp_lr,momentum=0.9)
                            # hyp_lr=hyp_lr*0.33
                            # hyp_pose_optimizer = torch.optim.Adam([pertubation_se3], lr=hyp_lr)
                            # hyp_pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                            #     hyp_pose_optimizer,
                            #     gamma=(hyp_lr_end/hyp_lr)**(1/stage_iter)
                            # )
                        
                        
                    
                # filter the pose
                ic(old_re_localize_error,accumulated_loss.min().item())
                if(old_re_localize_error>=accumulated_loss.min().item()):
                    ic("prepare to update the camera pose")
                    with torch.no_grad():
                        ic(pertubation_se3[accumulated_loss.argmin().item()])
                        ic("before pose ",models["radiance_field"].se3_refine.weight[outlier_pose_id].data)
                        models["radiance_field"].se3_refine.weight[outlier_pose_id].data.copy_(pertubation_se3[accumulated_loss.argmin().item()].detach())
                        ic("After pose ",models["radiance_field"].se3_refine.weight[outlier_pose_id].data)
                    
                    
                
                # visualize the hypothsis test pose
                save_camera_poses(args,train_dataset,outlier_ids=[outlier_id],path="poses_outlier_after")
                for key in optimizers:
                    # setting gradient to None to avoid extra computations.
                    optimizers[key].zero_grad(set_to_none=True)
                    hyp_pose_optimizer.zero_grad(set_to_none=True)    
                models["radiance_field"].zero_grad()
                models["estimator"].zero_grad()
                train_dataset.hypothesis_test=False
            
            #     train_dataset.hypothesis_test=False
            if step % 200 == 0:
                
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(loss) / np.log(10.0)
                wandb.log({
                    "render_loss": loss,
                    "PSNR": psnr,
                    #"feat_reg" : feat_reg
                },step=step)



                if step % 1000==0:
                    print(
                        f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                        f"loss={loss:.5f} | psnr={psnr:.2f} | "
                        f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                        f"max_depth={depth.max():.3f} | "
                    )
                    
                    
                    # show the pose and render image with depth and rgb
                    rot_error,trans_error,rgb_map_cpu,depth_map_cpu=validate(models , train_dataset, test_dataset)
                    wandb.log({
                        "rot_error": rot_error,
                        "trans_error": trans_error,
                        "rgb&&depth": [wandb.Image(rgb_map_cpu),wandb.Image(depth_map_cpu)]
                    },step=step)
                
                # if step==6000:
                #     exit()
        save_ckpt(save_dir=args.save_dir, iteration=step, models=models, optimizers=optimizers, schedulers=schedulers, final=True)
    else:
        step = iteration
    # evaluation
    print("Done training, start evaluation:")
    models["radiance_field"].eval()
    models["estimator"].eval()
    models['radiance_field'].testing = True

    with torch.no_grad():
        print("Plotting final pose alignment.")
        pose_refine = se3_to_SE3(models["radiance_field"].se3_refine.weight)
        gt_poses = train_dataset.camfromworld
        pred_poses = compose_poses([pose_refine, models["radiance_field"].pose_noise, gt_poses])
        pose_aligned, sim3 = prealign_cameras(pred_poses, gt_poses)
        error = evaluate_camera_alignment(pose_aligned, gt_poses)
        rot_error = np.rad2deg(error.R.mean().item())
        trans_error = error.t.mean().item()
        print("--------------------------")
        print("{} train rot error:   {:8.3f}".format(step, rot_error)) # to use numpy, the value needs to be on cpu first.
        print("{} train trans error: {:10.5f}".format(step, trans_error))
        print("--------------------------")
        # dump numbers
        quant_fname = os.path.join(args.save_dir, "quant_pose.txt")
        with open(quant_fname,"w") as file:
            for i, (err_R, err_t) in enumerate(zip(error.R, error.t)):
                file.write("{} {} {}\n".format(i, err_R.item(), err_t.item()))
    
        pose_aligned_detached, gt_poses_detached = pose_aligned.detach().cpu(), gt_poses.detach().cpu()
        fig = plt.figure(figsize=(10, 10))
        cam_dir = os.path.join(args.save_dir, "poses")
        os.makedirs(cam_dir, exist_ok=True)
        png_fname = viz.plot_save_poses_blender(fig=fig,
                                                pose=pose_aligned_detached, 
                                                pose_ref=gt_poses_detached, 
                                                path=cam_dir, 
                                                ep=step)
    
    # evaluate novel view synthesis
    test_dir = os.path.join(args.save_dir, "test_pred_view")
    os.makedirs(test_dir,exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    res = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
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
            rays = models["radiance_field"].query_rays(idx=None,
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
            mse = F.mse_loss(rgb_map, pixels)
            psnr = (-10.0 * torch.log(mse) / np.log(10.0)).item()
            ssim_val = ssim(rgb_map[None, ...], pixels[None, ...]).item()
            ms_ssim_val = ms_ssim(rgb_map[None, ...], pixels[None, ...]).item()
            lpips_loss_val = lpips_fn(rgb, loaded_pixels).item()
            res.append(edict(psnr=psnr, ssim=ssim_val, ms_ssim=ms_ssim_val, lpips=lpips_loss_val))
            # dump novel views
            rgb_map_cpu = rgb_map.cpu()
            gt_map_cpu = pixels.cpu()
            depth_map_cpu = invdepth_map.cpu()
            torchvision_F.to_pil_image(rgb_map_cpu).save("{}/rgb_{}.png".format(test_dir,i))
            torchvision_F.to_pil_image(gt_map_cpu).save("{}/rgb_GT_{}.png".format(test_dir,i))
            torchvision_F.to_pil_image(depth_map_cpu).save("{}/depth_{}.png".format(test_dir,i))
            
    plt.close()
    # show results in terminal
    avg_psnr = np.mean([r.psnr for r in res])
    avg_ssim = np.mean([r.ssim for r in res])
    avg_ms_ssim = np.mean([r.ms_ssim for r in res])
    avg_lpips = np.mean([r.lpips for r in res])
    print("--------------------------")
    print("PSNR:  {:8.2f}".format(avg_psnr))
    print("SSIM:  {:8.3f}".format(avg_ssim))
    print("MS-SSIM:  {:8.3f}".format(avg_ms_ssim))
    print("LPIPS: {:8.3f}".format(avg_lpips))
    print("--------------------------")
    # dump numbers to file
    quant_fname = os.path.join(args.save_dir, "quant.txt")
    with open(quant_fname,"w") as file:
        for i,r in enumerate(res):
            file.write("{} {} {} {} {}\n".format(i, r.psnr, r.ssim, r.ms_ssim, r.lpips))
            
    # assume the test view synthesis are already generated
    print("writing videos...")
    rgb_vid_fname = os.path.join(args.save_dir, "test_view_rgb.mp4")
    depth_vid_fname = os.path.join(args.save_dir, "test_view_depth.mp4")
    os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_dir, rgb_vid_fname))
    os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_dir, depth_vid_fname))

    print("Training and evaluation stops.")
        