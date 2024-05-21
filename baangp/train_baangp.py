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
from evaluation_utils import (
    evaluate_camera_alignment,
    evaluate_test_time_photometric_optim,
    prealign_cameras,
    pose_evaluate
)
import importlib
from lie_utils import se3_to_SE3,so3_t3_to_SE3
from nerfacc.estimators.occ_grid import OccGridEstimator
from pose_utils import compose_poses,compose_split,invert_pose
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
from icecream import ic, install
install()
from camera_utils import cam2world, rotation_distance, procrustes_analysis
from pose_utils import construct_pose
from utils import Rays
from s3im import S3IM
def save_camera_poses(args,train_dataset,outlier_ids : list =None,path="poses_outlier",name=None,pertubation_SE3=None):
    models["radiance_field"].eval()
    models["estimator"].eval()
    models['radiance_field'].testing = True
    with torch.no_grad():
        gt_poses = train_dataset.camfromworld
        rot_error,trans_error,pose_aligned,sim3=\
        pose_evaluate(models["radiance_field"].se3_refine_R.weight, 
                      models["radiance_field"].se3_refine_T.weight,
                      models["radiance_field"].pose_noise,
                      gt_poses)
        
        
        print("--------------------------")
        print("{} train rot error:   {:8.3f}".format(step, rot_error)) # to use numpy, the value needs to be on cpu first.
        print("{} train trans error: {:10.5f}".format(step, trans_error))
        print("--------------------------")
        # dump numbers

        pose_aligned_detached, gt_poses_detached = pose_aligned.detach().cpu(), gt_poses.detach().cpu()
        
        # # pertube pose
        if pertubation_SE3 is not None:
            # get the pertubation pose
            pertubation_SE3 = pertubation_SE3.clone().detach()
            # refine_noise_pose=compose_split(pertubation_pose, models["radiance_field"].pose_noise[outlier_ids])
            pertubation_pose = compose_poses([pertubation_SE3, models["radiance_field"].pose_noise[outlier_ids[0]], gt_poses[outlier_ids[0]]])
            
            # align the pertubation pose
            center = torch.zeros(1, 1, 3,device=pertubation_pose.device)
            pertubation_centers = cam2world(center, pertubation_pose)[:,0] # [N,3]
            
            sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3,device=pertubation_pose.device))
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
                                                name=name,
                                                ep=step,
                                                outlier=outlier_ids,other_pose=None if pertubation_SE3==None else pertubation_pose_aligned.detach().cpu()) 
   
    models["radiance_field"].train()
    models["estimator"].train()
    models['radiance_field'].testing = False
    

def validate(models , train_dataset, test_dataset,step=0):
    models["radiance_field"].eval()
    models["estimator"].eval()
    models['radiance_field'].testing = True

    with torch.no_grad():
        gt_poses = train_dataset.camfromworld
        rot_error,trans_error,pose_aligned,sim3=\
        pose_evaluate(models["radiance_field"].se3_refine_R.weight, 
                      models["radiance_field"].se3_refine_T.weight,
                      models["radiance_field"].pose_noise,
                      gt_poses,
                      dataset='llff')
        
        ic(sim3.t0, sim3.t1, sim3.s0, sim3.s1, sim3.R)
        print("--------------------------")
        print("{} train rot error:   {:8.3f}".format(step, rot_error)) # to use numpy, the value needs to be on cpu first.
        print("{} train trans error: {:10.5f}".format(step, trans_error))
        print("--------------------------")
        # dump numbers
    
        pose_aligned_detached, gt_poses_detached = pose_aligned.detach().cpu(), gt_poses.detach().cpu()
        fig = plt.figure(figsize=(10, 10))
        cam_dir = os.path.join(args.save_dir, "poses")
        os.makedirs(cam_dir, exist_ok=True)
        viz.plot_save_poses_llff(fig,pose_aligned_detached,pose_ref=gt_poses_detached,path=cam_dir,ep=step)
        # png_fname = viz.plot_save_poses_blender(fig=fig,
        #                                         pose=pose_aligned_detached, 
        #                                         pose_ref=gt_poses_detached, 
        #                                         path=cam_dir, 
        #                                         ep=step)
    
        # evaluate novel view synthesis
        test_dir = os.path.join(args.save_dir, "test_pred_view")
        os.makedirs(test_dir,exist_ok=True)
    
    # if step!=0:
    #     for i in tqdm.tqdm([0]):
    #         data = test_dataset[i]
    #         gt_poses, pose_refine_test = evaluate_test_time_photometric_optim(
    #             radiance_field=models["radiance_field"], estimator=models["estimator"],
    #             render_step_size=render_step_size,
    #             cone_angle=cone_angle,
    #             data=data, 
    #             sim3=sim3, lr_pose=optim_lr_pose, test_iter=100,
    #             alpha_thre=alpha_thre,
    #             device=device,
    #             near_plane=near_plane,
    #             far_plane=far_plane,
    #             )
        
        # with torch.no_grad():
        #     rays = models["radiance_field"].query_rays(idx=None,
        #                                             sim3=sim3, 
        #                                             gt_poses=gt_poses, 
        #                                             pose_refine_test=pose_refine_test, 
        #                                             mode='eval',
        #                                             test_photo=True,
        #                                             grid_3D=data['grid_3D'])
        #     # rendering
        #     rgb, opacity, depth, _,_ = render_image_with_occgrid(
        #         # scene
        #         radiance_field=models["radiance_field"],
        #         estimator=models["estimator"],
        #         rays=rays,
        #         # rendering options
        #         near_plane=near_plane,
        #         far_plane=far_plane,
        #         render_step_size=render_step_size,
        #         render_bkgd = data["color_bkgd"],
        #         cone_angle=cone_angle,
        #         alpha_thre=alpha_thre,
        #     )
        #     # evaluate view synthesis
        #     invdepth = 1/ depth
        #     loaded_pixels = data["pixels"]
        #     h, w, c = loaded_pixels.shape
        #     pixels = loaded_pixels.permute(2, 0, 1)
        #     rgb_map = rgb.view(h, w, 3).permute(2, 0, 1)
        #     invdepth_map = invdepth.view(h, w)[None,:,:]
        #     # dump novel views
        #     rgb_map_cpu = rgb_map.cpu()
        #     depth_map_cpu = invdepth_map.cpu()
        #     rgb_map_cpu=torchvision_F.to_pil_image(rgb_map_cpu)#.save("{}/rgb_{}.png".format(test_dir,i))
        #     depth_map_cpu=torchvision_F.to_pil_image(depth_map_cpu)#.save("{}/depth_{}.png".format(test_dir,i))
        #     rgb_map_cpu.save("{}/rgb_{}.png".format(test_dir,step))
        #     depth_map_cpu.save("{}/depth_{}.png".format(test_dir,step))
            
    models["radiance_field"].train()
    models["estimator"].train()
    models['radiance_field'].testing = False
    # return rot_error,trans_error,rgb_map_cpu,depth_map_cpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="the root dir of the dataset",
    )
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
        nargs="+",
        action="extend"
    )
    # relocalization
    parser.add_argument(
        "--filter_ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--hypythosis_num",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--stage_number",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--stage_iter",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--range_T",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--range_R",
        type=float,
        default=60,
    )
    
    
    parser.add_argument(
        "--wandb",
        action="store_true",
    )
    
    parser.add_argument(
        "--reloc",
        action="store_true",
    )
    

    parser.add_argument("--save-dir", type=str,
        required=True,
        help="The output root directory for saving models.")







    args = parser.parse_args()
    args.c2f=[0.1,0.5]
    device = "cuda:0"
    
    set_random_seed(args.seed)
    if os.path.exists(args.save_dir):
        print('%s exists!'%args.save_dir)
    else:
        print("Creating %s"%args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    # training parameters
    lr = 1.e-2  if args.dataset=='blender' else  1.e-2
    lr_end = 1.e-4 if args.dataset=='blender' else 1.e-4
    lr_pose_R = 3.e-3 if args.dataset=='blender' else  3.e-3#1.e-2
    lr_pose_T = 5.e-3 if args.dataset=='blender' else  3.e-3#1.e-2
    optim_lr_pose = 1.e-3
    max_steps = 40000 if args.dataset=='blender' else 40000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 1e-6
    
          
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    alpha_thre = 0.0
    cone_angle = 0.0

    #
    ema_alpha=0.9
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
    
    
    
    if args.dataset=="blender":
        dataset="ba_synthetic"
        # dataset parameters
        train_dataset_kwargs = {"factor": 2}
        test_dataset_kwargs = {"factor": 2}
    elif args.dataset=="llff":
        dataset="ba_real"
        train_dataset_kwargs = {"factor": 1}
        test_dataset_kwargs = {"factor": 1}
    else:
        assert False ,"Give the dataset!"
    
    
    
    
    
    dataset_module=importlib.import_module("datasets.{}".format(dataset))
    
    train_dataset = dataset_module.SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=init_batch_size,
        batch_over_images=True,
        device=device,
        **train_dataset_kwargs,
    )
    # scene parameters
    if args.dataset=='llff':
        aabb_scale=2**np.ceil(np.log2(train_dataset.max_bound))
        aabb = torch.tensor([-aabb_scale, -aabb_scale, -aabb_scale, aabb_scale, aabb_scale, aabb_scale], device=device)
        near_plane = 0
        ic(aabb_scale)
        far_plane = aabb_scale*np.sqrt(3)
    elif args.dataset=='blender':
        aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
        near_plane = 0.0
        far_plane = 1.0e10
    else:
        assert False,"Need to give the known dataset"
    
    render_step_size = 5e-3 if args.dataset=='blender' else 1e-2
    target_render_step_size = 5e-3 if args.dataset=='blender' else 5e-3
    
    # re-localization 
    re_localize_error_list=torch.zeros(len(train_dataset),device=device)
    
    # s3im loss
    s3im=S3IM()
    print("Found %d train images"%len(train_dataset.images))
    print("Train image shape", train_dataset.images.shape)
    print("Setup the test dataset.")
    test_dataset = dataset_module.SubjectLoader(
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
        dataset=args.dataset,
        geo_feat_dim=64
        ).to(device)
    
    print("Setting up optimizers...")
    # setting optimizer w/o se3_refine_R.weight and se3_refine_T.weight
    parameters = []
    for name, param in radiance_field.named_parameters():
        if 'se3_refine_R' not in name and 'se3_refine_T' not in name:
            parameters.append(param)
    
    optimizer = torch.optim.Adam(
        parameters, lr=lr, eps=1e-15, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=(lr_end/lr)**(1./max_steps)
            )
    models = {"radiance_field": radiance_field, "estimator": estimator}
    schedulers={"scheduler": scheduler}
    optimizers={"optimizer": optimizer}

    pose_optimizer = torch.optim.Adam(
        [{'params':models['radiance_field'].se3_refine_R.parameters(), 'lr':lr_pose_R},
         {'params':models['radiance_field'].se3_refine_T.parameters(), 'lr':lr_pose_T}],betas=[0.9,0.99])
    pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        pose_optimizer,
        gamma=(1.e-2)**(1./max_steps)
    )
    schedulers["pose_scheduler"] = pose_scheduler
    optimizers["pose_optimizer"] = pose_optimizer

    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()


    models, optimizers, schedulers, epoch, iteration, has_checkpoint = load_ckpt(save_dir=args.save_dir, models=models, optimizers=optimizers, schedulers=schedulers)
    # training
    current_pyramid_level=1
    train_dataset.change_pyramid_img(level=current_pyramid_level)
    print(f"Update pyramid level to {current_pyramid_level}")
    
    if not has_checkpoint:
        tic = time.time()
        loader = tqdm.trange(max_steps + 1, desc="training", leave=False)
        validate(models , train_dataset, test_dataset,step=0)
        for step in loader:
            models['radiance_field'].train()
            models["estimator"].train()
            
            # linear increase the step size
            # render_step_size = render_step_size + (target_render_step_size - render_step_size) * step / max_steps
            
            
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
            rgb, acc, depth, n_rendering_samples,alphas_t = render_image_with_occgrid(
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

            
            loss = F.smooth_l1_loss(rgb, pixels)#-0.001*torch.mean(alphas_t*torch.log(alphas_t+1e-8)+(1-alphas_t)*torch.log((1-alphas_t)+1e-8))#+0.001*feat_reg#+consistency_loss
            
            # s3im loss
            
            # size= np.floor((rgb.shape[0])**0.5).astype(int)
            # s3im_loss = s3im(rgb[:size**2], pixels[:size**2],size,size)
            # loss+=0.1*s3im_loss
            
            # with torch.no_grad():
            #     re_localize_error_list*=0.9
            #     re_localize_error_list.index_add_(0,image_ids,0.1*torch.mean((rgb.view(-1,3)-pixels.view(-1,3))**2.,dim=-1))
            old_pose_R=models["radiance_field"].se3_refine_R.weight.clone()
            old_pose_T=models["radiance_field"].se3_refine_T.weight.clone()
            
            
            

            
            
            # do not unscale it because we are using Adam.
            scaled_train_loss = grad_scaler.scale(loss)
            scaled_train_loss.backward()
            for key in optimizers:
                optimizers[key].step()
            for key in schedulers:
                schedulers[key].step()
            loader.set_postfix(it=step, loss="{:.4f}".format(scaled_train_loss.item()))

            # EMA pose
            
            # with torch.no_grad():
            #     ema_alpha=min(ema_ratio*ema_alpha,0.99)
            #     # ic(ema_alpha)
            #     models["radiance_field"].se3_refine_R.weight.data=((1-ema_alpha)*models["radiance_field"].se3_refine_R.weight+ema_alpha*old_pose_R)
            #     models["radiance_field"].se3_refine_T.weight.data=((1-ema_alpha)*models["radiance_field"].se3_refine_T.weight+ema_alpha*old_pose_T)
            
            if(args.reloc==True and step%1000==0 and step >5000):
                ic(models["radiance_field"].se3_refine_R.weight[outlier_id])
            
            if(args.reloc==True and (step+2500)%5000==0 and step!=0 and step <max_steps-5000):
                # fine and stop traing the outlier pose
                ic("the max error camera is : ",re_localize_error_list.argmax().item() , re_localize_error_list.max().item())
                old_re_localize_error=re_localize_error_list.max().item()
                outlier_id= re_localize_error_list.argmax().item()
                train_dataset.outlier_idx=outlier_id
            
            if(args.reloc==True and (step+1)%5000==0 and step!=0 and  step <max_steps-5000 ):
                train_dataset.outlier_idx=None
                train_dataset.hypothesis_test=True
                # show the outlier before relocalization
                save_camera_poses(args,train_dataset,outlier_ids=[outlier_id],path="poses_outlier_before")
                # init the relocalizer's parameters
                filter_ratio=args.filter_ratio
                outlier_pose_id=outlier_id
                outlier_se3_R,outlier_se3_T=models["radiance_field"].se3_refine_R.weight[outlier_pose_id].clone(),models["radiance_field"].se3_refine_T.weight[outlier_pose_id].clone()
                Stage_number=args.stage_number
                stage_iter=args.stage_iter
                reloc_lr_pose_R = 3.e-3 
                reloc_lr_pose_T = 5.e-3 
                reloc_grad_scaler = torch.cuda.amp.GradScaler(2**10)
                num_hypothesis=args.hypythosis_num
                train_dataset.hypothesis_cam_num=num_hypothesis
                soft_epoch=1 # 4096*train_dataset.hypothesis_cam_num//num_rays
                accumulated_loss=torch.zeros(num_hypothesis).to('cuda')
                # initial hypothesis pose
                range_T=args.range_T
                range_R=args.range_R
                delta_T=range_T*(torch.rand(num_hypothesis,3).to('cuda')-0.5)
                delta_R=torch.deg2rad(range_R*(torch.rand(num_hypothesis,3).to('cuda')-0.5))
                from camera_utils import generate_uniform_hypothesis
                hyp_R,hyp_T=generate_uniform_hypothesis(range_T,range_R,outlier_se3_T,outlier_se3_R,
                                                        models["radiance_field"].pose_noise[outlier_pose_id],train_dataset[outlier_pose_id]["gt_w2c"] ,num_hypothesis)
                
                # pertubation_se3_R=torch.nn.Parameter((rearrange(outlier_se3_R,'n -> 1 n')+delta_R).clone()) # [hyp_num,3]
                # pertubation_se3_T=torch.nn.Parameter((rearrange(outlier_se3_T,'n -> 1 n')+delta_T).clone()) # [hyp_num,3]
                hyp_R.requires_grad=True
                hyp_T.requires_grad=True
                
                pertubation_se3_R=torch.nn.Parameter(hyp_R.clone()) # [hyp_num,3]
                pertubation_se3_T=torch.nn.Parameter(hyp_T.clone()) # [hyp_num,3]
                
                
                
                
                reloc_pose_optimizer = torch.optim.Adam(
                [{'params':pertubation_se3_R, 'lr':lr_pose_R},
                {'params':pertubation_se3_T, 'lr':lr_pose_T}],betas=[0.9,0.99])
                
                # start re localization
                models["radiance_field"].eval()
                reloc_pose_optimizer.zero_grad(set_to_none=True)
                for epoch in range(Stage_number):
                    for it in tqdm.tqdm(range(stage_iter)):
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
                        poses_refine = so3_t3_to_SE3(pertubation_se3_R,pertubation_se3_T) # [1, 3, 4]
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
                        scaled_train_loss = reloc_grad_scaler.scale(F1)
                        scaled_train_loss.backward()
                    
                        
                        if (it+1)%soft_epoch==0:
                            reloc_pose_optimizer.step()
                            reloc_pose_optimizer.zero_grad(set_to_none=True)
                            
                        
                        if it%128==0:
                            name = ('init' if it==0 else f'E{epoch}_I{it}')
                            ic(outlier_pose_id)
                            save_camera_poses(args,train_dataset,outlier_ids=[outlier_pose_id],path="reloc_process",name=name,pertubation_SE3=poses_refine)
                    # filter the hypothsis pose
                    if epoch<Stage_number-1:
                        with torch.no_grad():
                            remain_pose_num=max(int((filter_ratio**(epoch+1))*train_dataset.hypothesis_cam_num),1)
                            accumulated_loss,indices=torch.topk(accumulated_loss,remain_pose_num,largest=False)
                            new_pertubation_se3_R=pertubation_se3_R[indices].detach().clone()
                            new_pertubation_se3_T=pertubation_se3_T[indices].detach().clone()
                            # resample arount the pose
                            num_resmple_hypothesis=(num_hypothesis-remain_pose_num)//remain_pose_num
                            # ic(num_resmple_hypothesis)
                            # ic()
                            range_T/=2
                            range_R/=2
                            
                            ic(models["radiance_field"].pose_noise[outlier_pose_id].shape)
                            hyp_R,hyp_T=generate_uniform_hypothesis(range_T,range_R,new_pertubation_se3_T.view(-1,1,3),new_pertubation_se3_R.view(-1,1,3),
                                                        models["radiance_field"].pose_noise[outlier_pose_id],hyp_gt_poses,num_resmple_hypothesis)

                            new_pertubation_se3_R=torch.cat([new_pertubation_se3_R,hyp_R],dim=0).view(-1,6)
                            new_pertubation_se3_T=torch.cat([new_pertubation_se3_T,hyp_T],dim=0).view(-1,6)
                            # ic(new_pertubation_se3.shape)
                        new_pertubation_se3_R=torch.nn.Parameter(new_pertubation_se3_R).view(-1,3)
                        pertubation_se3_R=new_pertubation_se3_R.detach().clone().requires_grad_(True)
                        new_pertubation_se3_T=torch.nn.Parameter(new_pertubation_se3_T).view(-1,3)
                        pertubation_se3_T=new_pertubation_se3_T.detach().clone().requires_grad_(True)
                        # reset the pose optimizer
                        accumulated_loss=torch.zeros(train_dataset.hypothesis_cam_num).to('cuda')
                        # old_lr=reloc_pose_optimizer.param_groups[0]['lr']
                        # old_lr*=0.1**(1/Stage_number)
                        reloc_pose_optimizer = torch.optim.Adam(
                        [{'params':pertubation_se3_R, 'lr':lr_pose_R},
                        {'params':pertubation_se3_T, 'lr':lr_pose_T}],betas=[0.9,0.99])
                
                        # ic(pertubation_se3.requires_grad)
                        hyp_grad_scaler = torch.cuda.amp.GradScaler(2**10)
                
                # update the outlier pose
                ic(old_re_localize_error,accumulated_loss.min().item())
                if(old_re_localize_error>=accumulated_loss.min().item()):
                    ic("prepare to update the camera pose")
                    with torch.no_grad():
                        models["radiance_field"].se3_refine_R.weight[outlier_pose_id].data.copy_(pertubation_se3_R[accumulated_loss.argmin().item()].detach())
                        models["radiance_field"].se3_refine_T.weight[outlier_pose_id].data.copy_(pertubation_se3_T[accumulated_loss.argmin().item()].detach())
                        
                        
                # visualize the hypothsis test pose
                save_camera_poses(args,train_dataset,outlier_ids=[outlier_id],path="poses_outlier_after")
                # recover the training data setting  
                models["radiance_field"].train()
                train_dataset.hypothesis_test=False
                for key in optimizers:
                    # setting gradient to None to avoid extra computations.
                    optimizers[key].zero_grad(set_to_none=True)
                    reloc_pose_optimizer.zero_grad(set_to_none=True)  
                models["radiance_field"].zero_grad()
                models["estimator"].zero_grad()            
              
            
            if step % 100 == 0:
                
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(loss) / np.log(10.0)
                wandb.log({
                    "render_loss": loss,
                    "PSNR": psnr,
                    #"feat_reg" : feat_reg
                },step=step)



                if step % 100==0:
                    print(
                        f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                        f"loss={loss:.5f} | psnr={psnr:.2f} | "
                        f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                        f"max_depth={depth.max():.3f} | "
                    )
                    
                    
                    # show the pose and render image with depth and rgb
                    # rot_error,trans_error,rgb_map_cpu,depth_map_cpu=validate(models , train_dataset, test_dataset)
                    if step!=0:
                        validate(models , train_dataset, test_dataset,step=step)
                    # wandb.log({
                    #     "rot_error": rot_error,
                    #     "trans_error": trans_error,
                    #     "rgb&&depth": [wandb.Image(rgb_map_cpu),wandb.Image(depth_map_cpu)]
                    # },step=step)
            # if step%10==0 and step!=0:
            #     validate(models , train_dataset, test_dataset,step=step)
                # if step==7000:
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
        # pose_refine = se3_to_SE3(models["radiance_field"].se3_refine.weight)
        
        pose_refine = so3_t3_to_SE3(models["radiance_field"].se3_refine_R.weight, models["radiance_field"].se3_refine_T.weight)
        gt_poses = train_dataset.camfromworld
        if args.dataset=='blender':
            pred_poses = compose_poses([pose_refine, models["radiance_field"].pose_noise, gt_poses])
        else:
            
            pred_poses= so3_t3_to_SE3(models["radiance_field"].se3_refine_R.weight, models["radiance_field"].se3_refine_T.weight)
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
        ic(data["pixels"].shape)
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
            rgb, opacity, depth, _,_ = render_image_with_occgrid(
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
            rgb=rgb.detach()
            opacity=opacity.detach()
            depth=depth.detach()
            # evaluate view synthesis
            invdepth = 1/ depth
            loaded_pixels = data["pixels"]
            h, w, c = loaded_pixels.shape
            pixels = loaded_pixels.permute(2, 0, 1)
            rgb_map = rgb.view(h, w, 3).permute(2, 0, 1).detach()
            invdepth_map = invdepth.view(h, w).detach()
            mse = F.mse_loss(rgb_map, pixels)
            psnr = (-10.0 * torch.log(mse) / np.log(10.0)).item()
            ssim_val = ssim(rgb_map[None, ...], pixels[None, ...]).item()
            ms_ssim_val = ms_ssim(rgb_map[None, ...], pixels[None, ...]).item()
            lpips_loss_val = lpips_fn(rgb, loaded_pixels).item()
            res.append(edict(psnr=psnr, ssim=ssim_val, ms_ssim=ms_ssim_val, lpips=lpips_loss_val))
            # dump novel views
            ic("asdsadsadsad")
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
        