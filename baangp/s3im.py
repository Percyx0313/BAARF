from ssim import SSIM
import torch
from einops import rearrange


class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper  
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(self, kernel_size=4, stride=4, repeat_time=10):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)
    def forward(self, src_vec, tar_vec, patch_height=64, patch_width=64):
        loss = 0.0
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list) # [M*N]
        tar_all = tar_vec[res_index] # [M*N,B,3]
        src_all = src_vec[res_index] # [M*N,B,3]
        # print("train all")
        # print(res_index.shape)
        # print(tar_all.shape)
        # print(weight.shape)
        # print(weight_all.shape)
        # exit()
        tar_patch=rearrange(tar_all,'(d H Wt) C -> d C H Wt',d=1,H=patch_height,Wt=patch_width*self.repeat_time)
        src_patch=rearrange(src_all,'(d H Wt) C -> d C H Wt',d=1,H=patch_height,Wt=patch_width*self.repeat_time)
        # print(tar_patch.shape)
        # tar_patch = tar_all.permute(1, 0).reshape(1,3, self.patch_height, self.patch_width * self.repeat_time)
        # src_patch = src_all.permute(1, 0).unsqueeze(0).unsqueeze(0).reshape(1,3, self.patch_height, self.patch_width * self.repeat_time)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss