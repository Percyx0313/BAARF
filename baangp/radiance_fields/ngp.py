"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from typing import Callable, List, Union


try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()
from icecream import ic

class _HashGradientScaler(torch.autograd.Function):  # typing: ignore
    """
    Scales the gradients of hash features based on a provided mask
    """

    @staticmethod
    def forward(ctx, value, mask):
        ctx.save_for_backward(mask)
        return value, mask

    @staticmethod
    def backward(ctx, output_grad, grad_scaling):
        (mask,) = ctx.saved_tensors
        N = mask.shape[1]
        D = mask.shape[2]
        B = output_grad.shape[0]
        in_shape = output_grad.shape
        output_grad = (output_grad.view(B, N, D) * mask).view(*in_shape)
        return output_grad, grad_scaling

class ScaleHash(torch.nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        num_layers: int = 2,
        hidden_dim: int = 64,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        n_features_per_level: int = 2,                                  # features_per_level
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        base_resolution: int = 16,                                      # min_res
        max_resolution: int = 2048,                                     # max_res
        geo_feat_dim: int = 15,
        n_levels: int = 16,                                             # num_levels
        log2_hashmap_size: int = 19,                                    # log2_hashmap_size
        c2f=None,                                                       # coarse_to_fine_iters
        hash_init_scale: float = 0.001,                                 # hash_init_scale
        implementation = "tcnn",                                        # implementation
        interpolation = None,                                           # interpolation 
    ) -> None:
        # super().__init__(in_dim=num_dim)
        super().__init__()
        self.num_levels = n_levels
        self.features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size
        self.coarse_to_fine_iters = c2f
        self.step = 0
        
        levels = torch.arange(n_levels)
        growth_factor = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)) if n_levels > 1 else 1
        self.scalings = torch.floor(base_resolution * growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size

        self.tcnn_encoding = None
        self.hash_table = torch.empty(0)
        
        # if implementation == "tcnn" and not TCNN_EXISTS:
        #     print_tcnn_speed_warning("HashEncoding")
        #     implementation = "torch"
        if implementation == "tcnn":
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": growth_factor,
            }
            if interpolation is not None:
                encoding_config["interpolation"] = interpolation

            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )
        elif implementation == "torch":
            self.hash_table = torch.rand(size=(self.hash_table_size * n_levels, n_features_per_level)) * 2 - 1
            self.hash_table *= hash_init_scale
            self.hash_table = torch.nn.Parameter(self.hash_table)
            
        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"
            
    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def set_step(self, step: int) -> None:
        self.step = step
    def hash_fn(self, in_tensor):
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x
    
    def pytorch_fwd(self, in_tensor):
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]
    
    def scale_grad_by_freq(self, outputs):
        """Scale gradients by frequency of hash table entries"""
        if self.coarse_to_fine_iters is None:
            return outputs
        B = outputs.shape[0]
        N = self.num_levels
        D = self.features_per_level
        # formula for getting frequency mask
        start, end = self.coarse_to_fine_iters
        assert (
            start >= 0 and end >= 0
        ), f"start and end iterations for bundle adjustment have to be positive, got start = {start} and end = {end}"
        L = N
        # From https://arxiv.org/pdf/2104.06405.pdf equation 14
        alpha = (self.step - start) / (end - start) * L
        k = torch.arange(L, dtype=outputs.dtype, device=outputs.device)
        mask_vals = (1.0 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        mask_vals = mask_vals[None, ..., None].repeat((B, 1, D))
        mask_vals[:,:,:2]=1.0
        out, _ = _HashGradientScaler.apply(outputs, mask_vals)  # type: ignore
        return out
    
    def forward(self, in_tensor):
        if self.tcnn_encoding is not None:
            out = self.tcnn_encoding(in_tensor)
        else:
            out = self.pytorch_fwd(in_tensor)
        return self.scale_grad_by_freq(out)
class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply

def random_weight_matrix(scale,alpha):
    weight_matrix=torch.FloatTensor(
            [[[scale**(i-j) if (j<=i and j<=(alpha+1)*2) else 0 ] for j in range(32) ] for i in range(32)]
        ).view(32,32).to('cuda').half()
        # print(self.weight_matrix)
    weight_matrix=torch.nn.functional.normalize(weight_matrix,dim=1)
    print(weight_matrix)
    return weight_matrix

# resnet block
class ResnetBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ResnetBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
        )

    def forward(self, x):
        return x + self.block(x)


class NGPRadianceField(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        num_layers: int = 3,
        hidden_dim: int = 64,
        num_layers_color: int = 2,
        hidden_dim_color: int = 64,
        n_features_per_level: int = 2, 
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        base_resolution: int = 16,
        max_resolution: int = 2048,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        c2f=None,
    ) -> None:
        """
        Args:
            aabb: region of interest.
            num_dim: input dimension.
            num_layers: number of hidden layers.
            hidden_dim: dimension of hidden layers.
            num_layers_color: number of hidden layers for color network.
            hidden_dim_color: dimension of hidden layers for color network.
            n_features_per_level: number of features per level.
            use_viewdirs: whether or not to use viewdirections
            density_activation: density activation function.
            base_resolution: may be set based on the aabb for 3D case.
            max_resolution: maximum resolution of the hashmap for the base mlp.
            geo_feat_dim: mlp input dimension.
            n_levels: number of levels used for hash grid.
            log2_hashmap_size: hash map size will be defined as 2^log2_hashmap_size. When N^num_dim <= 2^19, the hash grid maps 1:1,
                whereas when N^num_dim > 2^19, there is a chance of collission. N max is 2048, and N min is 16 in the original instant-NGP.
        Returns:
            NGPradianceField instance for bounded scene.
        """
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        self.n_features_per_level = n_features_per_level

        self.progress = torch.nn.Parameter(torch.tensor(0.))
        self.consistency_loss=0
        self.c2f=c2f
        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )
            

        mlp_encoding_config = {
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            }
        network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            }
        
        self.encoding = tcnn.Encoding(n_input_dims=num_dim, encoding_config=mlp_encoding_config)
        # self.encoding2 = tcnn.Encoding(n_input_dims=num_dim, encoding_config=mlp_encoding_config)
        
        
        
        self.mlp_base = tcnn.Network(n_input_dims=self.encoding.n_output_dims,
                                     n_output_dims=1 +self.geo_feat_dim,
                                     network_config=network_config) 
        
        self.mlp_base_torch=torch.nn.Sequential(
            ResnetBlock(self.encoding.n_output_dims, self.geo_feat_dim),
            torch.nn.ReLU(),
            ResnetBlock(self.geo_feat_dim, self.geo_feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.geo_feat_dim, 1 + self.geo_feat_dim)
        )
        
        self.depth_mlp=torch.nn.Sequential(
            torch.nn.Linear(4*6+3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1+ self.geo_feat_dim)
        )
        
        
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    (
                        self.direction_encoding.n_output_dims
                        if self.use_viewdirs
                        else 0
                    )
                    + self.geo_feat_dim
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_color,
                    "n_hidden_layers": num_layers_color - 1,
                },
            )
        self.encoding=ScaleHash(aabb,c2f=[0.0,0.5])
    def GaussianConv1d(self, x, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        # x: [B, C, L]
        # kernel: [C, K]
        # output: [B, C, L]
        B, C, L = x.size()
        K = kernel_size
        # pad = (K - 1) * dilation - padding
        pad = padding
        x = torch.nn.functional.pad(x, (pad, pad))
        x = x.unfold(2, K, stride)
        x = x.view(B, C, -1, K)
        kernel = torch.randn(C, K).to(x)
        x = x * kernel.unsqueeze(0).unsqueeze(-1)
        x = x.sum(-1)
        return x

    
    def gaussian_pdf(self,x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    def contract(self,x):
        mag = torch.linalg.norm(x, dim=-1)[..., None]
        return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
    def query_density(self, x, weights:torch.Tensor = None, return_feat: bool = False):
        
        # start,end = 0.1,0.5
        # alpha = (self.progress.data-start)/(end-start)*4
        # inverse_x=x
        # PE_x=self.positional_encoding(inverse_x.view(-1, self.num_dim),4,alpha)
        # PE_x=torch.cat([inverse_x,PE_x],dim=-1)
        
        # coarse_f=self.depth_mlp(PE_x)
        aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
        
        # x=(x-aabb_min*8)/(aabb_max*8-aabb_min*8)
        
        x=(self.contract(x)+2)/4 # o [-2,2]
        # norm_x=torch.norm(x,dim=-1)
        # x=(x+2)/(4) # normalize
        # x = (x - aabb_min) / (aabb_max - aabb_min) # normalize
        encoded_x = self.encoding(x.view(-1, self.num_dim)) # [N,32]
        self.encoded_x=encoded_x
        # encoded_x=torch.cat([encoded_x,PE_x],dim=-1)
        # 
        
        # encoded_x[norm_x>=1] = self.encoding2(x[norm_x>=1].view(-1, self.num_dim)) # [N,32]
        # if weights is not None:
        #     encoded_x = encoded_x * weights
        #     _, n_features = encoded_x.shape
        #     assert n_features == len(weights)
        #     # repeats last set of features for 0 mask levels.
        
        #     available_features = encoded_x[:, weights > 0]
            
        #     if len(available_features) <= 0:
        #         assert False, "no features are selected!"
        #     assert len(available_features) > 0
            
        #     # BAA's simple update rule
            
        #     coarse_features = available_features[:, -self.n_features_per_level:]
        #     coarse_repeats = coarse_features.repeat(1, self.n_levels)
        #     weights+=0.1
        #     encoded_x = encoded_x * weights + coarse_repeats * (1 - weights) 
            
            # revise smooth interpolation
            # current_level=available_features.size(1)
            # if current_level>=4:
            #     w=weights[current_level-1]
            #     coarse_features = w*available_features[:, -self.n_features_per_level:] + (1-w)*available_features[:, -self.n_features_per_level-2:-self.n_features_per_level]
            # else:
            #     coarse_features = available_features[:, -self.n_features_per_level:]
            # coarse_repeats = coarse_features.repeat(1, self.n_levels)
            # encoded_x = encoded_x * weights + coarse_repeats * (1 - weights) 

        # encoded_x=torch.cat([encoded_x,PE_x],dim=-1)
        # encoded_x=encoded_x.to(torch.float32)
        
        
        # ic(PE_x.shape)
        # ic(encoded_x.shape)
        # encoded_x=torch.cat([encoded_x,PE_x],dim=-1)
        # ic(encoded_x.shape)
        # encoded_x[norm_x>=1] = self.encoding2(x[norm_x>=1].view(-1, self.num_dim)) # [N,32]
            # noise_x=x.view(-1, self.num_dim)+torch.randn_like(x)*((1-alpha).clip(0,0.01))
            # normal_encoded_x = self.encoding(noise_x) # [N,32]
            # available_features = normal_encoded_x[:, weights > 0]
            # coarse_features = available_features[:, -self.n_features_per_level:]
            # coarse_repeats = coarse_features.repeat(1, self.n_levels)
            # normal_encoded_x = normal_encoded_x * weights + coarse_repeats * (1 - weights) 
            
            
            # w1=self.gaussian_pdf(encoded_x.to(torch.float32), encoded_x.to(torch.float32), (1-alpha).clip(0,0.01))
            # w2=self.gaussian_pdf(normal_encoded_x.to(torch.float32), encoded_x.to(torch.float32), (1-alpha).clip(0,0.01))
            # w=w1+w2
            # encoded_x=encoded_x.to(torch.float32)*w1/w+normal_encoded_x.to(torch.float32)*w2/w
            
        x = (
            self.mlp_base(encoded_x)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        

        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        
        density = (
            self.density_activation(density_before_activation)
        )
        self.density=density
        
        
        # x_guide=self.depth_mlp(PE_x)
        # density_before_activation_guide, base_mlp_out_guide = torch.split(
        #     x_guide, [1, self.geo_feat_dim], dim=-1
        # )
        # density_guide = (
        #     self.density_activation(density_before_activation_guide)
        # )
        if return_feat:
            return density, base_mlp_out#+base_mlp_out_guide
        else:
            return density#*density_guide

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
        weights: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
            density, embedding = self.query_density(positions, weights=weights, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore

    
    def positional_encoding(self,positions, freqs, progress=1.0):
        levels = torch.arange(freqs, device=positions.device)
        freq_bands = (2**levels)  # (F,)
        mask = (progress * freqs - levels).clamp_(min=0.0, max=1) # linear anealing
        pts = positions[..., None] * freq_bands
        pts_sin = torch.sin(pts)*mask
        pts_cos = torch.cos(pts)*mask
        pts = torch.cat([pts_sin , pts_cos], dim=-1)
        pts = pts.reshape(positions.shape[:-1] + (freqs * 2 * positions.shape[-1], ))  # (..., DF)
        return pts