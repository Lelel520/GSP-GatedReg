# KPConv encoder-decoder; see licenses/KPConv-MIT.txt.
import os
import torch
import torch.nn as nn
import numpy as np


class KPConvBackbone(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        from .modules import block_decider
        from .architectures import architectures
        from easydict import EasyDict as edict

        arch_name = os.environ.get('KP_ARCH', 'tdmatch_full')
        if arch_name not in architectures:
            raise ValueError(f'Unknown KPConv architecture: {arch_name}')
        is_kitti = arch_name == 'kitti_full'
        arch = architectures[arch_name]
        first_subsampling_dl = float(os.environ.get(
            'KP_FIRST_SUBSAMPLING_DL', '0.3' if is_kitti else '0.025'))
        conv_radius = float(os.environ.get(
            'KP_CONV_RADIUS', '4.25' if is_kitti else '2.5'))
        kp_extent = float(os.environ.get('KP_EXTENT', '2.0'))
        deform_radius = float(os.environ.get('KP_DEFORM_RADIUS', '5.0'))
        kp_cfg = edict({
            'in_points_dim': 3, 'in_feats_dim': 1, 'in_features_dim': 1,
            'first_subsampling_dl': first_subsampling_dl, 'conv_radius': conv_radius,
            'first_feats_dim': 128, 'first_features_dim': 128,
            'final_feats_dim': 32, 'intermediate_feats_dim': 32,
            'gnn_feats_dim': 256, 'num_kernel_points': 15,
            'KP_extent': kp_extent, 'deform_radius': deform_radius,
            'batch_norm_momentum': 0.02, 'use_batch_norm': True,
            'fixed_kernel_points': 'center', 'KP_influence': 'linear',
            'aggregation_mode': 'sum', 'deformable': False,
            'modulated': False, 'architecture': arch, 'gpu_mode': True,
        })
        self.out_dim = out_dim
        self.K = kp_cfg.num_kernel_points
        self.desc_head = None

        layer = 0
        r = kp_cfg.first_subsampling_dl * kp_cfg.conv_radius
        in_dim = kp_cfg.in_feats_dim
        out_dim_k = kp_cfg.first_feats_dim

        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        for block_i, block in enumerate(arch):
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
            if 'upsample' in block:
                break
            self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim_k, layer, kp_cfg))
            if 'simple' in block:
                in_dim = out_dim_k // 2
            else:
                in_dim = out_dim_k
            if 'pool' in block or 'strided' in block:
                layer += 1
                r *= 2
                out_dim_k *= 2

        start_i = 0
        for block_i, block in enumerate(arch):
            if 'upsample' in block:
                start_i = block_i
                break

        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        for block_i, block in enumerate(arch[start_i:]):
            if block_i > 0 and 'upsample' in arch[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)
            self.decoder_blocks.append(block_decider(block, r, in_dim, out_dim_k, layer, kp_cfg))
            in_dim = out_dim_k
            if 'upsample' in block:
                layer -= 1
                r *= 0.5
                out_dim_k = out_dim_k // 2

        decoder_out_dim = getattr(self.decoder_blocks[-1], 'out_dim', in_dim)
        self.decoder_out_dim = decoder_out_dim
        self.desc_head = nn.Sequential(
            nn.Linear(decoder_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, out_dim),
        )
        nn.init.normal_(self.desc_head[3].weight, std=0.01)
        nn.init.zeros_(self.desc_head[3].bias)

    def forward(self, batch_inputs: dict, return_decoder_features: bool = False):
        x = batch_inputs['features'].clone().detach()

        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch_inputs)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch_inputs)

        decoder_features = x
        x = self.desc_head(x)
        descriptors = nn.functional.normalize(x, p=2, dim=-1)
        # Keep the unused second output for compatibility with existing callers.
        if return_decoder_features:
            return descriptors, None, decoder_features
        return descriptors, None
