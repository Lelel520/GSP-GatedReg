import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from .common import get_correspondences, to_o3d_pcd, to_tsfm


class TDMatchDataset(Dataset):
    def __init__(self, infos, config, data_augmentation=True):
        super().__init__()
        self.infos = infos
        self.base_dir = config.root
        self.data_augmentation = data_augmentation
        self.config = config
        self.voxel_size = config.voxel_size
        self.search_voxel_size = config.overlap_radius
        self.points_lim = 30000
        self.rot_factor = 1.0
        self.augment_noise = config.augment_noise

    def __len__(self):
        return len(self.infos['rot'])

    def _limit_points(self, points):
        if points.shape[0] <= self.points_lim:
            return points
        indices = np.random.permutation(points.shape[0])[:self.points_lim]
        return points[indices]

    def _augment_pair(self, source, target, rotation, translation):
        angles = np.random.rand(3) * np.pi * 2.0 / self.rot_factor
        augmentation = Rotation.from_euler('zyx', angles).as_matrix()
        if np.random.rand(1)[0] > 0.5:
            source = np.matmul(augmentation, source.T).T
            rotation = np.matmul(rotation, augmentation.T)
        else:
            target = np.matmul(augmentation, target.T).T
            rotation = np.matmul(augmentation, rotation)
            translation = np.matmul(augmentation, translation)
        source += (np.random.rand(source.shape[0], 3) - 0.5) * self.augment_noise
        target += (np.random.rand(target.shape[0], 3) - 0.5) * self.augment_noise
        return source, target, rotation, translation

    def __getitem__(self, index):
        rotation = self.infos['rot'][index]
        translation = self.infos['trans'][index]
        source = torch.load(os.path.join(self.base_dir, self.infos['src'][index]))
        target = torch.load(os.path.join(self.base_dir, self.infos['tgt'][index]))
        source = self._limit_points(source)
        target = self._limit_points(target)

        if self.data_augmentation:
            source, target, rotation, translation = self._augment_pair(
                source, target, rotation, translation
            )
        if translation.ndim == 1:
            translation = translation[:, None]

        correspondences = get_correspondences(
            to_o3d_pcd(source), to_o3d_pcd(target),
            to_tsfm(rotation, translation), self.search_voxel_size,
        )
        source_features = np.ones_like(source[:, :1]).astype(np.float32)
        target_features = np.ones_like(target[:, :1]).astype(np.float32)
        return (
            source, target, source_features, target_features,
            rotation.astype(np.float32), translation.astype(np.float32),
            correspondences, source, target, torch.tensor([index]),
        )
