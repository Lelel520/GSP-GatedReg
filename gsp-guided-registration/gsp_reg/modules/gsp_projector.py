import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sampling(xyz: torch.Tensor, K: int):
    """Sample K centers from xyz [B, N, 3]; return centers and indices."""
    B, N, _ = xyz.shape
    device = xyz.device
    center_idx = torch.zeros(B, K, dtype=torch.long, device=device)
    dist = torch.full((B, N), float('inf'), device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    for k in range(K):
        center_idx[:, k] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest]
        d = ((xyz - centroid.unsqueeze(1)) ** 2).sum(-1)
        dist = torch.min(dist, d)
        farthest = dist.argmax(-1)
    centers = xyz[torch.arange(B, device=device).unsqueeze(1), center_idx]
    return centers, center_idx


class GSPProjector(nn.Module):
    """Spatial low-pass reconstruction C^T D^-1 C F, with D = diag(C 1)."""

    def __init__(self, num_superpoints: int = 128, sigma: float = 0.2):
        super().__init__()
        self.K = num_superpoints
        self.sigma = sigma

    def compute_assignment(self, xyz: torch.Tensor):
        """Return detached assignments [B, K, N], normalized over centers."""
        centers, _ = farthest_point_sampling(xyz, self.K)
        sq_c = (centers ** 2).sum(dim=-1, keepdim=True)
        sq_x = (xyz ** 2).sum(dim=-1, keepdim=True)
        cross = torch.bmm(centers, xyz.transpose(1, 2))
        sq_dist = (sq_c + sq_x.transpose(1, 2) - 2.0 * cross).clamp(min=0.0)
        return F.softmax(-sq_dist / (self.sigma ** 2), dim=1).detach()

    @staticmethod
    def delta_from_assignment(feat: torch.Tensor, C: torch.Tensor):
        C = C.to(device=feat.device, dtype=feat.dtype)
        # Normalize by superpoint mass to preserve constant descriptor fields.
        mass = C.sum(dim=2, keepdim=True).clamp_min(1e-6)
        super_feat = torch.bmm(C, feat) / mass
        feat_low = torch.bmm(C.transpose(1, 2), super_feat)
        return (feat - feat_low).norm(p=2, dim=-1)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        """Return residual norms [B, N] and assignments [B, K, N]."""
        C = self.compute_assignment(xyz)
        delta = self.delta_from_assignment(feat, C)
        return delta, C
