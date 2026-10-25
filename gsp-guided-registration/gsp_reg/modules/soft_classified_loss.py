import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def _soft_gate_from_response(
    response: torch.Tensor,
    tau_low: float = 0.03,
    tau_high: float = 0.20,
) -> torch.Tensor:
    return ((response - tau_low) / (tau_high - tau_low + 1e-8)).clamp(0.0, 1.0)


def _stopo_from_knn(
    xyz: torch.Tensor,
    knn_idx: torch.Tensor,
    tau_low: float = 0.03,
    tau_high: float = 0.20,
) -> torch.Tensor:
    """Combine surface variation and normal-boundary response with weights 0.6/0.4."""
    B, N, _ = xyz.shape
    K = knn_idx.shape[-1]
    device = xyz.device

    batch_idx = torch.arange(B, device=device).view(B, 1, 1)
    neighbors = xyz[batch_idx, knn_idx]
    centered = neighbors - xyz.unsqueeze(2)
    C_mat = torch.einsum('bnki,bnkj->bnij', centered, centered) / max(K, 1)
    eye = torch.eye(3, device=device, dtype=xyz.dtype)
    C_mat = C_mat + 1e-8 * eye.view(1, 1, 3, 3)

    # torch.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = torch.linalg.eigh(C_mat)
    eig_sum = eigenvalues.sum(dim=-1) + 1e-10
    curvature = (eigenvalues[:, :, 0] / eig_sum).clamp(0.0, 1.0)

    normals = eigenvectors[:, :, :, 0]
    normals = torch.nn.functional.normalize(normals, p=2, dim=-1)
    neighbor_normals = normals[batch_idx, knn_idx]
    normal_dot = (neighbor_normals * normals.unsqueeze(2)).sum(dim=-1).abs()
    boundary = (1.0 - normal_dot).mean(dim=-1).clamp(0.0, 1.0)

    response = (0.6 * curvature + 0.4 * boundary).clamp_min(0.0)
    return _soft_gate_from_response(response, tau_low=tau_low, tau_high=tau_high)


def compute_stopo(
    xyz: torch.Tensor,
    K: int = 30,
    tau_low: float = 0.03,
    tau_high: float = 0.20,
) -> torch.Tensor:
    """Compute geometry gates using fixed response thresholds."""
    B, N, _ = xyz.shape
    if N <= 1:
        return torch.zeros(B, N, device=xyz.device, dtype=xyz.dtype)

    K_eff = min(K, N - 1)
    sq_norm = (xyz ** 2).sum(dim=-1, keepdim=True)
    cross = torch.bmm(xyz, xyz.transpose(1, 2))
    sq_dist = (sq_norm + sq_norm.transpose(1, 2) - 2.0 * cross).clamp(min=0.0)
    _, knn_idx = torch.topk(sq_dist, k=K_eff + 1, dim=-1, largest=False)
    knn_idx = knn_idx[:, :, 1:]
    return _stopo_from_knn(xyz, knn_idx, tau_low=tau_low, tau_high=tau_high)


def compute_stopo_chunked(
    xyz: torch.Tensor,
    K: int = 30,
    chunk_size: int = 2048,
    max_full_points: int = 10000,
    tau_low: float = 0.03,
    tau_high: float = 0.20,
) -> torch.Tensor:
    """Compute geometry gates with chunked nearest-neighbor search."""
    B, N, _ = xyz.shape
    if N <= max_full_points:
        return compute_stopo(xyz, K=K, tau_low=tau_low, tau_high=tau_high)
    if N <= 1:
        return torch.zeros(B, N, device=xyz.device, dtype=xyz.dtype)

    K_eff = min(K, N - 1)
    out = []
    for b in range(B):
        xb = xyz[b].contiguous()
        sq_all = (xb ** 2).sum(dim=-1).view(1, N)
        chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = xb[start:end].contiguous()
            sq_chunk = (x_chunk ** 2).sum(dim=-1, keepdim=True)
            sq_dist = (sq_chunk + sq_all - 2.0 * x_chunk @ xb.t()).clamp(min=0.0)
            _, idx = torch.topk(sq_dist, k=K_eff + 1, dim=-1, largest=False)
            chunks.append(idx[:, 1:])
            del sq_dist
        knn_idx = torch.cat(chunks, dim=0).unsqueeze(0)
        out.append(_stopo_from_knn(
            xb.unsqueeze(0), knn_idx, tau_low=tau_low, tau_high=tau_high
        )[0])
    return torch.stack(out, dim=0)

class SoftClassifiedLoss(nn.Module):
    """Geometry-gated GSP, circle-inspired metric loss, and variance regularization."""

    def __init__(
        self,
        lambda_deg: float = 1.0,
        lambda_dist: float = 1.0,
        lambda_var: float = 1.0,
        var_gamma: float = 0.05,
        m_cross: float = 0.10,
        safe_radius: float = 0.10,
        alpha: float = 1.0,
        beta: float = 1.0,
        theta: float = 0.2,
        circle_pos_margin: Optional[float] = None,
        circle_neg_margin: float = 1.40,
        circle_log_scale: float = 10.0,
    ):
        super().__init__()
        self.lambda_deg = lambda_deg
        self.lambda_dist = lambda_dist
        self.lambda_var = lambda_var
        self.var_gamma = var_gamma
        self.circle_pos_margin = m_cross if circle_pos_margin is None else circle_pos_margin
        self.circle_neg_margin = circle_neg_margin
        self.circle_log_scale = circle_log_scale
        self.m_cross = self.circle_pos_margin
        self.safe_radius = safe_radius
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def forward(
        self,
        F_p: torch.Tensor,          # [1, N, D] source descriptors
        F_q: torch.Tensor,          # [1, M, D] target descriptors
        X_p: torch.Tensor,          # [1, N, 3] source coords
        X_q: torch.Tensor,          # [1, M, 3] target coords
        anchor_idx: torch.Tensor,   # [K] anchor indices in source
        pos_idx: torch.Tensor,      # [K] positive indices in target
        delta_p: torch.Tensor,      # [1, N] GSP high-pass energy
        s_topo_p: torch.Tensor,     # [1, N] geometric uniqueness scores
        grad_targets=None,          # optional tensors used for grad-ratio diagnostics
        compute_grad_ratio: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return the total loss and scalar diagnostics."""
        K = anchor_idx.shape[0]
        if K == 0:
            zero = torch.tensor(0.0, device=F_p.device, requires_grad=True)
            return zero, {"deg": 0.0, "dist": 0.0, "var": 0.0}

        s_a = s_topo_p[0, anchor_idx].clamp(0.0, 1.0)          # [K]
        w_deg = (1.0 - s_a).clamp(0.0, 1.0).pow(self.alpha).to(dtype=F_p.dtype)
        w_dist = s_a.pow(self.beta).to(dtype=F_p.dtype)
        delta_a = delta_p[0, anchor_idx]                       # [K]
        f_a = F_p[0, anchor_idx]                               # [K, D]
        f_pos = F_q[0, pos_idx]                                # [K, D]
        x_a = X_p[0, anchor_idx]                               # [K, 3]
        x_pos = X_q[0, pos_idx]                                # [K, 3]

        # GSP residual in low-response regions.
        loss_deg = (w_deg * delta_a.pow(2)).sum() / w_deg.sum().clamp_min(1e-8)

        # Circle-inspired metric loss in high-response regions.
        f_a_circle = F.normalize(f_a, p=2, dim=-1)
        f_pos_circle = F.normalize(f_pos, p=2, dim=-1)
        cos_sim = torch.mm(f_a_circle, f_pos_circle.t()).clamp(-1.0, 1.0)
        desc_dist = (2.0 - 2.0 * cos_sim).clamp(min=1e-12, max=4.0).sqrt()  # [K, K]
        diag_idx = torch.arange(K, device=f_a.device)
        pos_dist = desc_dist[diag_idx, diag_idx]

        # Exclude positive pairs and spatially close points from the negative set.
        sq_xa = (x_a ** 2).sum(dim=-1, keepdim=True)
        sq_xpos = (x_pos ** 2).sum(dim=-1, keepdim=True)
        cross_x = torch.mm(x_a, x_pos.t())
        spatial_dist = (sq_xa + sq_xpos.t() - 2.0 * cross_x).clamp(min=1e-12).sqrt()
        valid_neg = spatial_dist > self.safe_radius
        valid_neg[diag_idx, diag_idx] = False

        log_scale = max(float(self.circle_log_scale), 1e-8)
        pos_weight = torch.clamp(pos_dist - self.circle_pos_margin, min=0.0).detach()
        pos_logits = log_scale * (pos_dist - self.circle_pos_margin) * pos_weight

        neg_weight = torch.clamp(self.circle_neg_margin - desc_dist, min=0.0).detach()
        neg_logits = log_scale * (self.circle_neg_margin - desc_dist) * neg_weight
        neg_logits = neg_logits.masked_fill(~valid_neg, -1e5)
        lse_neg_row = torch.logsumexp(neg_logits, dim=-1)
        lse_neg_col = torch.logsumexp(neg_logits, dim=0)

        row_loss = F.softplus(pos_logits + lse_neg_row) / log_scale
        col_loss = F.softplus(pos_logits + lse_neg_col) / log_scale
        w_dist_sum = w_dist.sum().clamp_min(1e-8)
        loss_dist = 0.5 * (
            (w_dist * row_loss).sum() / w_dist_sum
            + (w_dist * col_loss).sum() / w_dist_sum
        )

        desc_masked = desc_dist.masked_fill(~valid_neg, float('inf'))
        has_valid = valid_neg.any(dim=-1)
        neg_dist = torch.zeros_like(pos_dist)
        if has_valid.any():
            neg_dist[has_valid] = desc_masked[has_valid].min(dim=-1).values

        valid_w = w_dist * has_valid.to(dtype=F_p.dtype)
        valid_w_sum = valid_w.sum().clamp_min(1e-8)
        dpos = pos_dist.mean()
        if has_valid.any():
            dpos_valid = pos_dist[has_valid].mean()
            dneg = neg_dist[has_valid].mean()
            gap = dneg - dpos_valid
        else:
            dneg = F_p.new_tensor(0.0)
            gap = F_p.new_tensor(0.0)
        dpos_w = (w_dist * pos_dist).sum() / w_dist_sum
        dpos_valid_w = (valid_w * pos_dist).sum() / valid_w_sum
        dneg_w = (valid_w * neg_dist).sum() / valid_w_sum
        gap_w = dneg_w - dpos_valid_w

        pos_active = pos_dist > self.circle_pos_margin
        neg_active = ((desc_dist < self.circle_neg_margin) & valid_neg).any(dim=-1)
        active_vec = (pos_active | neg_active).to(dtype=F_p.dtype)
        active = active_vec.mean()
        active_w = (w_dist * active_vec).sum() / w_dist_sum
        if has_valid.any():
            circle_acc = (pos_dist[has_valid] < neg_dist[has_valid]).to(dtype=F_p.dtype).mean()
            circle_acc_w = (valid_w * (pos_dist < neg_dist).to(dtype=F_p.dtype)).sum() / valid_w_sum
        else:
            circle_acc = F_p.new_tensor(0.0)
            circle_acc_w = F_p.new_tensor(0.0)

        # Weighted variance floor in high-response regions.
        f_all = F_p.reshape(-1, F_p.shape[-1])
        w_var = s_topo_p.reshape(-1).clamp(0.0, 1.0).pow(self.beta).to(dtype=F_p.dtype)
        w_var_sum_raw = w_var.sum()
        if w_var_sum_raw.detach().item() > 1e-6:
            w_var_sum = w_var_sum_raw.clamp_min(1e-8)
            mu = (w_var.unsqueeze(-1) * f_all).sum(dim=0) / w_var_sum
            var_dim = (w_var.unsqueeze(-1) * (f_all - mu).pow(2)).sum(dim=0) / w_var_sum
            sigma_dim = torch.sqrt(var_dim.clamp_min(0.0) + 1e-12)
            loss_var = F.relu(self.var_gamma - sigma_dim).pow(2).mean()
            sigma_mean = sigma_dim.mean()
            sigma_min = sigma_dim.min()
        else:
            loss_var = F_p.sum() * 0.0
            sigma_mean = F_p.new_tensor(0.0)
            sigma_min = F_p.new_tensor(0.0)

        weighted_deg = self.lambda_deg * loss_deg
        weighted_dist = self.lambda_dist * loss_dist
        weighted_var = self.lambda_var * loss_var
        total = weighted_deg + weighted_dist + weighted_var

        loss_dict = {
            "deg": loss_deg.item(),
            "dist": loss_dist.item(),
            "var": loss_var.item(),
            "s_mean": s_a.mean().item(),
            "w_deg_sum": w_deg.sum().item(),
            "w_dist_sum": w_dist.sum().item(),
            "w_var_sum": w_var_sum_raw.item(),
            "sigma_mean": sigma_mean.item(),
            "sigma_min": sigma_min.item(),
            "dpos": dpos.item(),
            "dneg": dneg.item(),
            "gap": gap.item(),
            "active": active.item(),
            "dpos_w": dpos_w.item(),
            "dneg_w": dneg_w.item(),
            "gap_w": gap_w.item(),
            "active_w": active_w.item(),
            "circle_acc": circle_acc.item(),
            "circle_acc_w": circle_acc_w.item(),
            "margin": self.circle_pos_margin,
            "pos_margin": self.circle_pos_margin,
            "neg_margin": self.circle_neg_margin,
            "log_scale": self.circle_log_scale,
            "var_gamma": self.var_gamma,
        }
        if compute_grad_ratio and grad_targets is not None:
            targets = tuple(t for t in grad_targets if isinstance(t, torch.Tensor) and t.requires_grad)
            if targets:
                g_deg = self._grad_norm(weighted_deg, targets)
                g_dist = self._grad_norm(weighted_dist, targets)
                g_var = self._grad_norm(weighted_var, targets)
                denom = g_deg + g_dist + g_var + 1e-12
                loss_dict.update({
                    "g_deg": g_deg.item(),
                    "g_dist": g_dist.item(),
                    "g_var": g_var.item(),
                    "r_deg": (g_deg / denom).item(),
                    "r_dist": (g_dist / denom).item(),
                    "r_var": (g_var / denom).item(),
                })
        return total, loss_dict

    @staticmethod
    def _grad_norm(loss: torch.Tensor, targets) -> torch.Tensor:
        if not loss.requires_grad:
            return loss.new_tensor(0.0)
        grads = torch.autograd.grad(
            loss, targets, retain_graph=True, allow_unused=True, create_graph=False
        )
        total = loss.new_tensor(0.0)
        for grad in grads:
            if grad is not None:
                total = total + grad.detach().pow(2).sum()
        return total.clamp_min(0.0).sqrt()
