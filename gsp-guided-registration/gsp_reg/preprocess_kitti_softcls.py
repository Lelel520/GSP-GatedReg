import argparse
import logging
import os
import sys
import time
from functools import partial

import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsp_reg.dataset.kitti import KITTIDataset
from gsp_reg.modules.kpconv_backbone.architectures import architectures
from gsp_reg.modules.kpconv_backbone.preprocessing import collate_fn_descriptor, calibrate_neighbors
from gsp_reg.modules.soft_classified_loss import compute_stopo_chunked


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/kitti")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default="data/kitti/preprocessed")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--keep-ratio", type=float, default=0.95)
    p.add_argument("--samples-threshold", type=int, default=1200)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--first-subsampling-dl", type=float, default=0.3)
    p.add_argument("--overlap-radius", type=float, default=0.45)
    p.add_argument("--conv-radius", type=float, default=4.25)
    p.add_argument("--max-points", type=int, default=512)
    return p.parse_args()


def make_cfg(args):
    return edict({
        "root": args.root,
        "voxel_size": args.first_subsampling_dl,
        "overlap_radius": args.overlap_radius,
        "augment_noise": 0.0,
        "augment_shift_range": 2.0,
        "augment_scale_max": 1.2,
        "augment_scale_min": 0.8,
        "max_points": args.max_points,
        "architecture": architectures["kitti_full"],
        "first_subsampling_dl": args.first_subsampling_dl,
        "conv_radius": args.conv_radius,
        "deform_radius": 5.0,
        "num_kernel_points": 15,
        "num_layers": 4,
        "in_points_dim": 3,
        "in_feats_dim": 1,
        "first_feats_dim": 128,
        "final_feats_dim": 32,
        "intermediate_feats_dim": 32,
        "gnn_feats_dim": 256,
        "node_id": -1,
        "mode": "train" if args.split == "train" else "test",
        "patch_overlap_thres": 0.2,
        "KP_extent": 2.0,
        "batch_norm_momentum": 0.02,
        "use_batch_norm": True,
        "fixed_kernel_points": "center",
        "KP_influence": "linear",
        "aggregation_mode": "sum",
        "deformable": False,
        "modulated": False,
    })


def main():
    args = parse_args()
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("preprocess_kitti_softcls")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = make_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = KITTIDataset(cfg, args.split, data_augmentation=False)
    n_all = len(dataset)
    n_pairs = n_all if args.limit <= 0 else min(args.limit, n_all)
    log.info(f"KITTI split={args.split}, pairs={n_all}, preprocessing={n_pairs}")
    log.info(f"root={args.root}, output={args.output_dir}, device={device}")

    limits = calibrate_neighbors(
        dataset, cfg, collate_fn=collate_fn_descriptor,
        keep_ratio=args.keep_ratio, samples_threshold=args.samples_threshold,
    )
    limits = limits.tolist() if hasattr(limits, "tolist") else list(limits)
    log.info(f"Neighborhood limits: {limits}")
    collate = partial(collate_fn_descriptor, config=cfg, neighborhood_limits=limits)

    t0 = time.time()
    for idx in range(n_pairs):
        out_path = os.path.join(args.output_dir, f"pair_{idx:06d}.pth")
        if os.path.exists(out_path) and not args.overwrite:
            if (idx + 1) % 100 == 0:
                log.info(f"[{idx + 1:5d}/{n_pairs}] skipped existing")
            continue

        batch = collate([dataset[idx]])
        s_len = batch["stack_lengths"][0]
        src_n = int(s_len[0].item())
        tgt_n = int(s_len[1].item())
        points_l0 = batch["points"][0]
        with torch.no_grad():
            batch["stopo_src"] = compute_stopo_chunked(
                points_l0[:src_n].unsqueeze(0).to(device)
            )[0].cpu()
            batch["stopo_tgt"] = compute_stopo_chunked(
                points_l0[src_n:src_n + tgt_n].unsqueeze(0).to(device)
            )[0].cpu()
        if torch.is_tensor(batch.get("correspondences")):
            batch["correspondences"] = batch["correspondences"].long()
        batch["pair_idx"] = idx
        batch["dataset"] = "KITTI"
        batch["split"] = args.split
        torch.save(batch, out_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (idx + 1) % 50 == 0 or idx + 1 == n_pairs:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n_pairs - idx - 1)
            log.info(f"[{idx + 1:5d}/{n_pairs}] {elapsed/60:.1f}m ETA={eta/60:.1f}m")

    log.info(f"Done: saved to {args.output_dir}")


if __name__ == "__main__":
    main()
