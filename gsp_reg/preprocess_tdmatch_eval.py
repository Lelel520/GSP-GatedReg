import argparse
import os
import pickle
import time
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from easydict import EasyDict as edict

from gsp_reg.dataset.tdmatch import TDMatchDataset
from gsp_reg.modules.kpconv_backbone.architectures import architectures
from gsp_reg.modules.kpconv_backbone.preprocessing import collate_fn_descriptor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="3DLoMatch", choices=["3DMatch", "3DLoMatch"])
    parser.add_argument("--info", default="")
    parser.add_argument("--data-root", default="data/3DMatch")
    parser.add_argument("--output_dir", default="data/3DMatch/preprocessed_3dlomatch")
    parser.add_argument("--limits", default="36,39,41,46")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    info_path = args.info or f"gsp_reg/benchmark_configs/tdmatch/{args.benchmark}.pkl"
    with open(info_path, "rb") as handle:
        info = pickle.load(handle)
    cfg = edict({"root":args.data_root, "voxel_size":0.025, "overlap_radius":0.0375,
        "augment_noise":0.0, "architecture":architectures["tdmatch_full"],
        "first_subsampling_dl":0.025, "conv_radius":2.5, "deform_radius":5.0,
        "num_kernel_points":15, "num_layers":5, "in_points_dim":3, "in_feats_dim":1,
        "first_feats_dim":128, "final_feats_dim":32, "intermediate_feats_dim":32,
        "gnn_feats_dim":256, "node_id":-3, "mode":"val", "patch_overlap_thres":0.3})
    dataset = TDMatchDataset(info, cfg, data_augmentation=False)
    limits = [int(x) for x in args.limits.split(",")]
    if len(limits) != 4:
        raise ValueError("--limits must contain four comma-separated integers")
    os.makedirs(args.output_dir, exist_ok=True)
    collate = partial(collate_fn_descriptor, config=cfg, neighborhood_limits=limits)
    print("benchmark=%s pairs=%d limits=%s output=%s" % (args.benchmark, len(dataset), limits, args.output_dir), flush=True)
    start = time.time()
    for idx in range(len(dataset)):
        output = os.path.join(args.output_dir, "pair_%06d.pth" % idx)
        if os.path.exists(output) and not args.overwrite:
            continue
        batch = collate([dataset[idx]])
        batch["pair_idx"] = idx
        torch.save(batch, output)
        if (idx + 1) % 50 == 0 or idx + 1 == len(dataset):
            elapsed = time.time() - start
            print("[%d/%d] %.1fs elapsed, ETA %.1fs" %
                  (idx + 1, len(dataset), elapsed, elapsed / (idx + 1) * (len(dataset) - idx - 1)), flush=True)
    print("done: %d pairs in %.1f min" % (len(dataset), (time.time() - start) / 60.0), flush=True)


if __name__ == "__main__":
    main()
