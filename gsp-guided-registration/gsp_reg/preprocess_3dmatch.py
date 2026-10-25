import os, sys, pickle, time, argparse
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsp_reg.dataset.tdmatch import TDMatchDataset
from gsp_reg.modules.kpconv_backbone.preprocessing import collate_fn_descriptor, calibrate_neighbors
from gsp_reg.modules.kpconv_backbone.architectures import architectures
from gsp_reg.modules.soft_classified_loss import compute_stopo_chunked
from easydict import EasyDict as edict
from functools import partial
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('preprocess')

parser = argparse.ArgumentParser(description='Preprocess 3DMatch training pairs.')
parser.add_argument('--data-root', default='data/3DMatch')
parser.add_argument('--output-dir', default='data/3DMatch/preprocessed')
parser.add_argument('--limit', type=int, default=5000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
device = torch.device(args.device)

dataset_cfg = edict({
    'root': args.data_root, 'voxel_size': 0.025, 'overlap_radius': 0.0375,
    'augment_noise': 0.0, 'architecture': architectures['tdmatch_full'],
    'first_subsampling_dl': 0.025, 'conv_radius': 2.5, 'deform_radius': 5.0,
    'num_kernel_points': 15, 'num_layers': 5, 'in_points_dim': 3,
    'in_feats_dim': 1, 'first_feats_dim': 128, 'final_feats_dim': 32,
    'intermediate_feats_dim': 32, 'gnn_feats_dim': 256, 'node_id': -3,
    'mode': 'test', 'patch_overlap_thres': 0.3,
})

with open('gsp_reg/benchmark_configs/tdmatch/train_info.pkl', 'rb') as f:
    train_info = pickle.load(f)

np.random.seed(args.seed)
indices = np.random.permutation(len(train_info['src']))[:args.limit]
sampled_info = {k: [v[i] for i in indices] for k, v in train_info.items()}
train_set = TDMatchDataset(sampled_info, dataset_cfg, data_augmentation=False)
n_pairs = len(train_set)
log.info(f'Sampled {n_pairs} pairs for preprocessing')

limits = calibrate_neighbors(train_set, dataset_cfg, collate_fn=collate_fn_descriptor,
                              keep_ratio=0.95, samples_threshold=2000)
log.info(f'Limits: {limits.tolist()}')

collate = partial(collate_fn_descriptor, config=dataset_cfg, neighborhood_limits=limits)

# Cached gates must follow the point order in points[0].
stopo_cache = {}


def aligned_stopo(frame_path, xyz_l0):
    cached = stopo_cache.get(frame_path)
    if cached is not None and cached.shape[0] == xyz_l0.shape[0]:
        return cached
    with torch.no_grad():
        s_topo = compute_stopo_chunked(xyz_l0.unsqueeze(0).to(device))[0].cpu()
    stopo_cache[frame_path] = s_topo
    torch.cuda.empty_cache()
    return s_topo


log.info('Preprocessing all pairs (collate_fn once)...')
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'selected_pair_indices.pkl'), 'wb') as handle:
    pickle.dump(indices.tolist(), handle)
t0 = time.time()
for idx in range(n_pairs):
    batch = collate([train_set[idx]])
    src_path = sampled_info['src'][idx]
    tgt_path = sampled_info['tgt'][idx]
    s_len = batch['stack_lengths'][0]
    src_N = int(s_len[0].item())
    tgt_N = int(s_len[1].item())
    points_l0 = batch['points'][0]
    batch['stopo_src'] = aligned_stopo(src_path, points_l0[:src_N])
    batch['stopo_tgt'] = aligned_stopo(tgt_path, points_l0[src_N:src_N + tgt_N])
    batch['pair_idx'] = idx
    torch.save(batch, os.path.join(args.output_dir, f'pair_{idx:06d}.pth'))
    if (idx+1) % 1000 == 0:
        elapsed = time.time() - t0
        log.info(f'  [{idx+1:5d}/{n_pairs}] {elapsed:.0f}s ETA {elapsed/(idx+1)*n_pairs:.0f}s')

elapsed = time.time() - t0
log.info(f'Done: {n_pairs} pairs in {elapsed/60:.1f}m')
log.info(f'Saved to {args.output_dir}')

