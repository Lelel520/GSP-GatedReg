import os, sys, time, pickle, logging, argparse
import numpy as np
import torch
from functools import partial
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsp_reg.model_builder import TrainConfig, ModelManager
from gsp_reg.confidence import validate_predator
from gsp_reg.dataset.tdmatch import TDMatchDataset
from gsp_reg.modules.kpconv_backbone.preprocessing import collate_fn_descriptor
from gsp_reg.modules.kpconv_backbone.architectures import architectures

parser = argparse.ArgumentParser()
parser.add_argument('ckpt')
parser.add_argument('--data_root', default='data/3DMatch')
parser.add_argument('--pairs', type=int, default=0, help='0 = full validation set')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_desc', type=int, default=5000)
parser.add_argument('--ransac_iters', type=int, default=50000)
parser.add_argument('--inlier_thresh', type=float, default=0.10, help='tau1: GT inlier distance threshold in meters')
parser.add_argument('--fmr_thresh', type=float, default=0.05, help='tau2: FMR threshold on GT inlier ratio')
parser.add_argument('--rr_rmse', type=float, default=0.20, help='RR threshold on registration RMSE in meters')
parser.add_argument('--log', type=str, default='')
parser.add_argument('--sample_mode', default='random', choices=['random', 'predator_prob'])
parser.add_argument('--predator_ckpt', type=str, default='third_party/OverlapPredator/weights/indoor.pth')
parser.add_argument('--predator_root', type=str, default='third_party/OverlapPredator')
parser.add_argument('--benchmark', default='3DMatch', choices=['val', '3DMatch', '3DLoMatch'])
parser.add_argument('--info', type=str, default='', help='explicit info pkl path; overrides --benchmark')
parser.add_argument('--preprocessed_dir', type=str, default='', help='directory of cached pair_*.pth batches; skips raw collation')
args = parser.parse_args()
if args.sample_mode == 'predator_prob':
    try:
        validate_predator(args.predator_root, args.predator_ckpt, 'indoor')
    except (FileNotFoundError, ImportError) as error:
        parser.error(str(error))

os.makedirs('logs', exist_ok=True)
log_path = args.log or f'logs/val_eval_{os.path.basename(args.ckpt).replace(".pth", "")}_{time.strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger('val_eval')

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
log.info(f'device={device} gpu={torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "cpu"}')
log.info(f'checkpoint={args.ckpt}')
log.info(f'sample_mode={args.sample_mode}')
log.info(f'Indoor metrics: tau1={args.inlier_thresh}m, FMR=IR_gt>{args.fmr_thresh}, RR=RMSE<{args.rr_rmse}m, RANSAC iters={args.ransac_iters}')
if args.sample_mode == 'predator_prob':
    log.info('Confidence sampling: official PREDATOR overlap * matchability')
log.info(f'log={log_path}')

rng = np.random.default_rng(123)
torch.manual_seed(123)

cfg = edict({
    'root': args.data_root, 'voxel_size': 0.025, 'overlap_radius': 0.0375,
    'augment_noise': 0.0, 'architecture': architectures['tdmatch_full'],
    'first_subsampling_dl': 0.025, 'conv_radius': 2.5, 'deform_radius': 5.0,
    'num_kernel_points': 15, 'num_layers': 5, 'in_points_dim': 3,
    'in_feats_dim': 1, 'first_feats_dim': 128, 'final_feats_dim': 32,
    'intermediate_feats_dim': 32, 'gnn_feats_dim': 256, 'node_id': -3,
    'mode': 'val', 'patch_overlap_thres': 0.3,
})
benchmark_paths = {
    'val': 'gsp_reg/benchmark_configs/tdmatch/val_info.pkl',
    '3DMatch': 'gsp_reg/benchmark_configs/tdmatch/3DMatch.pkl',
    '3DLoMatch': 'gsp_reg/benchmark_configs/tdmatch/3DLoMatch.pkl',
}
info_path = args.info or benchmark_paths[args.benchmark]
info = pickle.load(open(info_path, 'rb'))
if args.pairs > 0:
    for k in info:
        info[k] = info[k][:args.pairs]
log.info(f'benchmark={args.benchmark} info={info_path}')
log.info(f'validation pairs={len(info["src"])}')
if 'overlap' in info and len(info['overlap']) > 0:
    ov = np.array(info['overlap'], dtype=float)
    log.info(f'overlap: min={ov.min():.4f} max={ov.max():.4f} mean={ov.mean():.4f}')

if args.preprocessed_dir:
    class CachedPairDataset(torch.utils.data.Dataset):
        def __init__(self, directory):
            self.files = [os.path.join(directory, f'pair_{i:06d}.pth') for i in range(len(info['src']))]
            missing = [p for p in self.files if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f'{len(missing)} cached pairs are missing under {directory}; run preprocess_tdmatch_eval.py first')
        def __len__(self):
            return len(self.files)
        def __getitem__(self, index):
            return torch.load(self.files[index], map_location='cpu')
    ds = CachedPairDataset(args.preprocessed_dir)
    log.info(f'using cached preprocessed pairs: {args.preprocessed_dir}')
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                         collate_fn=lambda items: items[0], pin_memory=True)
else:
    ds = TDMatchDataset(info, cfg, data_augmentation=False)
    limits = [36, 39, 41, 46]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=partial(collate_fn_descriptor, config=cfg, neighborhood_limits=limits),
        pin_memory=True)


long_keys = {'neighbors', 'pools', 'upsamples'}
def to_gpu(k, x):
    if torch.is_tensor(x):
        y = x.to(device, non_blocking=True)
        return y.long() if k in long_keys else y
    if isinstance(x, list):
        out = []
        for v in x:
            if torch.is_tensor(v):
                v = v.to(device, non_blocking=True)
                if k in long_keys:
                    v = v.long()
            out.append(v)
        return out
    return x

def rotation_error(Rp, Rg):
    return float(np.arccos(np.clip((np.trace(Rg.T @ Rp) - 1) / 2, -1.0, 1.0)) * 180 / np.pi)

def rigid_fit_svd(src, tgt):
    if len(src) < 3:
        return np.eye(3), np.zeros(3)
    sm = src.mean(0); tm = tgt.mean(0)
    H = (src - sm).T @ (tgt - tm)
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.eye(3), np.zeros(3)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = tm - R @ sm
    return R, t


def ransac_svd(src, tgt, n_iter, thresh):
    M = len(src)
    if M < 3:
        return np.eye(3), np.zeros(3), np.zeros(M, dtype=bool)
    best = 0; bR = np.eye(3); bt = np.zeros(3); bm = np.zeros(M, dtype=bool)
    for _ in range(n_iter):
        ids = rng.choice(M, 3, replace=False)
        s = src[ids]; t = tgt[ids]
        sm = s.mean(0); tm = t.mean(0)
        H = (s - sm).T @ (t - tm)
        try:
            U, _, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            continue
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        tr = tm - R @ sm
        err = np.linalg.norm(src @ R.T + tr - tgt, axis=1)
        m = err < thresh
        c = int(m.sum())
        if c > best:
            best = c; bR = R; bt = tr; bm = m
            if best > 0.9 * M:
                break
    if int(bm.sum()) >= 3:
        bR, bt = rigid_fit_svd(src[bm], tgt[bm])
        err = np.linalg.norm(src @ bR.T + bt - tgt, axis=1)
        bm = err < thresh
    return bR, bt, bm

cfg_train = TrainConfig()
cfg_train._cfg['system']['device'] = str(device)
mgr = ModelManager(cfg_train, log)
mgr.build_model(stage=1)
ckpt = torch.load(args.ckpt, map_location=device)
state = ckpt
if isinstance(ckpt, dict):
    for key in ('state_dict', 'model_state_dict', 'backbone_state_dict', 'backbone', 'model'):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
cur = mgr.backbone.state_dict(); matched = {}; skipped = []
for k, v in state.items():
    mk = k[9:] if k.startswith('backbone.') else k
    if mk in cur and torch.is_tensor(v) and tuple(cur[mk].shape) == tuple(v.shape):
        matched[mk] = v
    else:
        skipped.append(k)
if set(matched) != set(cur):
    raise ValueError('Incomplete descriptor checkpoint: missing keys %s' % sorted(set(cur) - set(matched)))
cur.update(matched)
mgr.backbone.load_state_dict(cur, strict=True)
mgr.backbone.to(device).eval()
log.info(f'loaded tensors: matched={len(matched)} skipped={len(skipped)}')

predator_model = None
if args.sample_mode == 'predator_prob':
    predator_root = os.path.abspath(args.predator_root)
    if predator_root not in sys.path:
        sys.path.insert(0, predator_root)
    from easydict import EasyDict as _EasyDict
    from models.architectures import KPFCNN as _PredatorKPFCNN
    from configs.models import architectures as _predator_architectures
    from lib.utils import load_config as _load_predator_config

    predator_config = _EasyDict(_load_predator_config(
        os.path.join(predator_root, 'configs/test/indoor.yaml')
    ))
    predator_config.architecture = _predator_architectures['indoor']
    original_cwd = os.getcwd()
    try:
        os.chdir(predator_root)
        predator_model = _PredatorKPFCNN(predator_config).to(device)
    finally:
        os.chdir(original_cwd)
    predator_payload = torch.load(args.predator_ckpt, map_location=device)
    predator_state = predator_payload.get('state_dict', predator_payload)
    predator_model.load_state_dict(predator_state, strict=True)
    predator_model.eval()
    log.info(
        f'loaded PREDATOR: params={sum(p.numel() for p in predator_model.parameters()):,}'
    )

stats = {key: [] for key in ['ir_gt', 'ir_pred', 're', 'te', 'matches', 'rmse']}
fmr_success = rr_success = total_eval = reg_valid = bad = 0
t0 = time.time()

for i, batch in enumerate(loader, 1):
    try:
        bg = {k: to_gpu(k, v) for k, v in batch.items() if k in {'features', 'points', 'neighbors', 'pools', 'upsamples', 'stack_lengths'}}
        Ns = int(bg['stack_lengths'][0][0]); Nt = int(bg['stack_lengths'][0][1])
        with torch.no_grad():
            F = mgr.forward_backbone(bg)
            Fs = torch.nn.functional.normalize(F[:Ns], dim=1)
            Ft = torch.nn.functional.normalize(F[Ns:Ns + Nt], dim=1)
        X = bg['points'][0]; Xs = X[:Ns]; Xt = X[Ns:Ns + Nt]

        Kp = min(args.n_desc, Ns)
        Kq = min(args.n_desc, Nt)
        if args.sample_mode == 'predator_prob':
            with torch.no_grad():
                _, overlap, matchability = predator_model(bg)
                score = (overlap * matchability).clamp_min(1e-12)
            source_score = score[:Ns]
            target_score = score[Ns:Ns + Nt]
            ip = torch.multinomial(source_score / source_score.sum(), Kp, replacement=False)
            iq = torch.multinomial(target_score / target_score.sum(), Kq, replacement=False)
        else:
            ip = torch.randperm(Ns, device=device)[:Kp]
            iq = torch.randperm(Nt, device=device)[:Kq]
        sim = Fs[ip] @ Ft[iq].T
        n1 = sim.argmax(1); n2 = sim.argmax(0)
        mutual = n2[n1] == torch.arange(Kp, device=device)
        src_idx = ip[mutual]; tgt_idx = iq[n1[mutual]]
        M = int(src_idx.numel()); stats['matches'].append(M)
        total_eval += 1
        if M >= 3:
            src_np = Xs[src_idx].cpu().numpy(); tgt_np = Xt[tgt_idx].cpu().numpy()
            Rgt = batch['rot'].numpy(); tgt = batch['trans'].numpy().reshape(3)
            gt_err = np.linalg.norm(src_np @ Rgt.T + tgt - tgt_np, axis=1)
            ir_gt = float((gt_err < args.inlier_thresh).mean())
            stats['ir_gt'].append(ir_gt)
            fmr_success += int(ir_gt > args.fmr_thresh)
            Rp, tp, mask = ransac_svd(src_np, tgt_np, args.ransac_iters, args.inlier_thresh)
            stats['ir_pred'].append(float(mask.mean()) if len(mask) else 0.0)
            re = rotation_error(Rp, Rgt); te = float(np.linalg.norm(tp - tgt))
            pred_src = src_np @ Rp.T + tp
            gt_src = src_np @ Rgt.T + tgt
            rmse = float(np.sqrt(np.mean(np.sum((pred_src - gt_src) ** 2, axis=1))))
            stats['re'].append(re); stats['te'].append(te); stats['rmse'].append(rmse)
            reg_valid += 1
            rr_success += int(rmse < args.rr_rmse)
        else:
            stats['ir_gt'].append(0.0)
            stats['ir_pred'].append(0.0)
        if i % 50 == 0:
            log.info(f'[{i}/{len(ds)}] FMR={100*fmr_success/max(total_eval,1):.2f}% RR={100*rr_success/max(total_eval,1):.2f}% IR={100*np.mean(stats["ir_gt"]):.2f}% RE_med={np.median(stats["re"]):.2f} TE_med={np.median(stats["te"]):.3f} matches_med={np.median(stats["matches"]):.0f}')
    except Exception as e:
        bad += 1
        if bad <= 3:
            log.warning(f'pair {i} failed: {repr(e)}')

log.info('=== VAL RESULT ===')
log.info(f'pairs={len(ds)} eval_total={total_eval} reg_valid={reg_valid} bad={bad}')
log.info(f'FMR={100*fmr_success/max(total_eval,1):.4f}% fmr_success={fmr_success}/{total_eval}')
log.info(f'RR={100*rr_success/max(total_eval,1):.4f}% rr_success={rr_success}/{total_eval}')
log.info(f'RE_mean={np.mean(stats["re"]):.4f} RE_median={np.median(stats["re"]):.4f}')
log.info(f'TE_mean={np.mean(stats["te"]):.6f} TE_median={np.median(stats["te"]):.6f}')
log.info(f'RMSE_mean={np.mean(stats["rmse"]):.6f} RMSE_median={np.median(stats["rmse"]):.6f}')
log.info(f'IR_mean={100*np.mean(stats["ir_gt"]):.4f}% IR_median={100*np.median(stats["ir_gt"]):.4f}%')
log.info(f'IR_pred_mean={100*np.mean(stats["ir_pred"]):.4f}% IR_pred_median={100*np.median(stats["ir_pred"]):.4f}%')
log.info(f'matches_mean={np.mean(stats["matches"]):.4f} matches_median={np.median(stats["matches"]):.4f}')
log.info(f'elapsed_sec={time.time()-t0:.1f}')
