import os, sys, time, logging, argparse
import numpy as np
import torch
from functools import partial
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('KP_ARCH', 'kitti_full')
os.environ.setdefault('KP_FIRST_SUBSAMPLING_DL', '0.3')
os.environ.setdefault('KP_CONV_RADIUS', '4.25')
os.environ.setdefault('KP_EXTENT', '2.0')

from gsp_reg.model_builder import TrainConfig, ModelManager
from gsp_reg.confidence import validate_predator
from gsp_reg.dataset.kitti import KITTIDataset
from gsp_reg.modules.kpconv_backbone.preprocessing import collate_fn_descriptor, calibrate_neighbors
from gsp_reg.modules.kpconv_backbone.architectures import architectures

parser = argparse.ArgumentParser()
parser.add_argument('ckpt')
parser.add_argument('--data_root', default='data/kitti')
parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
parser.add_argument('--pairs', type=int, default=0, help='0 = full split')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_desc', type=int, default=5000)
parser.add_argument('--sample_mode', default='random', choices=['random', 'predator_prob'])
parser.add_argument('--predator_ckpt', type=str, default='third_party/OverlapPredator/weights/kitti.pth')
parser.add_argument('--predator_root', type=str, default='third_party/OverlapPredator')
parser.add_argument('--ransac_iters', type=int, default=50000)
parser.add_argument('--inlier_thresh', type=float, default=0.60, help='KITTI matching/RANSAC inlier radius in meters')
parser.add_argument('--rr_re', type=float, default=5.0, help='KITTI RR rotation threshold in degrees')
parser.add_argument('--rr_te', type=float, default=2.0, help='KITTI RR translation threshold in meters')
parser.add_argument('--log', type=str, default='')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()
if args.sample_mode == 'predator_prob':
    try:
        validate_predator(args.predator_root, args.predator_ckpt, 'kitti')
    except (FileNotFoundError, ImportError) as error:
        parser.error(str(error))

os.makedirs('logs', exist_ok=True)
log_path = args.log or f'logs/val_eval_kitti_{os.path.basename(args.ckpt).replace(".pth", "")}_{args.split}_{args.sample_mode}_n{args.n_desc}_{time.strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger('val_eval_kitti')

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
log.info(f'device={device} gpu={torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "cpu"}')
log.info(f'checkpoint={args.ckpt}')
log.info(f'split={args.split} sample_mode={args.sample_mode} n_desc={args.n_desc}')
log.info(f'GeoTransformer-style KITTI metrics: RR=(RRE<{args.rr_re}deg and RTE<{args.rr_te}m), inlier_thresh={args.inlier_thresh}m, RANSAC iters={args.ransac_iters}')
if args.sample_mode == 'predator_prob':
    log.info(
        f'PREDATOR confidence keypoints: n_desc={args.n_desc} '
        f'selection=probability sampling score=overlap*matchability '
        f'checkpoint={args.predator_ckpt}'
    )
log.info(f'log={log_path}')

rng = np.random.default_rng(123)
torch.manual_seed(123)

cfg = edict({
    'root': args.data_root,
    'augment_noise': 0.0,
    'augment_shift_range': 2.0,
    'augment_scale_max': 1.2,
    'augment_scale_min': 0.8,
    'architecture': architectures['kitti_full'],
    'first_subsampling_dl': 0.3,
    'conv_radius': 4.25,
    'deform_radius': 5.0,
    'num_kernel_points': 15,
    'num_layers': 4,
    'in_points_dim': 3,
    'in_feats_dim': 1,
    'first_feats_dim': 128,
    'final_feats_dim': 32,
    'intermediate_feats_dim': 32,
    'gnn_feats_dim': 256,
    'node_id': -1,
    'max_points': 512,
    'overlap_radius': 0.45,
    'mode': args.split,
})

ds_full = KITTIDataset(cfg, args.split, data_augmentation=False)
if args.pairs > 0:
    ds_full.files = ds_full.files[:args.pairs]
log.info(f'kitti pairs={len(ds_full)}')

# Estimate neighborhood limits from a fixed calibration subset.
calib = KITTIDataset(cfg, args.split, data_augmentation=False)
if len(calib.files) > 8:
    calib.files = calib.files[:8]
limits = calibrate_neighbors(
    calib, cfg, collate_fn=collate_fn_descriptor,
    keep_ratio=0.8, samples_threshold=2000,
)
log.info(f'neighborhood_limits={limits}')

loader = torch.utils.data.DataLoader(
    ds_full, batch_size=1, shuffle=False, num_workers=args.num_workers,
    collate_fn=partial(collate_fn_descriptor, config=cfg, neighborhood_limits=limits),
    pin_memory=True,
)

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
        R, tr = rigid_fit_svd(s, t)
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
log.info(f'backbone_params={sum(p.numel() for p in mgr.backbone.parameters())}')

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
        os.path.join(predator_root, 'configs/test/kitti.yaml')
    ))
    predator_config.architecture = _predator_architectures['kitti']
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

stats = {key: [] for key in ['ir_gt', 'ir_pred', 're', 'te', 'matches']}
rr_success = total_eval = reg_valid = bad = 0
t_start = time.time()

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
            stats['ir_gt'].append(float((gt_err < args.inlier_thresh).mean()))
            Rp, tp, mask = ransac_svd(src_np, tgt_np, args.ransac_iters, args.inlier_thresh)
            stats['ir_pred'].append(float(mask.mean()) if len(mask) else 0.0)
            re = rotation_error(Rp, Rgt); te = float(np.linalg.norm(tp - tgt))
            stats['re'].append(re); stats['te'].append(te)
            reg_valid += 1
            rr_success += int(re < args.rr_re and te < args.rr_te)
        else:
            stats['ir_gt'].append(0.0); stats['ir_pred'].append(0.0)
        if i % 50 == 0:
            log.info(f'[{i}/{len(ds_full)}] RR={100*rr_success/max(total_eval,1):.2f}% IR={100*np.mean(stats["ir_gt"]):.2f}% RRE_mean={np.mean(stats["re"]):.3f} RTE_mean={np.mean(stats["te"]):.3f} matches_med={np.median(stats["matches"]):.0f}')
    except Exception as e:
        bad += 1
        if bad <= 5:
            log.warning(f'pair {i} failed: {repr(e)}')

log.info('=== KITTI VAL RESULT ===')
log.info(f'pairs={len(ds_full)} eval_total={total_eval} reg_valid={reg_valid} bad={bad}')
log.info(f'RR={100*rr_success/max(total_eval,1):.4f}% rr_success={rr_success}/{total_eval}')
log.info(f'RRE_mean={np.mean(stats["re"]):.6f} RRE_median={np.median(stats["re"]):.6f}')
log.info(f'RTE_mean={np.mean(stats["te"]):.6f} RTE_median={np.median(stats["te"]):.6f}')
log.info(f'IR_mean={100*np.mean(stats["ir_gt"]):.4f}% IR_median={100*np.median(stats["ir_gt"]):.4f}%')
log.info(f'IR_pred_mean={100*np.mean(stats["ir_pred"]):.4f}% IR_pred_median={100*np.median(stats["ir_pred"]):.4f}%')
log.info(f'matches_mean={np.mean(stats["matches"]):.4f} matches_median={np.median(stats["matches"]):.4f}')
log.info(f'elapsed_sec={time.time()-t_start:.1f}')
