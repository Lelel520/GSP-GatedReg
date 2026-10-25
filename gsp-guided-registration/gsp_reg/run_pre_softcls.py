import os, sys, time, datetime, glob
from concurrent.futures import ThreadPoolExecutor
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsp_reg.model_builder import TrainConfig, ModelManager
from gsp_reg.modules.gsp_projector import GSPProjector
from gsp_reg.modules.soft_classified_loss import SoftClassifiedLoss, compute_stopo_chunked
import logging

run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(f'logs/softcls_pre_{run_id}.log'), logging.StreamHandler()])
logger = logging.getLogger('softcls_pre')
logger.info(f'Run ID: {run_id}')
SEED = int(os.environ.get('SEED', '42'))
np.random.seed(SEED)
torch.manual_seed(SEED)

GPU_ID = int(os.environ.get('GPU_ID', '0'))
device = torch.device(f'cuda:{GPU_ID}')
logger.info(f'GPU: {torch.cuda.get_device_name(GPU_ID)} (cuda:{GPU_ID})')

PREPROC_DIR = os.environ.get('PREPROC_DIR', 'data/3DMatch/preprocessed')
PREFETCH = int(os.environ.get('PREFETCH', '4'))
LOG_INTERVAL = int(os.environ.get('LOG_INTERVAL', '50'))
GRAD_DIAG_INTERVAL = int(os.environ.get('GRAD_DIAG_INTERVAL', '50'))
COLLAPSE_DIAG_INTERVAL = int(os.environ.get('COLLAPSE_DIAG_INTERVAL', '50'))
MAX_PAIRS = int(os.environ.get('MAX_PAIRS', '0'))
EPOCH_PAIRS = int(os.environ.get('EPOCH_PAIRS', '1000'))
CORR_DIR = os.environ.get('CORR_DIR', 'data/3DMatch/preprocessed_corr')
STOPO_DIR = os.environ.get('STOPO_DIR', 'data/3DMatch/preprocessed_stopo_abs')
TAU_LOW = float(os.environ.get('TAU_LOW', '0.03'))
TAU_HIGH = float(os.environ.get('TAU_HIGH', '0.20'))
GATE_ALPHA = float(os.environ.get('GATE_ALPHA', '1.0'))
GATE_BETA = float(os.environ.get('GATE_BETA', '1.0'))
GATE_THETA = float(os.environ.get('GATE_THETA', '0.2'))
pair_files = sorted(glob.glob(os.path.join(PREPROC_DIR, 'pair_*.pth')))
if MAX_PAIRS > 0:
    pair_files = pair_files[:MAX_PAIRS]
n_pairs = len(pair_files)
epoch_pair_count = n_pairs if EPOCH_PAIRS <= 0 else min(EPOCH_PAIRS, n_pairs)
logger.info(f'Preprocessed dir: {PREPROC_DIR}')
logger.info(f'Preprocessed pairs: {n_pairs}')
logger.info(f'Pairs per epoch: {epoch_pair_count}')
logger.info(f'Prefetch workers: {PREFETCH}')
logger.info(f'Log interval: {LOG_INTERVAL}, grad diagnostics interval: {GRAD_DIAG_INTERVAL}, collapse diagnostics interval: {COLLAPSE_DIAG_INTERVAL}')
logger.info(f'Correspondence sidecar dir: {CORR_DIR}')
logger.info(f'Stopo sidecar dir: {STOPO_DIR}')
logger.info(f'Gate params: tau_low={TAU_LOW}, tau_high={TAU_HIGH}, alpha={GATE_ALPHA}, beta={GATE_BETA}, theta={GATE_THETA}')


def load_matching_backbone(model, path):
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model_state_dict', 'backbone_state_dict', 'backbone', 'model'):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f'Unsupported checkpoint format: {path}')

    current = model.state_dict()
    matched = {}
    skipped_shape = []
    skipped_key = []
    for key, value in state.items():
        model_key = key[9:] if key.startswith('backbone.') else key
        if model_key not in current:
            skipped_key.append(key)
            continue
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(current[model_key].shape):
            ckpt_shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            skipped_shape.append((key, ckpt_shape, tuple(current[model_key].shape)))
            continue
        matched[model_key] = value

    current.update(matched)
    model.load_state_dict(current, strict=True)
    return len(matched), skipped_shape, skipped_key

cfg = TrainConfig()
mgr = ModelManager(cfg, logger)
mgr.build_model(stage=1)
PRETRAIN_BACKBONE = os.environ.get('PRETRAIN_BACKBONE', '')
if PRETRAIN_BACKBONE:
    n_loaded, skipped_shape, skipped_key = load_matching_backbone(mgr.backbone, PRETRAIN_BACKBONE)
    logger.info(f'Loaded pretrain: {PRETRAIN_BACKBONE}')
    logger.info(f'  matched tensors: {n_loaded}')
    if skipped_shape:
        logger.info(f'  skipped shape mismatch: {skipped_shape[:6]}')
    if skipped_key:
        logger.info(f'  skipped missing keys: {skipped_key[:6]}')
mgr.backbone.train()
logger.info(f'Params: {sum(p.numel() for p in mgr.backbone.parameters()):,}')

LR = float(os.environ.get('LR', '0.0005'))
opt = torch.optim.SGD(mgr.backbone.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)

gsp_proj = GSPProjector(num_superpoints=128, sigma=0.2).to(device)
LAMBDA_DEG = float(os.environ.get('LAMBDA_DEG', '0.1'))
LAMBDA_DIST = float(os.environ.get('LAMBDA_DIST', '1.0'))
LAMBDA_VAR = float(os.environ.get('LAMBDA_VAR', '1.0'))
VAR_GAMMA = float(os.environ.get('VAR_GAMMA', '0.06'))
SAFE_RADIUS = float(os.environ.get('SAFE_RADIUS', '0.10'))
CIRCLE_POS_MARGIN = float(os.environ.get('CIRCLE_POS_MARGIN', '0.10'))
CIRCLE_NEG_MARGIN = float(os.environ.get('CIRCLE_NEG_MARGIN', '1.40'))
CIRCLE_LOG_SCALE = float(os.environ.get('CIRCLE_LOG_SCALE', '10.0'))
classified_loss = SoftClassifiedLoss(
    lambda_deg=LAMBDA_DEG, lambda_dist=LAMBDA_DIST,
    lambda_var=LAMBDA_VAR, var_gamma=VAR_GAMMA,
    m_cross=CIRCLE_POS_MARGIN, safe_radius=SAFE_RADIUS,
    alpha=GATE_ALPHA, beta=GATE_BETA, theta=GATE_THETA,
    circle_pos_margin=CIRCLE_POS_MARGIN,
    circle_neg_margin=CIRCLE_NEG_MARGIN,
    circle_log_scale=CIRCLE_LOG_SCALE,
).to(device)
logger.info(
    f'Loss weights: lambda_deg={LAMBDA_DEG}, lambda_dist={LAMBDA_DIST}, '
    f'lambda_var={LAMBDA_VAR}, var_gamma={VAR_GAMMA}'
)
logger.info(
    f'Circle loss: pos_margin={CIRCLE_POS_MARGIN}, neg_margin={CIRCLE_NEG_MARGIN}, '
    f'log_scale={CIRCLE_LOG_SCALE}, safe_radius={SAFE_RADIUS}'
)

MAX_ANCHORS = 128
SAVE_EVERY = int(os.environ.get('SAVE_EVERY', '0'))
SAVE_AFTER_EPOCH = int(os.environ.get('SAVE_AFTER_EPOCH', '1200'))
SAVE_EPOCH_INTERVAL = int(os.environ.get('SAVE_EPOCH_INTERVAL', '50'))
epochs = int(os.environ.get('EPOCHS', '5000'))
START_EPOCH = int(os.environ.get('START_EPOCH', '1'))
os.makedirs('checkpoints', exist_ok=True)

logger.info(f'Starting training: start_epoch={START_EPOCH}, end_epoch={epochs}, lr={LR}')
if not pair_files:
    raise FileNotFoundError('No pair_*.pth files found in %s; preprocess the training split first.' % PREPROC_DIR)

warned_stopo_mismatch = False


def get_stopo(batch_gpu, key, xyz):
    """Load or compute geometry gates aligned with points[0]."""
    global warned_stopo_mismatch
    cached = batch_gpu.get(key)
    expected = xyz.shape[1]
    if isinstance(cached, torch.Tensor) and cached.numel() == expected:
        return cached.reshape(1, expected).to(device=device, dtype=xyz.dtype)

    if not warned_stopo_mismatch:
        got = None if cached is None else int(cached.numel())
        logger.warning(
            f'{key} length mismatch: got {got}, expected {expected}. '
            'Old preprocessed stopo is ignored; recomputing on points[0]. '
            'Run gsp_reg/preprocess_3dmatch.py again to persist aligned stopo.'
        )
        warned_stopo_mismatch = True
    with torch.no_grad():
        return compute_stopo_chunked(xyz, tau_low=TAU_LOW, tau_high=TAU_HIGH)


INDEX_LIST_KEYS = {'neighbors', 'pools', 'upsamples'}



def load_batch(path):
    batch = torch.load(path, map_location='cpu')
    if 'correspondences' not in batch:
        corr_path = os.path.join(CORR_DIR, os.path.basename(path))
        if os.path.exists(corr_path):
            corr_payload = torch.load(corr_path, map_location='cpu')
            batch['correspondences'] = corr_payload.get('correspondences')
    if 'stopo_src' not in batch:
        stopo_path = os.path.join(STOPO_DIR, os.path.basename(path))
        if os.path.exists(stopo_path):
            stopo_payload = torch.load(stopo_path, map_location='cpu')
            if torch.is_tensor(stopo_payload.get('stopo_src')):
                batch['stopo_src'] = stopo_payload['stopo_src']
    return batch


def iter_loaded_batches(paths):
    if PREFETCH <= 0:
        for path in paths:
            yield load_batch(path)
        return
    with ThreadPoolExecutor(max_workers=PREFETCH) as executor:
        iterator = iter(paths)
        futures = []
        for _ in range(PREFETCH):
            try:
                futures.append(executor.submit(load_batch, next(iterator)))
            except StopIteration:
                break
        while futures:
            future = futures.pop(0)
            try:
                next_path = next(iterator)
                futures.append(executor.submit(load_batch, next_path))
            except StopIteration:
                pass
            yield future.result()


def to_gpu_value(key, value):
    if isinstance(value, torch.Tensor):
        out = value.to(device, non_blocking=True)
        if key in INDEX_LIST_KEYS and out.dtype != torch.long:
            out = out.long()
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, torch.Tensor):
                item = item.to(device, non_blocking=True)
                if key in INDEX_LIST_KEYS and item.dtype != torch.long:
                    item = item.long()
            out.append(item)
        return out
    return value


def move_needed_to_gpu(batch):
    needed = {'points', 'neighbors', 'pools', 'upsamples', 'features',
              'stack_lengths', 'stopo_src', 'gsp_C_src', 'correspondences'}
    return {k: to_gpu_value(k, v) for k, v in batch.items() if k in needed}


def collapse_diagnostics(F_p, F_q, delta_p, s_topo_p, gate_thr=0.5):
    with torch.no_grad():
        feat = torch.cat([F_p.reshape(-1, F_p.shape[-1]), F_q.reshape(-1, F_q.shape[-1])], dim=0)
        feat_centered = feat - feat.mean(dim=0, keepdim=True)
        feat_var = feat_centered.pow(2).sum(dim=1).mean()

        delta = delta_p.reshape(-1)
        s = s_topo_p.reshape(-1).clamp(0.0, 1.0)
        deg_mask = s <= gate_thr
        dist_mask = s > gate_thr
        delta_deg = delta[deg_mask].mean() if deg_mask.any() else delta.new_tensor(float('nan'))
        delta_dist = delta[dist_mask].mean() if dist_mask.any() else delta.new_tensor(float('nan'))

        return {
            'feat_var': feat_var.item(),
            'delta_deg': delta_deg.item(),
            'delta_dist': delta_dist.item(),
            'n_deg': int(deg_mask.sum().item()),
            'n_dist': int(dist_mask.sum().item()),
        }


best_loss = float('inf')
# Reuse the same pair subset across epochs.
fixed_indices = np.arange(epoch_pair_count)
for epoch in range(START_EPOCH, epochs + 1):
    epoch_start = time.time()
    total_loss = 0.0; total_deg = 0.0; total_dist = 0.0; total_var = 0.0
    total_dpos = 0.0; total_dneg = 0.0; total_gap = 0.0; total_acc = 0.0
    total_r_deg = 0.0; total_r_dist = 0.0; total_r_var = 0.0
    total_varf = 0.0; total_delta_neg = 0.0; total_delta_dist = 0.0
    num_batches = 0
    num_ratio_batches = 0
    num_diag_batches = 0
    
    epoch_files = [pair_files[idx] for idx in fixed_indices]
    for batch_count, batch in enumerate(iter_loaded_batches(epoch_files)):
        batch_gpu = move_needed_to_gpu(batch)
        
        s_len = batch_gpu['stack_lengths'][0]
        X_all = batch_gpu['points'][0]
        src_N = int(s_len[0].item())
        tgt_N = int(s_len[1].item())
        X_p = X_all[:src_N].unsqueeze(0)
        X_q = X_all[src_N:src_N + tgt_N].unsqueeze(0)
        F_all = mgr.forward_backbone(batch_gpu)
        if torch.isnan(F_all).any():
            torch.cuda.empty_cache(); continue
        F_p = F_all[:src_N].unsqueeze(0)
        F_q = F_all[src_N:src_N + tgt_N].unsqueeze(0)
        
        s_p = get_stopo(batch_gpu, 'stopo_src', X_p)
        
        correspondences = batch_gpu.get('correspondences')
        if isinstance(correspondences, torch.Tensor) and correspondences.numel() > 0:
            correspondences = correspondences.long()
            valid_corr = (
                (correspondences[:, 0] >= 0) & (correspondences[:, 0] < X_p.shape[1]) &
                (correspondences[:, 1] >= 0) & (correspondences[:, 1] < X_q.shape[1])
            )
            correspondences = correspondences[valid_corr]
        if isinstance(correspondences, torch.Tensor) and correspondences.shape[0] > 0:
            K = min(MAX_ANCHORS, correspondences.shape[0])
            sel = torch.randperm(correspondences.shape[0], device=device)[:K]
            anchor_idx = correspondences[sel, 0].long()
            pos_idx = correspondences[sel, 1].long()
        else:
            logger.warning('No valid correspondences in batch; falling back to random positives')
            Kp = min(MAX_ANCHORS, X_p.shape[1])
            Kq = min(MAX_ANCHORS, X_q.shape[1])
            K = min(Kp, Kq)
            anchor_idx = torch.randperm(X_p.shape[1], device=device)[:K]
            pos_idx = torch.randperm(X_q.shape[1], device=device)[:K]
        
        # Reuse cached coordinate assignments when available.
        C_src = batch_gpu.get('gsp_C_src')
        if isinstance(C_src, torch.Tensor) and C_src.shape[-1] == X_p.shape[1]:
            if C_src.dim() == 2:
                C_src = C_src.unsqueeze(0)
            delta_p = gsp_proj.delta_from_assignment(F_p, C_src)
        else:
            delta_p, _ = gsp_proj(X_p, F_p)
        compute_grad_ratio = batch_count % GRAD_DIAG_INTERVAL == 0
        compute_collapse_diag = batch_count % COLLAPSE_DIAG_INTERVAL == 0
        diag_dict = collapse_diagnostics(F_p, F_q, delta_p, s_p, gate_thr=GATE_THETA) if compute_collapse_diag else {}
        loss, loss_dict = classified_loss(
            F_p, F_q, X_p, X_q, anchor_idx, pos_idx, delta_p, s_p,
            grad_targets=(F_p, F_q), compute_grad_ratio=compute_grad_ratio,
        )
        if torch.isnan(loss): continue
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mgr.backbone.parameters(), 10.0)
        opt.step()
        
        total_loss += loss.item()
        total_deg += loss_dict["deg"]
        total_dist += loss_dict["dist"]
        total_var += loss_dict.get("var", 0.0)
        total_dpos += loss_dict.get("dpos_w", loss_dict.get("dpos", 0.0))
        total_dneg += loss_dict.get("dneg_w", loss_dict.get("dneg", 0.0))
        total_gap += loss_dict.get("gap_w", loss_dict.get("gap", 0.0))
        total_acc += loss_dict.get("circle_acc_w", loss_dict.get("circle_acc", 0.0))
        if 'r_deg' in loss_dict:
            total_r_deg += loss_dict['r_deg']
            total_r_dist += loss_dict['r_dist']
            total_r_var += loss_dict.get('r_var', 0.0)
            num_ratio_batches += 1
        if diag_dict:
            total_varf += diag_dict['feat_var']
            total_delta_neg += diag_dict['delta_deg']
            total_delta_dist += diag_dict['delta_dist']
            num_diag_batches += 1
        num_batches += 1
        
        if SAVE_EVERY > 0 and epoch >= SAVE_AFTER_EPOCH and batch_count % SAVE_EVERY == 0 and batch_count > 0:
            torch.save(mgr.backbone.state_dict(), f'checkpoints/softcls_pre_{run_id}_e{epoch}_b{batch_count}.pth')
        
        if batch_count % LOG_INTERVAL == 0:
            elapsed = time.time() - epoch_start
            done = batch_count + 1
            eta = elapsed / done * (epoch_pair_count - done) if done > 0 else 0
            grad_msg = ''
            if 'r_deg' in loss_dict:
                grad_msg = (
                    f' | g_deg={loss_dict["g_deg"]:.2e} g_dist={loss_dict["g_dist"]:.2e} '
                    f'g_var={loss_dict.get("g_var", 0.0):.2e} '
                    f'r_deg={loss_dict["r_deg"]:.3f} r_dist={loss_dict["r_dist"]:.3f} '
                    f'r_var={loss_dict.get("r_var", 0.0):.3f} '
                    f's_mean={loss_dict.get("s_mean", 0.0):.3f} '
                    f'w_deg={loss_dict.get("w_deg_sum", 0.0):.1f} w_dist={loss_dict.get("w_dist_sum", 0.0):.1f} '
                    f'dpos={loss_dict.get("dpos_w", 0.0):.3f} dneg={loss_dict.get("dneg_w", 0.0):.3f} '
                    f'gap={loss_dict.get("gap_w", 0.0):.3f} act={loss_dict.get("active_w", 0.0):.2f} '
                    f'acc={loss_dict.get("circle_acc_w", 0.0):.2f} '
                    f'sig={loss_dict.get("sigma_mean", 0.0):.4f}/{loss_dict.get("sigma_min", 0.0):.4f}'
                )
            diag_msg = ''
            if diag_dict:
                diag_msg = (
                    f' | VarF={diag_dict["feat_var"]:.4f} '
                    f'Delta_deg={diag_dict["delta_deg"]:.4f} Delta_dist={diag_dict["delta_dist"]:.4f} '
                    f'Ndeg={diag_dict["n_deg"]} Ndist={diag_dict["n_dist"]}'
                )
            logger.info(
                f'[e{epoch}][{done:5d}/{epoch_pair_count}] '
                f'total={loss.item():.4f} L_deg={loss_dict["deg"]:.3f} '
                f'L_dist={loss_dict["dist"]:.3f} L_var={loss_dict.get("var", 0.0):.4f}'
                f'{grad_msg}{diag_msg} | '
                f'{elapsed/60:.1f}m ETA={eta/60:.1f}m | {done/elapsed:.1f}p/s')
    
    elapsed = time.time() - epoch_start
    scheduler.step()
    avg_loss = total_loss / max(num_batches, 1)
    logger.info(
        f'=== Epoch {epoch} ({elapsed/60:.1f}m) Total={avg_loss:.4f} '
        f'L_deg={total_deg/max(num_batches,1):.3f} '
        f'L_dist={total_dist/max(num_batches,1):.3f} '
        f'L_var={total_var/max(num_batches,1):.4f} '
        f'dpos={total_dpos/max(num_batches,1):.3f} '
        f'dneg={total_dneg/max(num_batches,1):.3f} '
        f'gap={total_gap/max(num_batches,1):.3f} '
        f'acc={total_acc/max(num_batches,1):.2f} '
        f'r_deg={total_r_deg/max(num_ratio_batches,1):.3f} '
        f'r_dist={total_r_dist/max(num_ratio_batches,1):.3f} '
        f'r_var={total_r_var/max(num_ratio_batches,1):.3f} '
        f'VAR_F={total_varf/max(num_diag_batches,1):.4f} '
        f'Delta_neg={total_delta_neg/max(num_diag_batches,1):.4f} '
        f'delta_dist={total_delta_dist/max(num_diag_batches,1):.4f} ==='
    )
    
    if epoch >= SAVE_AFTER_EPOCH:
        torch.save(mgr.backbone.state_dict(), f'checkpoints/softcls_pre_{run_id}_epoch{epoch}.pth')
    if avg_loss < best_loss:
        best_loss = avg_loss
        if epoch >= SAVE_AFTER_EPOCH:
            torch.save(mgr.backbone.state_dict(), f'checkpoints/softcls_pre_{run_id}_best.pth')
        logger.info(f'  New best: {best_loss:.4f}')

logger.info('Done')
