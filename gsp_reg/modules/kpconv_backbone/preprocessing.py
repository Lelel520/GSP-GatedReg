import torch
import numpy as np
import time

class Timer:
    def __init__(self):
        self.total_time = 0.0
        self._start = 0.0
    def tic(self):
        self._start = time.time()
    def toc(self):
        self.total_time += time.time() - self._start

def point2node_correspondences(src_node, src_pcd, tgt_node, tgt_pcd, matching_inds,
                                thres=0.3):
    import numpy as np
    src_dist = np.linalg.norm(
        src_pcd[:, None, :] - src_node[None, :, :], axis=-1)  # [N_sp, N_snode]
    src_pt2node = src_dist.argmin(axis=-1)  # [N_sp]
    tgt_dist = np.linalg.norm(
        tgt_pcd[:, None, :] - tgt_node[None, :, :], axis=-1)  # [N_tp, N_tnode]
    tgt_pt2node = tgt_dist.argmin(axis=-1)  # [N_tp]

    matching_mask = np.zeros(len(matching_inds), dtype=bool)
    node_corr = {}

    for mi, (si, ti) in enumerate(matching_inds):
        sn = src_pt2node[si]
        tn = tgt_pt2node[ti]
        src_in_node = (src_pt2node == sn).sum()
        tgt_in_node = (tgt_pt2node == tn).sum()
        if src_in_node > 0 and tgt_in_node > 0:
            overlap = 2.0 * min(src_in_node, tgt_in_node) / (src_in_node + tgt_in_node)
            if overlap > thres:
                matching_mask[mi] = True
                node_corr.setdefault(int(sn), []).append(int(tn))

    return matching_mask, node_corr

try:
    import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
    import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
except ImportError:
    try:
        from gsp_reg.cpp_wrappers.cpp_subsampling import grid_subsampling as cpp_subsampling
        from gsp_reg.cpp_wrappers.cpp_neighbors import radius_neighbors as cpp_neighbors
    except ImportError:
        import sys
        sys.path.insert(0, '.')
        cpp_subsampling = None
        cpp_neighbors = None


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """Grid subsampling with barycenter aggregation."""
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """Radius neighbors for stacked point clouds, truncated to max_neighbors."""

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_descriptor(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []
    assert len(list_data) == 1

    for ind, (
    src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd_raw, tgt_pcd_raw, sample) in enumerate(
            list_data):
        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))

    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    r_normal = config.first_subsampling_dl * config.conv_radius

    layer_blocks = []
    layer = 0

    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):

        if 'global' in block or 'upsample' in block:
            break

        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        if layer_blocks:
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        if 'pool' in block or 'strided' in block:

            dl = 2 * r_normal / config.conv_radius

            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsampling uses the next layer's neighborhood radius.
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        batched_points = pool_p
        batched_lengths = pool_b

        r_normal *= 2
        layer += 1
        layer_blocks = []

    len_src_pcd = input_batches_len[0][0]
    pcds = input_points[0]
    src_pcd, tgt_pcd = pcds[:len_src_pcd], pcds[len_src_pcd:]

    len_src_nodes = input_batches_len[config.node_id][0]
    nodes = input_points[config.node_id]
    src_node, tgt_node = nodes[:len_src_nodes], nodes[len_src_nodes:]
    if not config.mode == 'test':
        matching_mask, node_corr = point2node_correspondences(src_node, src_pcd, tgt_node, tgt_pcd, matching_inds, thres=config['patch_overlap_thres'])
    else:
        matching_mask, node_corr = None, None

    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'rot': torch.from_numpy(rot),
        'trans': torch.from_numpy(trans),
        'correspondences': matching_inds,
        'src_pcd_raw': torch.from_numpy(src_pcd_raw).float(),
        'tgt_pcd_raw': torch.from_numpy(tgt_pcd_raw).float(),
        'sample': sample,
        'matching_mask': matching_mask,
        'node_corr': node_corr
    }
    return dict_inputs


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = None

    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        hists_stack = np.vstack(hists)  # (n_layers, hist_n), n_layers varies
        if neighb_hists is None:
            neighb_hists = hists_stack
        else:
            # Pad to same number of layers across samples
            n_curr = hists_stack.shape[0]
            n_exist = neighb_hists.shape[0]
            if n_curr > n_exist:
                neighb_hists = np.pad(neighb_hists, ((0, n_curr - n_exist), (0, 0)))
            elif n_curr < n_exist:
                hists_stack = np.pad(hists_stack, ((0, n_exist - n_curr), (0, 0)))
            neighb_hists += hists_stack
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits
