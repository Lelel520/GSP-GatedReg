# Datasets

## 3DMatch and 3DLoMatch

Download the [preprocessed point-cloud archive](https://share.phys.ethz.ch/~gseg/Predator/data.zip)
provided by [PREDATOR's official download script](https://github.com/prs-eth/OverlapPredator/blob/8c78f125fc58d62ad7d149adf1fed43ed54937e4/scripts/download_data_weight.sh).
This loader expects `.pth` point-cloud tensors, not raw RGB-D images or arbitrary
PLY files. The [3DMatch project page](https://3dmatch.cs.princeton.edu/) describes
the original benchmark.

After extraction, place the indoor directory containing `train/` and `test/`
under `data/3DMatch/`. The benchmark pair metadata is already included in this
repository. 3DLoMatch uses the same point-cloud files and a separate pair list;
it does not require another point-cloud archive.

```text
data/3DMatch/
  train/<scene>/<fragment>.pth
  test/<scene>/<fragment>.pth
```

The exact relative names are the `src` and `tgt` fields of the bundled
`gsp_reg/benchmark_configs/tdmatch/*.pkl` metadata. Do not rename fragments
without updating metadata. `3DMatch.pkl` has 1623 evaluation pairs;
`3DLoMatch.pkl` has 1781 evaluation pairs. These uppercase benchmark files are
not interchangeable with historical lowercase internal validation subsets.
3DLoMatch is an official low-overlap test split, not a training split.

Training uses `train_info.pkl`; validation uses `val_info.pkl`. The historical
preprocessor selects 5000 training pairs with `numpy.random.seed(42)` followed
by a permutation. It does **not** take the first 5000 original metadata entries.
The release preserves that procedure and records `selected_pair_indices.pkl`
alongside the generated caches for auditability.

Preprocessing is deterministic only to the extent supported by the original
Open3D/NumPy/PyTorch operations and point-limit sampling. Keep the generated
cache if exact comparisons between runs are required. Evaluation caches are
bound to metadata order and preprocessing settings; never reuse a cache for
another benchmark by renaming its directory.

## KITTI odometry

From the [official KITTI Odometry download page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php),
download **velodyne laser data** and **ground truth poses**. Access may require
registration/login and acceptance of the dataset terms. Use sequences 00-10,
which have public ground truth; this registration protocol does not use the
official unlabeled odometry test sequences 11-21. Camera images are not needed.
The date/drive directory layout of KITTI Raw is not the expected input format.

Arrange the downloaded files as follows:

```text
data/kitti/dataset/
  sequences/00/velodyne/000000.bin
  ...
  sequences/10/velodyne/...
  poses/00.txt
  ...
  poses/10.txt
data/kitti/icp/  # created on first use
```

Training sequences: 00-05. Validation: 06-07. Test: 08-10.
`configs/kitti/*.txt` contains the sequence lists. The loader follows the
distance-based pair generation protocol and excludes its known invalid test pair.
It refines evaluation ground-truth transforms with ICP and saves them in
`data/kitti/icp/`. Messages about missing ICP files on the first run are expected;
this ground-truth refinement is not the learned registration estimator.

Use the `--data_root` evaluator option to select another dataset location, or
use a symbolic link at the default relative path. Do not commit data, caches,
or local symbolic links into Git.
