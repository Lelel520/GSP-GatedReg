# GSP-GatedReg

[English](README.md) | [简体中文](README_zh-CN.md)

Code for **GSP-Guided Soft-Gated Descriptor Learning for Robust Point Cloud
Registration in Locally Ambiguous Geometry**.

Kangle Xu, Aozhe Wang, Chong Wang, Hongjian Yu, Zhijiang Du, and Weidong Wang.
*IEEE Robotics and Automation Letters*, 2026.

[IEEE Xplore](https://ieeexplore.ieee.org/document/11743888) |
[DOI: 10.1109/LRA.2026.3746155](https://doi.org/10.1109/LRA.2026.3746155)

GSP-GatedReg trains a 256-dimensional KPConv descriptor using regional GSP
reconstruction, geometry-gated metric supervision, and variance regularization.
At inference, official pretrained PREDATOR overlap and matchability scores guide
keypoint sampling. Descriptors are matched by mutual nearest neighbors, followed
by RANSAC pose estimation. Random sampling is also available without PREDATOR.

## Install

Tested reference environment: Linux, Python 3.8, PyTorch 2.2.2+cu121,
NumPy 1.24.4, Open3D 0.19.0. A C++ compiler and Python headers are required.

```bash
conda create -n gsp-reg python=3.8 pip numpy=1.24.4 setuptools -y
conda activate gsp-reg
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/build_extensions.sh
```

Run all commands from the repository root. Use `PYTHON=/path/to/python` to choose
an interpreter in the shell entry points.

## Data and pretrained models

| Dataset | Download | Required data |
| --- | --- | --- |
| 3DMatch / 3DLoMatch | [Preprocessed point-cloud tensors](https://share.phys.ethz.ch/~gseg/Predator/data.zip), provided by [PREDATOR](https://github.com/prs-eth/OverlapPredator#datasets-and-pretrained-models) | Shared indoor `.pth` fragments; both benchmark splits are bundled here |
| KITTI Odometry | [Official download page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | Velodyne laser data and ground-truth poses for sequences 00-10; login may be required |

See [docs/DATASETS.md](docs/DATASETS.md) for the expected directory structure.
3DLoMatch uses the same point clouds as 3DMatch with a different pair list; no
separate point-cloud download is needed. Datasets and generated caches are not
included in the source archive.

**Descriptor weights are distributed separately from the source archive.**
The two author-trained checkpoints are:

| Checkpoint | Dataset | Size |
| --- | --- | --- |
| `gsp_gatedreg_3dmatch.pth` | 3DMatch / 3DLoMatch | 16.5 MiB |
| `gsp_gatedreg_kitti.pth` | KITTI | 16.5 MiB |

Public download links for these two checkpoints have not been published yet.
They must be obtained separately before running evaluation.

Place the two descriptor release assets in `checkpoints/`, or verify and install
them from their download directory:

```bash
python scripts/install_weights.py --source-dir /path/to/downloaded/assets
```

See [checkpoints/README.md](checkpoints/README.md) for model names and checksums.
For confidence-guided sampling, install PREDATOR and obtain its official weights
as described in [docs/CONFIDENCE.md](docs/CONFIDENCE.md).

## Evaluate

One evaluation entry point supports all three benchmarks. By default it uses
5000 **random keypoints**, the GSP descriptor alone, and at most 50000 RANSAC
iterations. PREDATOR is not required for these commands:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh 3DMatch
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh 3DLoMatch
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh KITTI
```

To use the paper's confidence-guided pipeline, first install PREDATOR and its
official weights, then explicitly enable confidence sampling:

```bash
CUDA_VISIBLE_DEVICES=0 SAMPLE_MODE=predator_prob bash scripts/evaluate.sh 3DMatch
CUDA_VISIBLE_DEVICES=0 SAMPLE_MODE=predator_prob bash scripts/evaluate.sh 3DLoMatch
CUDA_VISIBLE_DEVICES=0 SAMPLE_MODE=predator_prob bash scripts/evaluate.sh KITTI
```

Random sampling is not equivalent to the paper's full confidence-guided results.
Confidence mode checks the external source, required Python dependencies, and
weight file before loading dataset pairs. Missing requirements cause a clear
error; the evaluator never silently switches to random sampling.

Set `N_DESCS`, `RANSAC_ITERS`, or `CKPT` to override the defaults. Additional
arguments are forwarded to the evaluator:

```bash
N_DESCS=2500 RANSAC_ITERS=1000 bash scripts/evaluate.sh 3DLoMatch --pairs 20
bash scripts/evaluate.sh KITTI --rr_re 1.0 --rr_te 0.5
```

Indoor IR uses a 0.10 m inlier threshold; FMR requires IR > 5%; RR requires
registration RMSE < 0.20 m. KITTI defaults to a 0.60 m inlier threshold and
joint RRE < 5 degrees / RTE < 2 m success. Translation errors are logged in
meters. The RANSAC budget is a maximum, with early exit at >90% consensus.
Evaluation elapsed time includes data preparation and is not an inference-only
speed measurement.

Optional indoor caches avoid repeated KPConv collation:

```bash
python gsp_reg/preprocess_tdmatch_eval.py --benchmark 3DMatch \
  --output_dir data/3DMatch/preprocessed_3dmatch
CACHE_DIR=data/3DMatch/preprocessed_3dmatch bash scripts/evaluate.sh 3DMatch
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python gsp_reg/preprocess_3dmatch.py
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh 3DMatch

CUDA_VISIBLE_DEVICES=0 python gsp_reg/preprocess_kitti_softcls.py \
  --root data/kitti --split train --output-dir data/kitti/preprocessed
CUDA_VISIBLE_DEVICES=0 EPOCHS=250 bash scripts/train.sh KITTI
```

Fresh training starts without pretrained weights. Set `PRETRAIN_BACKBONE` only
when intentionally fine-tuning a compatible checkpoint. This loads model weights,
not optimizer/scheduler state. `EPOCH_PAIRS=0` uses all prepared pairs. Validation
is run separately; never use final test pairs to select checkpoints.

The supplied historical checkpoints were warm-started. The indoor checkpoint's
recorded final run used `lambda_deg=0.05`; KITTI used `0.10`. Both used fixed gate
thresholds `(0.03,0.20)`, `lambda_var=200`, and `var_gamma=0.06`. The wrapper
loss defaults follow these recorded runs. Fresh training is not guaranteed to follow
the exact historical optimization trajectory.

## Code structure

```text
gsp_reg/model_builder.py        Model configuration and construction
gsp_reg/modules/                KPConv, GSP reconstruction, and training loss
gsp_reg/dataset/                Dataset loaders
gsp_reg/preprocess_*.py         Training data preparation and indoor eval cache
gsp_reg/run_pre_softcls*.py     Descriptor training
gsp_reg/val_eval*_checkpoint.py Indoor and KITTI evaluation
scripts/                       Build, train, evaluate, and weight installation
tests/                         Small numerical tests for the core objective
```

Run `python -m unittest discover -s tests -v` to check the core objective.
Only load trusted PyTorch and pickle assets.

## Citation and license

If you use this code in your research, please cite:

```bibtex
@article{xu2026gspgatedreg,
  author  = {Xu, Kangle and Wang, Aozhe and Wang, Chong and Yu, Hongjian and Du, Zhijiang and Wang, Weidong},
  title   = {{GSP-Guided Soft-Gated Descriptor Learning for Robust Point Cloud Registration in Locally Ambiguous Geometry}},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  doi     = {10.1109/LRA.2026.3746155},
  url     = {https://ieeexplore.ieee.org/document/11743888}
}
```

The authors' original code uses the [MIT License](LICENSE). Third-party code,
data, and pretrained models retain their own terms. Sources and license details
are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
