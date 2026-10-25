# PREDATOR confidence model

Confidence-guided selection uses the official pretrained PREDATOR network.
For each point, overlap and matchability scores are multiplied, normalized
within its cloud, and used for sampling without replacement. GSP-GatedReg's
separate network supplies the descriptors used in matching.

Install the reference source:

```bash
git clone https://github.com/prs-eth/OverlapPredator.git third_party/OverlapPredator
git -C third_party/OverlapPredator checkout --detach 8c78f125fc58d62ad7d149adf1fed43ed54937e4
```

Obtain official `indoor.pth` and `kitti.pth` from the upstream
[weights archive](https://share.phys.ethz.ch/~gseg/Predator/weights.zip), linked by
PREDATOR's [download script](https://github.com/prs-eth/OverlapPredator/blob/8c78f125fc58d62ad7d149adf1fed43ed54937e4/scripts/download_data_weight.sh).
Place them in `third_party/OverlapPredator/weights/`. Install upstream dependencies
in the same environment. The provided evaluation batches use this repository's
preprocessing extensions.

Evaluation defaults to random sampling and does not require PREDATOR. Enable
confidence sampling explicitly with `SAMPLE_MODE=predator_prob` in the shell
entry point, or `--sample_mode predator_prob` in a Python evaluator.
`scripts/evaluate.sh` selects the indoor model for 3DMatch/3DLoMatch and the KITTI
model for KITTI. Override `PREDATOR_CKPT` to choose another compatible checkpoint.
The evaluator checks the source files, Python dependencies, and nonempty weight
file before loading data. It fails with installation guidance rather than
silently changing the sampling method. Checkpoint compatibility is subsequently
verified by strict model loading.

Official weights are not included among the authors' descriptor assets.
PREDATOR source copyright and license remain unchanged; see
`licenses/PREDATOR-MIT.txt`. Its inference cost includes a separate network,
not just a small head attached to GSP descriptors.
