# Third-Party Notices

Upstream repositories checked on October 25, 2026. The root MIT license covers
the authors' original code; third-party material retains its own terms.

## Verified Licenses

| Project | Material used here | License |
| --- | --- | --- |
| [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) | Kernel construction, convolution blocks, and neighborhood/subsampling extensions | [MIT](licenses/KPConv-MIT.txt) |
| [PREDATOR](https://github.com/prs-eth/OverlapPredator) | External confidence network; shared KPConv and KITTI-loading implementations; indoor benchmark metadata | [MIT](licenses/PREDATOR-MIT.txt) |
| [nanoflann](https://github.com/jlblancoc/nanoflann) | Bundled KD-tree header | BSD 2-Clause; full notice retained in `gsp_reg/cpp_wrappers/cpp_utils/nanoflann/nanoflann.hpp` |

The KPConv license was checked at commit
[`d19c575d3fa9fcfd5a74845b5b27aac7e50472c7`](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/d19c575d3fa9fcfd5a74845b5b27aac7e50472c7/LICENSE.txt).
The PREDATOR license and shared files were checked at commit
[`8c78f125fc58d62ad7d149adf1fed43ed54937e4`](https://github.com/prs-eth/OverlapPredator/blob/8c78f125fc58d62ad7d149adf1fed43ed54937e4/LICENSE).
Preserve the supplied license texts and upstream copyright notices.

The four files in `gsp_reg/benchmark_configs/tdmatch/` are byte-identical to
PREDATOR's `configs/indoor/{3DMatch,3DLoMatch,train_info,val_info}.pkl` at that
commit. These files contain benchmark metadata, not dataset point clouds.

PREDATOR source and pretrained weights are obtained separately; see
[docs/CONFIDENCE.md](docs/CONFIDENCE.md). The local confidence adapter uses
overlap and matchability scores from the official network, while this project's
KPConv network supplies the descriptors used for matching. The source license
does not by itself establish separate data or pretrained-model permissions.

Dataset point clouds and official PREDATOR weights are not bundled in this
source archive. Their respective distribution terms still apply.
