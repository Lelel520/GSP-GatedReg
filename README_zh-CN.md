# GSP-GatedReg

[English](README.md) | [简体中文](README_zh-CN.md)

论文 **GSP-Guided Soft-Gated Descriptor Learning for Robust Point Cloud Registration in Locally Ambiguous Geometry** 的代码。

作者：Kangle Xu、Aozhe Wang、Chong Wang、Hongjian Yu、Zhijiang Du、Weidong Wang。
期刊：*IEEE Robotics and Automation Letters*，2026。

[IEEE Xplore 论文页面](https://ieeexplore.ieee.org/document/11743888) |
[DOI: 10.1109/LRA.2026.3746155](https://doi.org/10.1109/LRA.2026.3746155)

GSP-GatedReg 使用区域 GSP 重构、几何门控度量监督和方差正则化，训练输出 256 维描述子的 KPConv 网络。
推理时，可使用官方预训练 PREDATOR 网络预测的重叠概率与可匹配性得分引导关键点采样；选取的描述子通过互为最近邻匹配建立对应关系，再由 RANSAC 求解位姿。
仓库也支持不依赖 PREDATOR 的随机关键点采样。

## 安装

已测试的参考环境：Linux、Python 3.8、PyTorch 2.2.2+cu121、NumPy 1.24.4、Open3D 0.19.0。
编译扩展需要 C++ 编译器和 Python 开发头文件。

```bash
conda create -n gsp-reg python=3.8 pip numpy=1.24.4 setuptools -y
conda activate gsp-reg
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/build_extensions.sh
```

所有命令均在仓库根目录运行。Shell 脚本支持通过 `PYTHON=/path/to/python` 指定解释器。

## 数据与预训练权重

| 数据集 | 下载入口 | 所需内容 |
| --- | --- | --- |
| 3DMatch / 3DLoMatch | [预处理点云数据包](https://share.phys.ethz.ch/~gseg/Predator/data.zip)，由 [PREDATOR 官方仓库](https://github.com/prs-eth/OverlapPredator#datasets-and-pretrained-models) 提供 | 室内 `.pth` 点云片段；本仓库已包含两套评估配对列表 |
| KITTI Odometry | [KITTI 官方下载页面](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | 序列 00-10 的 Velodyne 激光点云和真值位姿，下载可能需要登录 |

目录结构见 [数据准备说明](docs/DATASETS.md)。3DLoMatch 与 3DMatch 共用点云片段，仅配对列表不同，不需要重复下载点云。
源码压缩包不包含数据集和生成的缓存。

**我们训练的描述子权重与源码分开发放，源码 ZIP 内不包含权重。**

| 权重文件 | 适用数据集 | 大小 | 下载 |
| --- | --- | --- | --- |
| `gsp_gatedreg_3dmatch.pth` | 3DMatch / 3DLoMatch | 16.5 MiB | [下载](https://github.com/Lelel520/GSP-GatedReg/releases/download/v1.0.0/gsp_gatedreg_3dmatch.pth) |
| `gsp_gatedreg_kitti.pth` | KITTI | 16.5 MiB | [下载](https://github.com/Lelel520/GSP-GatedReg/releases/download/v1.0.0/gsp_gatedreg_kitti.pth) |

两个权重均已在 [v1.0.0 Release](https://github.com/Lelel520/GSP-GatedReg/releases/tag/v1.0.0) 发布，可直接点击上表链接下载。

将两个描述子权重文件放入 `checkpoints/`，或从下载目录校验并安装：

```bash
python scripts/install_weights.py --source-dir /path/to/downloaded/assets
```

权重文件名和校验信息见 [checkpoints/README.md](checkpoints/README.md)。
使用置信度引导采样前，需按 [置信度模块说明](docs/CONFIDENCE.md) 安装 PREDATOR 并获取其官方权重。

## 评估

三个基准共用一个评估入口。默认使用 5000 个**随机关键点**、本方法的描述子网络，以及最多 50000 次 RANSAC 迭代。
以下命令不需要 PREDATOR：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh 3DMatch
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh 3DLoMatch
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh KITTI
```

若要使用论文中的置信度引导流程，请先安装 PREDATOR 及其官方权重，再显式启用：

```bash
CUDA_VISIBLE_DEVICES=0 SAMPLE_MODE=predator_prob bash scripts/evaluate.sh 3DMatch
CUDA_VISIBLE_DEVICES=0 SAMPLE_MODE=predator_prob bash scripts/evaluate.sh 3DLoMatch
CUDA_VISIBLE_DEVICES=0 SAMPLE_MODE=predator_prob bash scripts/evaluate.sh KITTI
```

随机采样不等同于论文完整置信度引导流程的结果。
置信度模式会在加载数据对之前检查外部源码、所需 Python 依赖和权重文件；缺少依赖时会明确报错，不会静默切换为随机采样。

通过 `N_DESCS`、`RANSAC_ITERS` 或 `CKPT` 修改关键点数、RANSAC 迭代上限或描述子权重。
其余命令行参数会传给对应评估程序：

```bash
N_DESCS=2500 RANSAC_ITERS=1000 bash scripts/evaluate.sh 3DLoMatch --pairs 20
bash scripts/evaluate.sh KITTI --rr_re 1.0 --rr_te 0.5
```

室内评估的内点距离阈值为 0.10 m，FMR 要求 IR > 5%，RR 要求配准 RMSE < 0.20 m。
KITTI 默认使用 0.60 m 的内点距离阈值，配准成功需同时满足 RRE < 5° 和 RTE < 2 m；日志中的平移误差单位为米。
RANSAC 参数是迭代上限，一致集比例超过 90% 时会提前结束。
评估日志中的总耗时包含数据准备，不能直接作为纯推理速度。

室内评估可使用缓存，避免重复构建 KPConv 输入：

```bash
python gsp_reg/preprocess_tdmatch_eval.py --benchmark 3DMatch \
  --output_dir data/3DMatch/preprocessed_3dmatch
CACHE_DIR=data/3DMatch/preprocessed_3dmatch bash scripts/evaluate.sh 3DMatch
```

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python gsp_reg/preprocess_3dmatch.py
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh 3DMatch

CUDA_VISIBLE_DEVICES=0 python gsp_reg/preprocess_kitti_softcls.py \
  --root data/kitti --split train --output-dir data/kitti/preprocessed
CUDA_VISIBLE_DEVICES=0 EPOCHS=250 bash scripts/train.sh KITTI
```

默认从头训练，不加载预训练权重。仅在需要用兼容权重进行微调时设置 `PRETRAIN_BACKBONE`；该选项只加载模型参数，不恢复优化器或学习率调度器状态。
`EPOCH_PAIRS=0` 表示使用全部已预处理的训练对。验证需单独运行，不要使用最终测试集选择模型。

提供的历史权重均经过预训练权重初始化。室内权重最后一段训练记录使用 `lambda_deg=0.05`，KITTI 使用 `0.10`。
两者均使用固定门控阈值 `(0.03, 0.20)`、`lambda_var=200` 和 `var_gamma=0.06`。
训练脚本的默认损失配置与这些记录一致，但从头训练不保证重现完全相同的历史优化轨迹。

## 代码结构

```text
gsp_reg/model_builder.py        模型配置与构建
gsp_reg/modules/                KPConv、GSP 重构和训练损失
gsp_reg/dataset/                数据集加载
gsp_reg/preprocess_*.py         训练数据预处理与室内评估缓存
gsp_reg/run_pre_softcls*.py     描述子训练
gsp_reg/val_eval*_checkpoint.py 室内与 KITTI 评估
scripts/                       扩展编译、训练、评估和权重安装
tests/                         核心功能数值测试
```

运行 `python -m unittest discover -s tests -v` 执行测试。
仅加载来源可信的 PyTorch 权重和 pickle 文件。

## 论文引用与许可证

若本代码对你的研究有帮助，请引用：

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

作者原创代码采用 [MIT 许可证](LICENSE)。第三方代码、数据集和预训练模型遵循各自的许可条款。
第三方代码来源及许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
