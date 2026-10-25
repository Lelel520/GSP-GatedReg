import os

import torch


class TrainConfig:
    def __init__(self):
        self._cfg = {
            "model": {"feat_dim": 256},
            "system": {"device": "cuda:" + os.environ.get("GPU_ID", "0")},
        }


class ModelManager:
    def __init__(self, config, logger):
        self.cfg = config
        self.logger = logger
        self.device = torch.device(
            config._cfg["system"]["device"] if torch.cuda.is_available() else "cpu"
        )
        self.backbone = None

    def build_model(self, stage):
        if stage != 1:
            raise ValueError("This release contains descriptor training (stage=1) only.")
        from .modules.kpconv_backbone import KPConvBackbone

        self.backbone = KPConvBackbone(
            out_dim=self.cfg._cfg["model"]["feat_dim"]
        ).to(self.device)
        self.logger.info(
            "KPConv descriptor model: %s parameters",
            format(sum(p.numel() for p in self.backbone.parameters()), ","),
        )

    def forward_backbone(self, batch_inputs):
        output = self.backbone(batch_inputs)
        return output[0] if isinstance(output, (tuple, list)) else output
