from __future__ import annotations
from typing import Any
from monai.networks.nets import UNet
from torch import nn

def build_unet(config: dict[str, Any]) -> nn.Module:
    model_cfg = config['model']
    return UNet(spatial_dims=3, in_channels=int(model_cfg['in_channels']), out_channels=int(model_cfg['out_channels']), channels=tuple((int(c) for c in model_cfg['channels'])), strides=tuple((int(s) for s in model_cfg['strides'])), num_res_units=int(model_cfg.get('num_res_units', 2)))