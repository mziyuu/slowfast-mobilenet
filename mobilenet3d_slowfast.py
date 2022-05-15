# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from thop import profile

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..builder import BACKBONES
from mobilenet3d import MobileNet3d,Block,Convbn

try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

class LateralConv(nn.Module):
    def __init__(self,inchannel, outchannel, stride, kernel):
        super().__init__()
        self.lateral_conv = nn.Sequential(
            nn.Conv3d(inchannel, outchannel,kernel_size = kernel,stride = stride,padding=(2, 0, 0)),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.lateral_conv(x)
        return out

class MobileNet3dPathway(MobileNet3d):
    """A pathway of Slowfast based on MobileNet3d.
    """
    def __init__(self,
                 *args,
                 lateral=False,
                 #lateral_norm=False,
                 speed_ratio=None,
                 channel_ratio=None,
                 fusion_kernel=(5,1,1),
                 **kwargs):
        self.lateral = lateral
        # self.lateral_norm = lateral_norm
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)

        # lateral=True指示fastpath
        if self.lateral:
            conv1_lateral = LateralConv(
                self.base_channel,
                self.base_channel * self.channel_ratio,
                kernel = self.fusion_kernel,
                stride = (self.speed_ratio, 1, 1)
            )
            lateral_connections = []
            for i in range(len(self.stage_blocks)-1):  # 最后一层不做侧向连接
                channels, _, _ = self.stage_blocks[i]  # 输出通道
                channels = channels // self.channel_ratio
                lateral_connections.append(LateralConv(
                    channels,
                    channels * self.channel_ratio, 
                    kernel = self.fusion_kernel,
                    stride = (self.speed_ratio, 1, 1)
                    )
                )
            self.conv1_lateral = conv1_lateral
            self.layer2_lateral = lateral_connections[0]
            self.layer3_lateral = lateral_connections[1]
            self.layer4_lateral = lateral_connections[2]
            self.layer5_lateral = lateral_connections[3]





pathway_cfg = {
    'mobilenet3d': MobileNet3dPathway,
    # TODO: BNInceptionPathway
}


def build_pathway(cfg, *args, **kwargs):
    """Build pathway.
    """
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop('type')
    if pathway_type not in pathway_cfg:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')

    pathway_cls = pathway_cfg[pathway_type]  #
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@BACKBONES.register_module()
class MobileNet3dSlowFast(nn.Module):
    """Slowfast backbone.
    """
    def __init__(self,
                 pretrained=False,
                 resample_rate=8,  # 输入采样率
                 speed_ratio=8,    # paper中的alpha
                 channel_ratio=8,  # paper中的1/beta
                 slow_pathway=dict(
                     type='mobilenet3d',
                     pretrained=None,
                     lateral=False,
                     base_channels=32),
                 fast_pathway=dict(
                     type='mobilenet3d',
                     pretrained=None,
                     lateral=True,
                     base_channels=32)):
        super().__init__()
        self.pretrained = pretrained
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if fast_pathway['lateral']:
            fast_pathway['speed_ratio'] = speed_ratio
            fast_pathway['channel_ratio'] = channel_ratio
            fast_pathway['width_mult'] = 1/channel_ratio

        self.slow_path = build_pathway(slow_pathway)
        self.fast_path = build_pathway(fast_pathway)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        # Override the init_weights of i3d
        super().init_weights()
        # 初始化侧向连接
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)





    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        x_slow = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / self.resample_rate, 1.0, 1.0),
            )

        x_fast = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0, 1.0),
            )

        x_slow = self.slow_path.conv1(x_slow)
        x_fast = self.fast_path.conv1(x_fast)
        lateral = self.fast_path.conv1_lateral(x_fast)
        x_slow = x_slow + lateral

        x_slow = self.slow_path.layer2(x_slow)
        x_fast = self.fast_path.layer2(x_fast)
        lateral = self.fast_path.layer2_lateral(x_fast)
        x_slow = x_slow + lateral

        x_slow = self.slow_path.layer3(x_slow)
        x_fast = self.fast_path.layer3(x_fast)
        lateral = self.fast_path.layer3_lateral(x_fast)
        x_slow = x_slow + lateral

        x_slow = self.slow_path.layer4(x_slow)
        x_fast = self.fast_path.layer4(x_fast)
        lateral = self.fast_path.layer4_lateral(x_fast)
        x_slow = x_slow + lateral

        x_slow = self.slow_path.layer5(x_slow)
        x_fast = self.fast_path.layer5(x_fast)
        lateral = self.fast_path.layer5_lateral(x_fast)
        x_slow = x_slow + lateral

        x_slow = self.slow_path.layer6(x_slow)
        x_fast = self.fast_path.layer6(x_fast)

        out = (x_slow, x_fast)

        return out

    def init_weights(self, pretrained=None):
        if pretrained:
            self.fast_path.init_weights()
            self.slow_path.init_weights()


if mmdet_imported:
    MMDET_BACKBONES.register_module()(MobileNet3dSlowFast)

if __name__ == '__main__':
    model = MobileNet3dSlowFast()
    model = model.cuda(1)
    #model = nn.DataParallel(model, device_ids=None)
    #print(model)

    input_var = torch.randn(1, 3, 32, 224, 224)  # batch_size, rgb_channel, frame_num, h, w

    input_var = input_var.cuda(1)

    flops,params = profile(model, (input_var,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

    # output = model(input_var)
    # print(output[0].shape)
    # print(output[1].shape)
