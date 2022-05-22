'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''

import torch
import torch.nn as nn
from thop import profile

class Convbn(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.convbn = nn.Sequential(
            nn.Conv3d(inp, oup, kernel_size=(1,3,3), stride=stride, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.convbn(x)
        return out



class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        w = 6
        self.dwconv = nn.Sequential(
            nn.Conv3d(in_planes, in_planes*w, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes*w, in_planes*w, kernel_size=(1,3,3), stride=stride, padding=(0,1,1), groups=in_planes*w, bias=False),
            nn.BatchNorm3d(in_planes*w),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes*w, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        out = self.dwconv(x)
        if out.shape == x.shape:
            out = x + out
        return out


class MobileNet3dV2(nn.Module):
    def __init__(self, num_classes=600,
                sample_size=224,
                width_mult=1.,
                pretrained=False,
                base_channels=32):
        super().__init__()

        cfg = [
        # c, n, s
        [16, 1, (1,1,1)],
        [24, 2, (1,2,2)],
        [32, 3, (1,2,2)],
        [64, 4, (1,2,2)],
        [96, 3, (1,1,1)],
        [160, 3, (1,2,2)],
        [320, 1, (1,1,1)],
        ]
        print("-"*50,base_channels)
        print("-"*50,width_mult)
        base_channels = int(base_channels * width_mult)
        self.base_channel = base_channels
        self.stage_blocks = cfg
        self.width_mult = width_mult
        self.last_channel = int(cfg[-1][0]*self.width_mult)

        # conv layer
        conv1 = Convbn(3, base_channels, (1,2,2))
        conv9 = Convbn(self.last_channel, self.last_channel * 4, (1,2,2))  # 320 1280
        layers = []
        # layers
        for i in range(len(self.stage_blocks)):
            layer = []
            c, n, s = self.stage_blocks[i]  # 输出通道，重复次数，步长（仅限每个layer的第一个block）
            output_channel = int(c * self.width_mult)
            for j in range(n):
                stride = s if j == 0 else 1
                layer.append(Block(base_channels, output_channel, stride))
                base_channels = output_channel
            layers.append(nn.Sequential(*layer))

        self.conv1 = conv1
        self.layer2 = nn.Sequential(layers[0])
        self.layer3 = nn.Sequential(layers[1])
        self.layer4 = nn.Sequential(layers[2])
        self.layer5 = nn.Sequential(layers[3])
        self.layer6 = nn.Sequential(layers[4])
        self.layer7 = nn.Sequential(layers[5])
        self.layer8 = nn.Sequential(layers[6])
        self.conv9 = conv9
        #self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.conv9(x)
        return x

    def init_weights(self, pretrained=None):
        # 使用默认初始化
        pass

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNet3dV2(**kwargs)
    return model



if __name__ == '__main__':
    model = get_model(num_classes=600, sample_size = 112, width_mult=1.)
    model = model.cuda(1)
    #print(model)
    #model = nn.DataParallel(model, device_ids=None)

    input_var = torch.randn(1, 3, 32, 224, 224)  # batch_size, rgb_channel, frame_num, h, w
    input_var = input_var.cuda(1)

    flops,params = profile(model, (input_var,))

    output = model(input_var)
    print(output.shape)
