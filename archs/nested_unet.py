
import torch
import torch.nn as nn
from torch.nn import Conv2d

def conv2_depth(in_ch, out_ch, kernel_size, pad):
    depth_conv = Conv2d(
        in_channels=in_ch,
        out_channels=in_ch,
        kernel_size=kernel_size,
        padding=pad,
        groups=in_ch,
    )
    point_conv = Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
    depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
    return depthwise_separable_conv

class VGGB(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv2_depth(in_channels, middle_channels, 3, pad=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = conv2_depth(middle_channels, out_channels, 3, pad=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        # print('conv1',out.shape)
        out = self.bn1(out)
        # print('bn1',out.shape)
        out = self.relu(out)
        # print('relu1',out.shape)
        out = self.conv2(out)
        # print('conv2',out.shape)
        out = self.bn2(out)
        # print('bn2',out.shape)
        out = self.relu(out)
        # print('relu2',out.shape)
        return out

class NestedUNetsmall(nn.Module):
    def __init__(
        self, num_classes=1, input_channels=1, deep_supervision=False, **kwargs
    ):
        super().__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [8, 16, 32, 64, 128]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGB(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGB(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGB(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGB(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGB(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGB(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGB(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGB(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGB(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGB(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGB(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGB(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGB(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGB(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGB(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = conv2_depth(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = conv2_depth(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = conv2_depth(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = conv2_depth(nb_filter[0], num_classes, kernel_size=1)
        else:
            # self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.finaly = conv2_depth(nb_filter[0], num_classes, kernel_size=3, pad=1)

    def getnumberofparams(self, model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        # print('x0_0',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1',x1_1.shape)

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3',x0_3.shape)

        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            # output = self.final(x0_4)
            outputy = self.finaly(x0_4)
            # outputy = self.finaly(x0_2)

            return outputy

if __name__ == "__main__":
    model = NestedUNetsmall(input_channels=1, num_classes=1)
    
    num_params = model.getnumberofparams(model)
    print(f"Number of parameters: {num_params}")
    x = torch.randn(1, 1, 480, 480)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
