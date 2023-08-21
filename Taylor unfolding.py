import torch
from torch import Tensor
import torch.nn as nn
import math


class Taylor(nn.Module):
    def __init__(self, order_num: int):
        super(Taylor, self).__init__()
        self.F = Fnet()
        self.P = MGC()
        self.order_num = order_num
        self.up = nn.Upsample(scale_factor=4)
        highorderblock_list = []
        for i in range(order_num):

            highorderblock_list.append(Qth())
        self.highorderblock_list = nn.ModuleList(highorderblock_list)

    def forward(self, Input1: Tensor, Input2: Tensor):
        # print(Input1.shape, Input2.shape)
        # torch.Size([2, 8, 64, 64]) torch.Size([2, 8, 256, 256])
        # torch.Size([16, 8, 256, 256]) torch.Size([16, 8, 64, 64])
        # print(Input2.shape)
        # print(Input1.shape)torch.Size([4, 1, 256, 256])torch.Size([4, 8, 64, 64])
        feature_head = self.P(Input1)
        # print(feature_head.shape)
        # torch.Size([16, 32, 256, 256])

        Input2 = self.up(Input2)
        zero_term = self.F(Input2)
        # print(zero_term.shape)
        # torch.Size([2, 8, 256, 256])
        # torch.Size([16, 4, 64, 64])
        out_term, pre_term = zero_term, zero_term
        a = []
        # b = []
        a.append(out_term)
        # print(pre_term.shape, Input1.shape,feature_head.shape)torch.Size([4, 8, 256, 256]) torch.Size([4, 8, 256, 256]) torch.Size([4, 32, 256, 256])
        # torch.Size([2, 8, 256, 256]) torch.Size([2, 8, 256, 256]) torch.Size([2, 32, 256, 256])
        # torch.Size([4, 8, 1024, 1024]) torch.Size([4, 8, 1024, 1024]) torch.Size([4, 32, 64, 64])
        for order_id in range(self.order_num):
            update_term = self.highorderblock_list[order_id](pre_term, Input2, feature_head) + order_id * pre_term
            pre_term = update_term
            out_term = out_term + update_term / math.factorial(order_id+1)
            # a.append(pre_term)
            a.append(out_term)
            # b.append(update_term / math.factorial(order_id+1))
        # print(out_term.shape)torch.Size([4, 8, 256, 256])
        # torch.Size([16, 4, 64, 64])
        # out_term = self.up(out_term)
        # print(out_term.shape)
        return out_term, a


class Fnet(nn.Module):
    def __init__(self):
        super(Fnet, self).__init__()
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(0.1)
        )
        self.conv3 = nn.Sequential(
             nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(0.1),
             nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
        )
        self.ST = SpectralTransform()

    def forward(self, a):
        # print(a.shape)
        y = self.ST(a)
        # torch.Size([16, 4, 64, 64])
        x1 = self.conv1(a)
        x2 = self.conv2(x1)
        x3 = x1 + x2
        x3 = torch.cat((y, x3), dim=1)
        x4 = self.conv3(x3)
        x = a + x4
        # print(x.shape)torch.Size([16, 4, 64, 64])
        return x


class SpectralTransform(nn.Module):
    def __init__(self):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(0.1)
        )
        # self.conv0 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(0.1)
        )
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        # z = self.conv0(y)
        # print(y.shape)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(y, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((y.shape[0], -1,) + ffted.size()[3:])
        # print(ffted.shape)
        z1 = self.conv2(ffted)
        # print(z1.shape)
        ffted = z1.view((z1.shape[0], -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        z4 = output + y
        # print(z4.shape)
        z5 = self.conv3(z4)
        return z5


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        # self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        # self.up = nn.Upsample(scale_factor=16)
        self.conv4 = nn.Sequential(
             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input1, input2):
        # print(input1.shape, input2.shape)
        # torch.Size([16, 32, 256, 256]) torch.Size([16, 32, 256])
        # input1 = self.up(input1)
        # print(input1.shape)
        # torch.Size([16, 32, 256, 256])
        z = torch.cat((input1, input2), dim=1)
        # print(z.shape)torch.Size([16, 64, 256, 256])
        z = self.conv4(z)
        z = torch.sigmoid(z)
        # print(z.shape)
        # torch.Size([16, 32, 256, 256])
        return z


class SAB1(nn.Module):
    def __init__(self):
        super(SAB1, self).__init__()
        # self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        # self.up = nn.Upsample(scale_factor=16)
        self.conv4 = nn.Sequential(
             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input1, input2):
        # print(input1.shape, input2.shape)
        # torch.Size([16, 32, 256, 256]) torch.Size([16, 32, 64, 64])
        # input1 = self.up(input1)
        # print(input1.shape)
        # torch.Size([16, 32, 256, 256])
        z = torch.cat((input1, input2), dim=1)
        # print(z.shape)torch.Size([16, 64, 256, 256])
        z = self.conv4(z)
        z = torch.sigmoid(z)
        # print(z.shape)
        # torch.Size([16, 32, 256, 256])
        return z


class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1=1, kernel_size2=3, kernel_size3=5, groups=3):
        super(GConv2d, self).__init__()
        # self.conv2d_block = nn.ModuleList()
        self.groups = groups
        # self.conv2d_block.append(nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
        #                                    kernel_size=kernel_size1, padding=kernel_size1 // 2))
        # self.conv2d_block.append(RCAB())
        # self.conv2d_block.append(nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
        #                                    kernel_size=kernel_size2, padding=kernel_size2 // 2))
        # self.conv2d_block.append(RCAB())
        # self.conv2d_block.append(nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
        #                                    kernel_size=kernel_size3, padding=kernel_size3 // 2))
        # self.conv2d_block.append(RCAB())
        self.conv1 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.conv2 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size2, padding=kernel_size2 // 2)
        self.conv3 = nn.Conv2d(in_channels=in_channels // groups, out_channels=out_channels // groups,
                               kernel_size=kernel_size3, padding=kernel_size3 // 2)
        self.scab2 = RCAB(kernel=3, pad=1)
        self.scab3 = RCAB(kernel=3, pad=1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # print(x.shape)
        x1, x2, x3 = torch.chunk(x, self.groups, 1)
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y2 = self.scab2(y2)
        y3 = self.conv3(x3)
        y3 = self.scab3(y3)
        y3 = self.conv4(y3)
        z = torch.cat((y2, y3), dim=1)
        z = self.conv5(z)
        y4 = torch.cat((y1, z), dim=1)
        y4 = self.conv6(y4)
        return y4

        # return torch.cat([filterg(xg) for filterg, xg in zip(self.conv2d_block, torch.chunk(x, self.groups, 1))], dim=1)


class RCAB(nn.Module):
    def __init__(self, kernel, pad):
        super(RCAB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel, stride=1, padding=pad),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel, stride=1, padding=pad),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel, stride=1, padding=pad),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel, stride=1, padding=pad),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        y = self.conv1(x)
        y1 = self.avg_pool(y)
        # print(y1.shape)
        y2 = self.conv2(y1)
        y3 = y * y2
        z = y3 + x
        # print(z.shape)
        return z


## Channel Attention (CA) Layer
# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
#
#
# ## Residual Channel Attention Block (RCAB)
# class RCAB(nn.Module):
#     def __init__(
#             self, conv, n_feat, kernel_size, reduction,
#             bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(RCAB, self).__init__()
#         modules_body = []
#         # i=0时使用激活函数。Conv ->  act -> Conv
#         for i in range(2):
#             # conv 为标准卷积
#             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         modules_body.append(CALayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         print(x.shape)
#         res = self.body(x)
#         # res = self.body(x).mul(self.res_scale)
#         res += x
#         return res


class MGC(nn.Module):
    def __init__(self):
        super(MGC, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv = GConv2d(in_channels=24, out_channels=24, kernel_size1=1, kernel_size2=3, kernel_size3=5)
        self.conv0 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Sequential(
             nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, stride=1, padding=0),
             nn.LeakyReLU(0.1)
         )

    def forward(self, Input2):
        # print(Input2.shape)
        # torch.Size([4, 1, 256, 256])
        # torch.Size([16, 1, 256, 256])
        Fp = self.conv5(Input2)
        # print(Fp.shape)
        # torch.Size([16, 24, 256, 256])
        Fp = self.conv(Fp)
        Fp = self.conv0(Fp)
        Fp = self.conv6(Fp)
        # print(Fp.shape)
        # torch.Size([16, 32, 256, 256])
        return Fp


class MGC1(nn.Module):
    def __init__(self):
        super(MGC1, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv = GConv2d(in_channels=24, out_channels=24, kernel_size1=1, kernel_size2=3, kernel_size3=5)
        self.conv0 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Sequential(
             nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, stride=1, padding=0),
             nn.LeakyReLU(0.1)
         )

    def forward(self, Input2):
        # print(Input2.shape)torch.Size([16, 1, 256, 256])
        Fp = self.conv5(Input2)
        # print(Fp.shape)
        # torch.Size([16, 24, 256, 256])
        Fp = self.conv(Fp)
        Fp = self.conv0(Fp)
        Fp = self.conv6(Fp)
        # print(Fp.shape)
        # torch.Size([16, 32, 256, 256])
        return Fp


class Qth(nn.Module):
    def __init__(self):
        super(Qth, self).__init__()
        self.MGC = MGC()
        self.MGC1 = MGC1()
        self.SAB = SAB()
        self.SAB1 = SAB1()
        self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.up = nn.Upsample(scale_factor=16, mode='bicubic', align_corners=True)
        self.down = nn.MaxPool2d(4)
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
        )
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, pre_term: Tensor, Input2: Tensor, feature_head: Tensor):
        # print(pre_term.shape, Input2.shape, feature_head.shape)torch.Size([4, 8, 256, 256]) torch.Size([4, 8, 256, 256]) torch.Size([4, 32, 256, 256])
        # torch.Size([2, 8, 256, 256]) torch.Size([2, 8, 256, 256]) torch.Size([2, 32, 256, 256])
        # torch.Size([4, 8, 1024, 1024]) torch.Size([4, 8, 1024, 1024]) torch.Size([4, 32, 64, 64])
        # torch.Size([16, 4, 256, 256])torch.Size([16, 4, 64, 64])torch.Size([16, 32, 256, 256]) torch.Size([4, 8, 1024, 1024]) torch.Size([4, 8, 1024, 1024]) torch.Size([4, 32, 64, 64])
        pre_term = self.conv(pre_term)
        # print(pre_term.shape)torch.Size([4, 1, 256, 256])
        # torch.Size([4, 1, 1024, 1024])
        # torch.Size([16, 1, 64, 64])
        x = self.MGC1(pre_term)
        # print(x.shape)
        # torch.Size([4, 32, 1024, 1024])
        # torch.Size([16, 32, 64, 64])
        Input2 = self.conv(Input2)
        x2 = self.MGC1(Input2)
        # print(x2.shape)torch.Size([2, 32, 256, 256])
        # torch.Size([4, 32, 1024, 1024])
        # torch.Size([16, 32, 64, 64])
        # x0 = self.up(x2)
        # print(x0.shape)
        x3 = self.SAB(feature_head, x2)
        # feature_head = self.up(feature_head)
        # print(x3.shape,feature_head.shape)
        # torch.Size([2, 32, 64, 64]) torch.Size([2, 32, 1024, 1024])
        # torch.Size([16, 32, 256, 256])
        x3 = x3 * feature_head
        # print(x3.shape)torch.Size([4, 32, 256, 256])
        # torch.Size([16, 32, 256, 256])
        # print(x.shape, x2.shape)
        # torch.Size([16, 32, 64, 64]) torch.Size([16, 32, 64, 64])
        # x3 = self.down(x3)
        # print(x3.shape)
        # torch.Size([16, 32, 64, 64])
        x4 = self.SAB(x, x2)
        # print(x4.shape)torch.Size([4, 32, 256, 256])
        # torch.Size([16, 32, 64, 64])
        x4 = x4 * x
        # print(x4.shape)
        # torch.Size([16, 32, 64, 64])
        x5 = x4 + x2 + x3
        # print(x5.shape)
        # torch.Size([16, 32, 64, 64])
        x5 = self.conv7(x5)
        # print(x5.shape)
        # torch.Size([4, 8, 256, 256])
        # torch.Size([16, 4, 64, 64])
        return x5


if  __name__ == "__main__":
    dc = Taylor(order_num=4)
    A = torch.FloatTensor(size=(1, 1, 256, 256)).normal_(0, 1)
    B = torch.FloatTensor(size=(1, 4, 64, 64)).normal_(0, 1)
    out = dc(A, B)
    # print()