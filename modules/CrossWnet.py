# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Net_blocks import StackDecoder, StackEncoder, ConvBnRelu2d


class UNet(nn.Module):
    def __init__(self, in_shape, is_bn=False):
        super(UNet, self).__init__()
        C, H, W = in_shape

        self.v_down1_1 = StackEncoder(C, 32, kernel_size=(5, 3), padding=(2, 1), is_bn=is_bn)  # 后一个卷积层及pooling层同时输出
        self.v_down1_4 = StackEncoder(C, 32, (3, 3), (1, 1), is_bn)
        self.v_down1_2 = StackEncoder(C, 32, (3, 1), (1, 0), is_bn)
        self.v_down1_3 = ConvBnRelu2d(32*3, 32, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False) #after concat
        self.v_down1_res = ConvBnRelu2d(C, 32, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False) #for residual
       # self.v_down1_pool = ConvBnRelu2d(32, 32, kernel_size=3, padding=1, stride=2, is_bn=is_bn)
       # self.v_down1_5 = ConvBnRelu2d(32 * 2, 32, kernel_size=1, padding=0, stride=1, is_bn=is_bn)

        self.v_down2_1 = StackEncoder(32, 64, (5, 3), (2, 1), is_bn)  # 32
        self.v_down2_2 = StackEncoder(32, 64, (3, 1), (1, 0), is_bn)
        self.v_down2_4 = StackEncoder(32, 64, (3, 3), (1, 1), is_bn)
        self.v_down2_3 = ConvBnRelu2d(64 * 3, 64, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        self.v_down2_res = ConvBnRelu2d(32, 64, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        #self.v_down2_pool = ConvBnRelu2d(64, 64, kernel_size=3, padding=1, stride=2, is_bn=is_bn)
        #self.v_down2_5 = ConvBnRelu2d(64 * 2, 64, kernel_size=1, padding=0, stride=1, is_bn=is_bn)

        self.v_down3_1 = StackEncoder(64, 128, (5, 3), (2, 1), is_bn)  # 32
        self.v_down3_2 = StackEncoder(64, 128, (3, 1), (1, 0), is_bn)
        self.v_down3_4 = StackEncoder(64, 128, (3, 3), (1, 1), is_bn)
        self.v_down3_3 = ConvBnRelu2d(128 * 3, 128, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        self.v_down3_res = ConvBnRelu2d(64, 128, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        #self.v_down3_pool = ConvBnRelu2d(128, 128, kernel_size=3, padding=1, stride=2, is_bn=is_bn)
        #self.v_down3_5 = ConvBnRelu2d(128 * 2, 128, kernel_size=1, padding=0, stride=1, is_bn=is_bn)
        self.for_atten3_1 = ConvBnRelu2d(128, 64, kernel_size=3, padding=1, stride=1, is_bn=is_bn)
        self.for_atten3_2 = ConvBnRelu2d(64, 1, kernel_size=3, padding=1, stride=1, is_bn=is_bn, is_relu=False)

        self.v_down4_1 = StackEncoder(128, 256, (5, 3), (2, 1), is_bn)  # 16
        self.v_down4_2 = StackEncoder(128, 256, (3, 1), (1, 0), is_bn)
        self.v_down4_4 = StackEncoder(128, 256, (3, 3), (1, 1), is_bn)
        self.v_down4_3 = ConvBnRelu2d(256 * 3, 256, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        self.v_down4_res = ConvBnRelu2d(128, 256, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        #self.v_down4_pool = ConvBnRelu2d(256, 256, kernel_size=3, padding=1, stride=2, is_bn=is_bn)
        #self.v_down4_5 = ConvBnRelu2d(256 * 2, 256, kernel_size=1, padding=0, stride=1, is_bn=is_bn)

        self.v_down5_1 = StackEncoder(256, 512, (5, 3), (2, 1), is_bn)  # 8
        self.v_down5_2 = StackEncoder(256, 512, (3, 1), (1, 0), is_bn)
        self.v_down5_4 = StackEncoder(256, 512, (3, 3), (1, 1), is_bn)
        self.v_down5_3 = ConvBnRelu2d(512 * 3, 512, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        self.v_down5_res = ConvBnRelu2d(256, 512, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        #self.v_down5_pool = ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=2, is_bn=is_bn)
        #self.v_down5_5 = ConvBnRelu2d(512 * 2, 512, kernel_size=1, padding=0, stride=1, is_bn=is_bn)

        self.h_down1_1 = StackEncoder(C, 32, kernel_size=(3, 5), padding=(1, 2), is_bn=is_bn)  # 后一个卷积层及pooling层同时输出
        self.h_down1_2 = StackEncoder(C, 32, (1, 3), (0, 1), is_bn)
        # self.h_down1_skip = ConvBnRelu2d(C, 32, kernel_size=3, padding=1, stride=1, is_bn=is_bn)
        self.h_down1_3 = ConvBnRelu2d(C, 32, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        #self.h_down1_4 = ConvBnRelu2d(32 * 3, 32, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)

        self.h_down2_1 = StackEncoder(32, 64, (3, 5), (1, 2), is_bn)  # 64
        self.h_down2_2 = StackEncoder(32, 64, (1, 3), (0, 1), is_bn)
        # self.h_down2_skip = ConvBnRelu2d(16, 16, kernel_size=3, padding=1, stride=1, is_bn=is_bn)
        self.h_down2_3 = ConvBnRelu2d(32, 64, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        #self.h_down2_4 = ConvBnRelu2d(64 * 3, 64, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)

        self.h_down3_1 = StackEncoder(64, 128, (3, 5), (1, 2), is_bn)  # 32
        self.h_down3_2 = StackEncoder(64, 128, (1, 3), (0, 1), is_bn)
        # self.h_down3_skip = ConvBnRelu2d(32, 32, kernel_size=3, padding=1, stride=1, is_bn=is_bn)
        self.h_down3_3 = ConvBnRelu2d(64, 128, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        #self.h_down3_4 = ConvBnRelu2d(128 * 3, 128, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)

        self.h_down4_1 = StackEncoder(128, 256, (3, 5), (1, 2), is_bn)  # 16
        self.h_down4_2 = StackEncoder(128, 256, (1, 3), (0, 1), is_bn)
        # self.h_down4_skip = ConvBnRelu2d(64, 64, kernel_size=3, padding=1, stride=1, is_bn=is_bn)
        self.h_down4_3 = ConvBnRelu2d(128, 256, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)
        #self.h_down4_4 = ConvBnRelu2d(256 * 3, 256, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)

        self.h_down5_1 = StackEncoder(256, 512, (3, 5), (1, 2), is_bn)  # 8
        self.h_down5_2 = StackEncoder(256, 512, (1, 3), (0, 1), is_bn)
        # self.h_down5_skip = ConvBnRelu2d(128, 128, kernel_size=3, padding=1, stride=1, is_bn=is_bn)
        self.h_down5_3 = ConvBnRelu2d(256, 512, kernel_size=1, padding=0, stride=1, is_bn=is_bn,
                                      is_relu=False)
        #self.h_down5_4 = ConvBnRelu2d(512 * 3, 512, kernel_size=1, padding=0, stride=1, is_bn=is_bn, is_relu=False)

        # 两网最后一个pooling层拼在一起以后再1*1卷积
        self.center = ConvBnRelu2d(1024, 512, kernel_size=1, padding=0, stride=1, is_bn=is_bn)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 8
        # h_big_channels,v_big_channels是当前层的上层pooling之前的卷积层的channel, x_channels是当前层的，之和相当于in_channel
        self.up5 = StackDecoder(512, 512, 512, 256, kernel_size=3, is_bn=is_bn)
        self.up4 = StackDecoder(256, 256, 256, 128, kernel_size=3, is_bn=is_bn)  # 16
        self.up3 = StackDecoder(128, 128, 128, 64, kernel_size=3, is_bn=is_bn)  # 32
        self.up2 = StackDecoder(64, 64, 64, 32, kernel_size=3, is_bn=is_bn)  # 64
        self.up1 = StackDecoder(32, 32, 32, 16, kernel_size=3, is_bn=is_bn)  # 128
        # self.up1 = StackDecoder(16, 16, 16, 8, kernel_size=3, is_bn=is_bn)  # 256

        self.classify = nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        v_out = x
        v_out1_1 = self.v_down1_1(v_out)
        v_out1_2 = self.v_down1_2(v_out)
        v_out1_3 = self.v_down1_4(v_out)
        v_cat = torch.cat((v_out1_1, v_out1_2, v_out1_3), 1)
        v_out1 = self.v_down1_3(v_cat)
        v_out1_res = self.v_down1_res(v_out)
        v_down1 = F.relu(v_out1 + v_out1_res, inplace=True)
        v_out = self.maxpool(v_down1)

        v_out2_1 = self.v_down2_1(v_out)
        v_out2_2 = self.v_down2_2(v_out)
        v_out2_3 = self.v_down2_4(v_out)
        v_cat = torch.cat((v_out2_1, v_out2_2, v_out2_3), 1)
        v_out2 = self.v_down2_3(v_cat)
        v_out2_res = self.v_down2_res(v_out)
        v_down2 = F.relu(v_out2 + v_out2_res, inplace=True)
        v_out = self.maxpool(v_down2)

        v_out3_1 = self.v_down3_1(v_out)
        v_out3_2 = self.v_down3_2(v_out)
        v_out3_3 = self.v_down3_4(v_out)
        v_cat = torch.cat((v_out3_1, v_out3_2, v_out3_3), 1)
        v_out3 = self.v_down3_3(v_cat)
        v_out3_res = self.v_down3_res(v_out)
        v_down3 = F.relu(v_out3 + v_out3_res, inplace=True)
        v_out = self.maxpool(v_down3)

        v_atten3 = self.for_atten3_1(v_down3)
        v_atten3 = self.for_atten3_2(v_atten3)
        v_atten3 = F.sigmoid(v_atten3)

        v_out4_1 = self.v_down4_1(v_out)
        v_out4_2 = self.v_down4_2(v_out)
        v_out4_3 = self.v_down4_4(v_out)
        v_cat = torch.cat((v_out4_1, v_out4_2, v_out4_3), 1)
        v_out4 = self.v_down4_3(v_cat)
        v_out4_res = self.v_down4_res(v_out)
        v_down4 = F.relu(v_out4 + v_out4_res, inplace=True)
        v_out = self.maxpool(v_down4)

        v_out5_1 = self.v_down5_1(v_out)
        v_out5_2 = self.v_down5_2(v_out)
        v_out5_3 = self.v_down5_4(v_out)
        v_cat = torch.cat((v_out5_1, v_out5_2, v_out5_3), 1)
        v_out5 = self.v_down5_3(v_cat)
        v_out5_res = self.v_down5_res(v_out)
        v_down5 = F.relu(v_out5 + v_out5_res, inplace=True)
        v_out = self.maxpool(v_down5)

        h_out = x
        h_out1_1 = self.h_down1_1(h_out)  # 32
        h_out1_2 = self.h_down1_2(h_out)  # 32
        h_out1_3 = self.h_down1_3(h_out)
        # h_out1_3 = self.h_down1_skip(h_out)  #16，要加到up1最后一层
        h_cat = torch.cat((h_out1_1, h_out1_2, h_out1_3), 1)  # 64+16
        h_out1 = self.v_down1_3(h_cat)
        h_out1_res = self.v_down1_res(h_out)
        h_down1 = F.relu(h_out1 + h_out1_res, inplace=True)
        h_out = self.maxpool(h_down1)

        h_out2_1 = self.h_down2_1(h_out)  # 32, 64
        h_out2_2 = self.h_down2_2(h_out)  # 32, 64
        # h_out2_3 = self.h_down2_skip(h_out)#32,32
        h_out2_3 = self.h_down2_3(h_out)
        h_cat = torch.cat((h_out2_1, h_out2_2, h_out2_3), 1)  # 64 + 64+32
        h_out2 = self.v_down2_3(h_cat)  # 64 + 64+32, 32
        h_out2_res = self.v_down2_res(h_out)  # 32, 64
        h_down2 = F.relu(h_out2 + h_out2_res, inplace=True)
        h_out = self.maxpool(h_down2)

        h_out3_1 = self.h_down3_1(h_out)
        h_out3_2 = self.h_down3_2(h_out)
        h_out3_3 = self.h_down3_3(h_out)
        # h_out3_3 = self.h_down3_skip(h_out)
        h_cat = torch.cat((h_out3_1, h_out3_2, h_out3_3), 1)
        h_out3 = self.v_down3_3(h_cat)
        h_out3_res = self.v_down3_res(h_out)
        h_down3 = F.relu(h_out3 + h_out3_res, inplace=True)
        h_out = self.maxpool(h_down3)

        h_atten3 = self.for_atten3_1(h_down3)
        h_atten3 = self.for_atten3_2(h_atten3)
        h_atten3 = F.sigmoid(h_atten3)

        h_out4_1 = self.h_down4_1(h_out)
        h_out4_2 = self.h_down4_2(h_out)
        h_out4_3 = self.h_down4_3(h_out)
        # h_out4_3 = self.h_down4_skip(h_out)
        h_cat = torch.cat((h_out4_1, h_out4_2, h_out4_3), 1)
        h_out4 = self.v_down4_3(h_cat)
        h_out4_res = self.v_down4_res(h_out)
        h_down4 = F.relu(h_out4 + h_out4_res, inplace=True)
        h_out = self.maxpool(h_down4)

        h_out5_1 = self.h_down5_1(h_out)
        h_out5_2 = self.h_down5_2(h_out)
        h_out5_3 = self.h_down5_3(h_out)
        # h_out5_3 = self.h_down5_skip(h_out)
        h_cat = torch.cat((h_out5_1, h_out5_2, h_out5_3), 1)
        h_out5 = self.v_down5_3(h_cat)
        h_out5_res = self.v_down5_res(h_out)
        h_down5 = F.relu(h_out5 + h_out5_res, inplace=True)
        h_out = self.maxpool(h_down5)

        # 两个分支连在一起
        out = torch.cat((v_out, h_out), 1)  # 横着拼在一起
        out = self.center(out)

        out = self.up5(v_down5, h_down5, out)  # 这个out是center的输出，down5是它的上一层pooling之前的卷积结果
        out = self.up4(v_down4, h_down4, out)
        out = self.up3(v_down3, h_down3, out)
        out = self.up2(v_down2, h_down2, out)
        out = self.up1(v_down1, h_down1, out)
        # out = self.up1(v_cat, h_cat, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out, v_atten3, h_atten3 #用来计算loss