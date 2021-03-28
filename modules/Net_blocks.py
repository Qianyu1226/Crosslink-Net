# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True,
                 is_relu=True):
        """
        Convolution + Batch Norm + Relu for 2D feature maps
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:
        :param dilation:
        :param stride:
        :param groups:
        :param is_bn:
        :param is_relu:
        """
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False) #Hout=(H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn is False:
            self.bn = None

        if is_relu is False:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)
        return x

    def merge_bn(self):
        if self.bn is None:
            return

        assert (self.conv.bias is None)

        conv_weight = self.conv.weight.data
        bn_weight = self.bn.weight.data
        bn_bias = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        bn_eps = self.bn.eps

        # https://github.com/sanghoon/pva-faster-rcnn/issues/5
        # https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N, C, KH, KW = conv_weight.size()
        std = 1 / (torch.sqrt(bn_running_var + bn_eps))
        std_bn_weight = (std * bn_weight).repeat(C * KH * KW, 1).t().contiguous().view(N, C, KH, KW)
        conv_weight_hat = std_bn_weight * conv_weight
        conv_bias_hat = (bn_bias - bn_weight * std * bn_running_mean)

        self.bn = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                              kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation,
                              groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat  # fill in
        self.conv.bias.data = conv_bias_hat


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size, padding, is_bn=True): #x_channel is in_channel, y_channel is out_channel
        super(StackEncoder, self).__init__()
        #padding = (kernel_size - 1) // 2  #padding=1
        self.encode1 = ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1, is_bn=is_bn)
        self.encode = nn.Sequential(
            #ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
             #            groups=1, is_bn=is_bn),
            ConvBnRelu2d(y_channels, y_channels//2, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1, is_bn=is_bn),
            ConvBnRelu2d(y_channels//2, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1, is_bn=is_bn)
        )

    def forward(self, x):
        x = self.encode1(x)
        x_skip = x
        x = self.encode(x)
        x = x_skip + x
        #x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x#, x_small # 后一个卷积层及pooling层同时输出


class StackDecoder(nn.Module):
    def __init__(self, v_big_channels, x_channels, h_big_channels, y_channels, kernel_size=3, is_bn=True):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2 #padding=1
        self.decode1 = ConvBnRelu2d(v_big_channels + h_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding,
                                    dilation=1,
                                    stride=1, groups=1, is_bn=is_bn)
        self.decode = nn.Sequential(
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1, is_bn=is_bn),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1, is_bn=is_bn)
        )

    def forward(self, v_big, h_big, x): #v_big是当前层的上层pooling之前的卷积结果，x是当前层
        N, C, H, W = v_big.size()
        y = F.upsample(x, size=(H, W), mode='bilinear')
        y = torch.cat((v_big, y, h_big), 1) #横着拼在一起
        y = self.decode1(y)
        y_skip = y
        y = self.decode(y)
        y = y_skip + y
        return y

