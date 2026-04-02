"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.

这段代码实现了一个混合编码器架构，结合了CNN和Transformer的优势，用于目标检测任务。
主要包含卷积归一化层、CSP模块、Transformer编码器以及特征金字塔网络等组件。
"""

import copy  # 用于深度复制模块
from collections import OrderedDict  # 有序字典，保持层定义的顺序

import torch
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from ultralytics.nn.modules import Conv

from engine.deim.utils import get_activation  # 从本地工具模块导入激活函数获取方法
from engine.core import register  # 注册装饰器，用于模型注册

from engine.newaddmodules.add_ELAM import *
from engine.newaddmodules.add_downsample import *
from engine.newaddmodules.add_upsample import  *
from engine.newaddmodules.add_fusion import *
from engine.newaddmodules.add_MLP import *
from engine.newaddmodules.Secondary_innovation import *


__all__ = ['HybridEncoder_AiGuai']  # 定义模块公开接口

class ConvNormLayer_fuse(nn.Module):
    """可融合的卷积-归一化层，支持训练后转换为部署模式"""
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        """
        初始化卷积归一化层
        Args:
            ch_in: 输入通道数
            ch_out: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            g: 分组卷积的组数，默认为1(普通卷积)
            padding: 填充大小，None时自动计算保持尺寸的padding
            bias: 是否使用偏置
            act: 激活函数类型，None时为恒等映射
        """
        super().__init__()
        # 自动计算保持尺寸的padding
        padding = (kernel_size-1)//2 if padding is None else padding
        # 定义卷积层
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        # 批归一化层
        self.norm = nn.BatchNorm2d(ch_out)
        # 激活函数，None时为恒等映射
        self.act = nn.Identity() if act is None else get_activation(act)
        # 保存参数用于后续部署转换
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.g = g
        self.padding = padding
        self.bias = bias

    def forward(self, x):
        """前向传播"""
        # 部署模式下使用融合后的卷积
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            # 训练模式下正常执行卷积+批归一化
            y = self.norm(self.conv(x))
        return self.act(y)  # 应用激活函数

    def convert_to_deploy(self):
        """转换为部署模式，融合卷积和批归一化层"""
        if not hasattr(self, 'conv_bn_fused'):
            # 创建融合后的卷积层
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)  # 融合后需要偏置

        # 获取等效的融合核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        # 删除原始层
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        """计算融合后的等效核和偏置"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor()
        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        """实际融合卷积和批归一化的计算"""
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)  # 重塑为与核相同的维度
        return kernel * t, beta - running_mean * gamma / std  # 融合后的核和偏置


class ConvNormLayer(nn.Module):
    """基础的卷积-归一化层"""
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        # 自动计算保持尺寸的padding
        padding = (kernel_size-1)//2 if padding is None else padding
        # 定义卷积层
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        # 批归一化层
        self.norm = nn.BatchNorm2d(ch_out)
        # 激活函数
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        """前向传播：卷积->批归一化->激活"""
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    """空间通道下采样模块"""
    def __init__(self, c1, c2, k, s, act=None):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            act: 激活函数类型
        """
        super().__init__()
        # 1x1卷积改变通道数
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        # 深度可分离卷积进行空间下采样
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        """前向传播：1x1卷积->深度可分离卷积"""
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    """VGG风格的基本块，包含3x3和1x1两条路径"""
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        # 3x3卷积路径
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        # 1x1卷积路径
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        # 激活函数
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        """前向传播：两条路径相加后激活"""
        if hasattr(self, 'conv'):
            # 部署模式下使用融合后的单一卷积
            y = self.conv(x)
        else:
            # 训练模式下使用两条路径的和
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        """转换为部署模式，融合3x3和1x1路径"""
        if not hasattr(self, 'conv'):
            # 创建融合后的3x3卷积
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        # 获取等效核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # 删除原始层
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        """计算融合后的等效核和偏置"""
        # 融合3x3路径
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        # 融合1x1路径
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        # 将1x1核填充为3x3后相加
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """将1x1卷积核填充为3x3"""
        if kernel1x1 is None:
            return 0
        else:
            # 四周各填充1个0
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):

        """融合卷积和批归一化层"""
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)  # 重塑为与核相同的维度
        return kernel * t, beta - running_mean * gamma / std  # 融合后的核和偏置


class CSPLayer(nn.Module):
    """跨阶段部分网络层，包含两条路径的特征融合"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_blocks: 瓶颈块数量
            expansion: 扩展因子，控制隐藏层通道数
            bias: 是否使用偏置
            act: 激活函数类型
            bottletype: 瓶颈块类型
        """
        super(CSPLayer, self).__init__()
        # 计算隐藏层通道数
        hidden_channels = int(out_channels * expansion)
        # 两条路径的初始1x1卷积
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        # 瓶颈层序列
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        # 输出1x1卷积，如果通道数不同
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()  # 通道数相同时使用恒等映射

    def forward(self, x):
        """前向传播：两条路径相加后输出"""
        x_2 = self.conv2(x)  # 直接路径
        x_1 = self.conv1(x)  # 瓶颈路径
        x_1 = self.bottlenecks(x_1)  # 通过瓶颈层
        return self.conv3(x_1 + x_2)  # 融合后输出


class RepNCSPELAN4(nn.Module):
    """可重参数化的跨阶段部分ELAN网络"""
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            c3: 中间通道数1
            c4: 中间通道数2
            n: 瓶颈块数量
            bias: 是否使用偏置
            act: 激活函数类型
        """
        super().__init__()
        self.c = c3//2  # 分块大小
        # 初始1x1卷积
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        # 第一个CSP路径
        self.cv2 = nn.Sequential(
            CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4,3 , 1, bias=bias, act=act))
        # 第二个CSP路径
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        # 最终1x1卷积
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        """分块处理的前向传播"""
        y = list(self.cv1(x).chunk(2, 1))  # 将输入分成两部分
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])  # 对第二部分应用两个CSP路径
        return self.cv4(torch.cat(y, 1))  # 合并所有路径并输出

    def forward(self, x):
        """常规前向传播"""
        y = list(self.cv1(x).split((self.c, self.c), 1))  # 按指定大小分割输入
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])  # 对最后一部分应用两个CSP路径
        return self.cv4(torch.cat(y, 1))  # 合并所有路径并输出


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层，包含多头自注意力和前馈网络"""
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 mlp_blocks=None):
        """
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络维度
            dropout: dropout率
            activation: 激活函数类型
            normalize_before: 是否在之前进行归一化
        """
        super().__init__()
        self.normalize_before = normalize_before

        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.mlp_flag =False
        self.mlp_block=nn.ModuleList()
        if mlp_blocks in {'CGLU','EDFFN','FMFFN','DFFN','SEFN','ConvFFN','EFFN','ConvMlp','RMB'}:
            self.mlp_flag = True
            print("Ai缝合怪-带你冲顶会、顶刊--使用了mlp_blocks=", mlp_blocks)
            mlp_module = globals()[mlp_blocks]  # 从全局变量中获取
            self.mlp_block.append(
                nn.Sequential(mlp_module(d_model,dim_feedforward))
            )

        # 激活函数
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        """添加位置编码"""
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        """前向传播"""
        residual = src
        # 第一个归一化
        if self.normalize_before:
            src = self.norm1(src)
        # 添加位置编码
        q = k = self.with_pos_embed(src, pos_embed)
        # 自注意力计算
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

       #Transformer 自注意力变体改进



        # 残差连接和dropout
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        # 前馈网络部分
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        # print('src shape:',src.shape)
        if self.mlp_flag:
            # 使用新的改进MLP模块
            src = self.mlp_block[0](src)
        else:
            # 前馈网络计算
            src = self.linear2(self.dropout(self.activation(self.linear1(src))))
            # 残差连接和dropout
            src = residual + self.dropout2(src)

        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    """Transformer编码器，包含多个编码器层"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        Args:
            encoder_layer: 编码器层实例
            num_layers: 层数
            norm: 最终归一化层
        """
        super(TransformerEncoder, self).__init__()
        # 复制编码器层
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm  # 最终归一化层

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        """前向传播：逐层处理"""
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        # 最终归一化
        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder_AiGuai(nn.Module):
    """混合编码器，结合CNN和Transformer"""
    __share__ = ['eval_spatial_size', ]  # 共享属性

    def __init__(self,
                 in_channels=[512, 1024, 2048],  # 输入通道数列表
                 feat_strides=[8, 16, 32],  # 特征步长列表
                 hidden_dim=256,  # 隐藏层维度
                 nhead=8,  # Transformer头数
                 dim_feedforward=1024,  # 前馈网络维度
                 dropout=0.0,  # dropout率
                 enc_act='gelu',  # 编码器激活函数
                 use_encoder_idx=[2],  # 使用编码器的索引
                 num_encoder_layers=1,  # 编码器层数
                 pe_temperature=10000,  # 位置编码温度
                 expansion=1.0,  # 扩展因子
                 depth_mult=1.0,  # 深度乘数
                 act='silu',  # 激活函数
                 eval_spatial_size=None,  # 评估空间尺寸
                 version='dfine',  # 版本选择
                 fpn_pan_blocks=None , #自定义跨阶段增强增强提取网络层
                 downsample_blocks=None,#自定义下采样模块
                 upsample_blocks=None,#自定义上采样模块
                 fusion_blocks=None, #自定义特征融合
                 mlp_blocks=None, #自定义多层感知机模块
                 C3k_flag = False,
                 shortcut = True #比如C2f和C3k2模块中都涉及残差链接，默认是True启动，自己也可以手动关闭
                 ):
        super().__init__()
        # 初始化参数
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.version = version

        # 通道投影层
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            # 使用1x1卷积进行通道投影
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ]))
            self.input_proj.append(proj)

        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
            mlp_blocks = mlp_blocks
        )
        # 为每个使用编码器的特征层创建编码器
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # 自上而下的FPN路径
        self.lateral_convs = nn.ModuleList()  # 横向连接卷积
        self.fpn_blocks = nn.ModuleList()    # FPN融合块
        self.upsample_convs = nn.ModuleList()  #上采样卷积
        self.upsample_flag = False
        for _ in range(len(in_channels) - 1, 0, -1):  # 从深层到浅层
            # 根据版本选择不同的上采样方式
            if version == 'deim' and upsample_blocks in {'DySample_UP', 'EUCB', 'MEUM','CARAFE','LSB_up','SCEU'}:
                self.upsample_flag = True
                print("Ai缝合怪-带你冲顶会、顶刊--使用了upsample_blocks=", upsample_blocks)
                up_module = globals()[upsample_blocks]  # 从全局变量中获取
                self.upsample_convs.append(
                    nn.Sequential(up_module(hidden_dim, hidden_dim))
                )
            else:
               pass

        for _ in range(len(in_channels) - 1, 0, -1):  # 从深层到浅层
            # 根据版本选择不同的横向连接
            if version in {'dfine','deim'}:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))

            # 根据版本选择不同的FPN融合块
            if version == 'dfine':
                self.fpn_blocks.append(
                    RepNCSPELAN4(
                        hidden_dim * 2, hidden_dim, hidden_dim * 2,
                        round(expansion * hidden_dim // 2),
                        round(3 * depth_mult),
                        act=act
                    )
                )
            elif version == 'deim' and fpn_pan_blocks in {'C3k2', 'C3k2_MSGDC', 'C2f_MSCS', 'C3k2_ShuffleAttn','C3k2_ARConv', 'C3k2_CAS', 'C3k2_DBlock',
                                                          'C3k2_EBlock', 'C3k2_FADConv', 'C3k2_FDConv', 'C3k2_IEL','C3k2_LEG', 'C3k2_Mona',
                                                          'C3k2_DyTMona', 'C3k2_MoCAA', 'C3k2_PolaLinearAttention','C3k2_SAA', 'C3k2_SCGA', 'C3k2_SSA',
                                                          'C3k2_TSA', 'C3k2_TSSA','C3k2_VMMA','C3k2_RMB','C3k2_ConvAtt','C3k2_GSB','C3k2_HFP','C3k2_HMHA','C3k2_HMHABlock',
                                                          'C3k2_MALA','C3k2_MALABlock','C3k2_GCBAM','C3k2_RCSAB','C3k2_DSA','C3k2_wConv','C3k2_MBRConv3','C3k2_MBRConv5',
                                                          'C3k2_SFSConv','C3k2_SPConv','C3k2_CSAM','C3k2_CASAB','C3k2_MRFAConv','C3k2_MultiScaleAttention','C3k2_AAFMBlock','C3k2_AAFM',
                                                          'C3k2_PATConv','C3k2_StripConv','C3k2_StripBlock','C3k2_GCConv','C3k2_CGHalfConv','C3k2_DSPM','C3k2_HLKConv','C3k2_RCSSC',
                                                          'C3k2_LKLGL'}:
                print("Ai缝合怪-带你冲顶会、顶刊--使用了fpn_pan_blocks=",fpn_pan_blocks)
                module_blocks = globals()[fpn_pan_blocks]  # 从全局变量中获取
                self.fpn_blocks.append(
                    module_blocks(
                        hidden_dim * 2, hidden_dim,
                        round(3 * depth_mult), #代表 n 重复执行次数
                        c3k=C3k_flag,
                        shortcut=shortcut)
                )
            elif version == 'deim' and fpn_pan_blocks in {'C2f','C2f_MSGDC','C2f_MSCS','C2f_ShuffleAttn','C2f_ARConv','C2f_CAS','C2f_DBlock',
                                                          'C2f_EBlock','C2f_FADConv','C2f_FDConv','C2f_IEL','C2f_LEG','C2f_Mona',
                                                          'C2f_DyTMona','C2f_MoCAA','C2f_PolaLinearAttention','C2f_SAA','C2f_SCGA','C2f_SSA',
                                                          'C2f_TSA','C2f_TSSA','C2f_VMMA','C2f_RMB','C2f_ConvAtt','C2f_GSB','C2f_HFP','C2f_HMHA','C2f_HMHABlock',
                                                          'C2f_MALA','C2f_MALABlock','C2f_GCBAM','C2f_RCSAB','C2f_DSA','C2f_wConv','C2f_MBRConv3','C2f_MBRConv5',
                                                          'C2f_SFSConv','C2f_SPConv','C2f_CSAM','C2f_CASAB','C2f_MRFAConv','C2f_MultiScaleAttention','C2f_AAFMBlock','C2f_AAFM',
                                                          'C2f_PATConv','C2f_StripConv','C2f_StripBlock','C2f_GCConv','C2f_CGHalfConv','C2f_DSPM','C2f_HLKConv','C2f_RCSSC',
                                                          'C2f_LKLGL'}:
                print("Ai缝合怪-带你冲顶会、顶刊--使用了fpn_pan_blocks=",fpn_pan_blocks)
                module_blocks = globals()[fpn_pan_blocks]  # 从全局变量中获取
                self.fpn_blocks.append(
                    module_blocks(
                        hidden_dim * 2, hidden_dim,
                        round(3 * depth_mult),  # 代表 n 重复执行次数
                        shortcut=shortcut
                        )
                )
            else:
                self.fpn_blocks.append(
                    CSPLayer(
                        hidden_dim * 2, hidden_dim,
                        round(3 * depth_mult),
                        act=act,
                        expansion=expansion,
                        bottletype=VGGBlock
                    )
                )

        # 自下而上的PAN路径
        self.downsample_convs = nn.ModuleList()  # 下采样卷积
        self.pan_blocks = nn.ModuleList()        # PAN融合块
        for _ in range(len(in_channels) - 1):
            # 根据版本选择不同的下采样方式
            if version == 'dfine':
                self.downsample_convs.append(
                    nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act))
                )
            elif version == 'deim' and downsample_blocks in {'SPDConv','HWDown','WTFDown','DRFDown','ARConvDown','LSB_down'}:
                print("Ai缝合怪-带你冲顶会、顶刊--使用了downsample_blocks=", downsample_blocks)
                Down_module = globals()[downsample_blocks]  # 从全局变量中获取
                self.downsample_convs.append(
                    nn.Sequential(Down_module(hidden_dim, hidden_dim))
                )
            else:
                self.downsample_convs.append(
                    ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
                )
            # 根据版本选择不同的PAN融合块
            if version == 'dfine':
                self.pan_blocks.append(
                    RepNCSPELAN4(
                        hidden_dim * 2, hidden_dim, hidden_dim * 2,
                        round(expansion * hidden_dim // 2),
                        round(3 * depth_mult),
                        act=act
                    )
                )
            elif version == 'deim' and fpn_pan_blocks in {'C3k2','C3k2_MSGDC','C2f_MSCS','C3k2_ShuffleAttn','C3k2_ARConv','C3k2_CAS','C3k2_DBlock',
                                                          'C3k2_EBlock','C3k2_FADConv','C3k2_FDConv','C3k2_IEL','C3k2_LEG','C3k2_Mona',
                                                          'C3k2_DyTMona','C3k2_MoCAA','C3k2_PolaLinearAttention','C3k2_SAA','C3k2_SCGA','C3k2_SSA',
                                                          'C3k2_TSA','C3k2_TSSA','C3k2_VMMA','C3k2_RMB','C3k2_ConvAtt','C3k2_GSB','C3k2_HFP','C3k2_HMHA','C3k2_HMHABlock',
                                                          'C3k2_MALA','C3k2_MALABlock','C3k2_GCBAM','C3k2_RCSAB','C3k2_DSA','C3k2_wConv','C3k2_MBRConv3','C3k2_MBRConv5',
                                                          'C3k2_SFSConv','C3k2_SPConv','C3k2_CSAM','C3k2_CASAB','C3k2_MRFAConv','C3k2_MultiScaleAttention','C3k2_AAFMBlock','C3k2_AAFM',
                                                          'C3k2_PATConv','C3k2_StripConv','C3k2_StripBlock','C3k2_GCConv','C3k2_CGHalfConv','C3k2_DSPM','C3k2_HLKConv','C3k2_RCSSC',
                                                          'C3k2_LKLGL'}:
                print("Ai缝合怪-带你冲顶会、顶刊--使用了fpn_pan_blocks=",fpn_pan_blocks)
                module_blocks = globals()[fpn_pan_blocks]  # 从全局变量中获取
                self.pan_blocks.append(
                    module_blocks(
                        hidden_dim * 2, hidden_dim,
                        round(3 * depth_mult), #代表 n 重复执行次数
                        c3k=C3k_flag,
                        shortcut=shortcut)
                )
            elif version == 'deim' and fpn_pan_blocks in {'C2f','C2f_MSGDC','C2f_MSCS','C2f_ShuffleAttn','C2f_ARConv','C2f_CAS','C2f_DBlock',
                                                          'C2f_EBlock','C2f_FADConv','C2f_FDConv','C2f_IEL','C2f_LEG','C2f_Mona',
                                                          'C2f_DyTMona','C2f_MoCAA','C2f_PolaLinearAttention','C2f_SAA','C2f_SCGA','C2f_SSA',
                                                          'C2f_TSA','C2f_TSSA','C2f_VMMA','C2f_RMB','C2f_ConvAtt','C2f_GSB','C2f_HFP','C2f_HMHA','C2f_HMHABlock',
                                                          'C2f_MALA','C2f_MALABlock','C2f_GCBAM','C2f_RCSAB','C2f_DSA','C2f_wConv','C2f_MBRConv3','C2f_MBRConv5',
                                                          'C2f_SFSConv','C2f_SPConv','C2f_CSAM','C2f_CASAB','C2f_MRFAConv','C2f_MultiScaleAttention','C2f_AAFMBlock','C2f_AAFM',
                                                          'C2f_PATConv','C2f_StripConv','C2f_StripBlock','C2f_GCConv','C2f_CGHalfConv','C2f_DSPM','C2f_HLKConv','C2f_RCSSC',
                                                          'C2f_LKLGL'}:
                print("Ai缝合怪-带你冲顶会、顶刊--使用了fpn_pan_blocks=", fpn_pan_blocks)
                module_blocks = globals()[fpn_pan_blocks]  # 从全局变量中获取
                self.pan_blocks.append(
                    module_blocks(
                        hidden_dim * 2, hidden_dim,
                        round(3 * depth_mult),  # 代表 n 重复执行次数
                        shortcut=shortcut
                        )
                )
            else:
                self.pan_blocks.append(
                    CSPLayer(
                        hidden_dim * 2, hidden_dim,
                        round(3 * depth_mult),
                        act=act,
                        expansion=expansion,
                        bottletype=VGGBlock
                    )
                )

        #特征融合
        self.fusion_blocks = nn.ModuleList()  # PAN融合块
        self.fusion_flag = False
        if version == 'deim' and fusion_blocks in {'CGAFusion','CAFMFusion','MSPCA','GlobalFusionModule','MSAFM','FCModule','FeatureFusion','MFM',
                                                   'CFEM','SDPFusion','MSCAFusion','MSAM','RLAB_fusion','IIA_Fusion','DPCF','HFFE'} :
            self.fusion_flag = True
            print("Ai缝合怪-带你冲顶会、顶刊--使用了fusion_blocks=", fusion_blocks)

            fusion_module = globals()[fusion_blocks]  # 从全局变量中获取
            self.fusion_blocks.append(nn.Sequential(fusion_module(hidden_dim)))
            self.fusion_blocks.append(nn.Sequential(nn.Conv2d(hidden_dim,2*hidden_dim,1)))
        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        """重置参数，初始化位置编码"""
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                # 构建2D位置编码
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,  # 宽度
                    self.eval_spatial_size[0] // stride,  # 高度
                    self.hidden_dim,
                    self.pe_temperature)
                # 注册为缓冲区
                setattr(self, f'pos_embed{idx}', pos_embed)
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        构建2D正弦余弦位置编码
        Args:
            w: 宽度
            h: 高度
            embed_dim: 嵌入维度
            temperature: 温度参数
        Returns:
            位置编码张量
        """
        # 创建网格坐标
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

        # 检查嵌入维度是否可被4整除
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'

        pos_dim = embed_dim // 4  # 每个方向的位置编码维度
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)  # 频率计算

        # 计算水平和垂直方向的位置编码
        out_w = grid_w.flatten()[..., None] @ omega[None]  # 外积计算
        out_h = grid_h.flatten()[..., None] @ omega[None]

        # 合并四个分量形成最终位置编码
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        """前向传播"""
        assert len(feats) == len(self.in_channels), "输入特征数量与配置不匹配"

        # 1. 通道投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]  # 将输入的特征图通道数统一

        # 2. Transformer编码器处理
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]  # 获取高度和宽度
                # 展平特征图 [B, C, H, W] -> [B, H*W, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

                # 获取位置编码
                if self.training or self.eval_spatial_size is None:
                    # 训练时动态构建位置编码
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    # 评估时使用预计算的位置编码
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                # Transformer编码
                memory = self.encoder[i](src_flatten, src_mask=None, pos_embed=pos_embed)
                # 恢复形状 [B, H*W, C] -> [B, C, H, W]
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # 3. 自上而下的FPN路径
        inner_outs = [proj_feats[-1]]  # 从最深层的特征开始
        for idx in range(len(self.in_channels) - 1, 0, -1):  # 从深层到浅层
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            # 横向连接处理
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            # 上采样并拼接
            if self.upsample_flag:
                upsample_feat = self.upsample_convs[0](feat_heigh)
            else:
                upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')

            if self.fusion_flag:
                inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                    self.fusion_blocks[1](self.fusion_blocks[0]([upsample_feat, feat_low]))
                )
            else:
                inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                    torch.concat([upsample_feat, feat_low], dim=1)
                )
            inner_outs.insert(0, inner_out)  # 添加到列表开头

        # 4. 自下而上的PAN路径
        outs = [inner_outs[0]]  # 从最浅层的特征开始
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]

            # 下采样并拼接
            downsample_feat = self.downsample_convs[idx](feat_low)

            if self.fusion_flag:
                out = self.pan_blocks[idx](
                    self.fusion_blocks[1](self.fusion_blocks[0]([downsample_feat, feat_height]))
                )
            else:
                out = self.pan_blocks[idx](
                    torch.concat([downsample_feat, feat_height], dim=1)
                )
            outs.append(out)  # 添加到列表末尾

        return outs  # 返回多尺度特征
    # def forward(self, feats):
    #     """前向传播"""
    #     assert len(feats) == len(self.in_channels), "输入特征数量与配置不匹配"
    #
    #     # 1. 通道投影
    #     proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]  #将输入的特征图通道数统一
    #
    #     # 2. Transformer编码器处理
    #     if self.num_encoder_layers > 0:
    #         for i, enc_ind in enumerate(self.use_encoder_idx):
    #             h, w = proj_feats[enc_ind].shape[2:]
    #             # 展平特征图 [B, C, H, W] -> [B, H*W, C]
    #             src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
    #
    #             # 获取位置编码
    #             if self.training or self.eval_spatial_size is None:
    #                 # 训练时动态构建位置编码
    #                 pos_embed = self.build_2d_sincos_position_embedding(
    #                     w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
    #             else:
    #                 # 评估时使用预计算的位置编码
    #                 pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
    #
    #             # Transformer编码
    #             memory = self.encoder[i](src_flatten, src_mask=None, pos_embed=pos_embed)
    #             # 恢复形状 [B, H*W, C] -> [B, C, H, W]
    #             proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
    #
    #     # 3. 自上而下的FPN路径
    #     inner_outs = [proj_feats[-1]]  # 从最深层的特征开始
    #     for idx in range(len(self.in_channels) - 1, 0, -1):  # 从深层到浅层
    #         feat_heigh = inner_outs[0]
    #         feat_low = proj_feats[idx - 1]
    #
    #         # 横向连接处理
    #         feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
    #         inner_outs[0] = feat_heigh
    #
    #         # 上采样并拼接
    #         # upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
    #
    #         if self.upsample_flag:
    #             upsample_feat = self.upsample_convs[0](feat_heigh)
    #         else:
    #             upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
    #
    #         # inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](
    #         #     torch.concat([upsample_feat, feat_low], dim=1)
    #         # )
    #
    #         if self.fusion_flag:
    #             inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
    #                 self.fusion_blocks[1](self.fusion_blocks[0]([upsample_feat, feat_low]))
    #             )
    #         else:
    #             inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](
    #                 torch.concat([upsample_feat, feat_low], dim=1)
    #             )
    #         inner_outs.insert(0, inner_out)  # 添加到列表开头
    #
    #     # 4. 自下而上的PAN路径
    #     outs = [inner_outs[0]]  # 从最浅层的特征开始
    #     for idx in range(len(self.in_channels) - 1):
    #         feat_low = outs[-1]
    #         feat_height = inner_outs[idx + 1]
    #
    #         # 下采样并拼接
    #         downsample_feat = self.downsample_convs[idx](feat_low)
    #
    #
    #         # out = self.pan_blocks[idx](
    #         #     torch.concat([downsample_feat, feat_height], dim=1)
    #         # )
    #         if self.fusion_flag:
    #             out = self.pan_blocks[idx](
    #                 self.fusion_blocks[1](self.fusion_blocks[0]([downsample_feat, feat_height]))
    #             )
    #         else:
    #             out = self.pan_blocks[idx](
    #                 torch.concat([downsample_feat, feat_height], dim=1)
    #             )
    #         outs.append(out)  # 添加到列表末尾
    #
    #     return outs  # 返回多尺度特征



