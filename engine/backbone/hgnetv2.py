
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from engine.backbone.common import FrozenBatchNorm2d  # 导入冻结BN层
from engine.core import register  # 注册模块
import logging  # 日志记录

from engine.newaddmodules.add_attention import *
from engine.newaddmodules.add_Conv import *

# 初始化函数别名
kaiming_normal_ = nn.init.kaiming_normal_  #  kaiming正态分布初始化
zeros_ = nn.init.zeros_  # 全零初始化
ones_ = nn.init.ones_  # 全一初始化

__all__ = ['HGNetv2']  # 导出HGNetv2类

class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,  # 缩放因子初始值
            bias_value=0.0    # 偏置初始值
    ):
        super().__init__()
        # 定义可学习的缩放因子和偏置
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias  # 仿射变换：y = scale * x + bias

class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,  # 输入通道数
            out_chs,  # 输出通道数
            kernel_size,  # 卷积核大小
            stride=1,  # 步长
            groups=1,  # 分组卷积组数
            padding='',  # 填充方式（'same'或数值）
            use_act=True,  # 是否使用激活函数
            use_lab=False  # 是否使用可学习仿射块
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab

        # 处理填充方式
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),  # 左右和上下各填充1（针对kernel_size=3）
                nn.Conv2d(
                    in_chs, out_chs, kernel_size, stride,
                    groups=groups, bias=False  # 卷积层不使用偏置（BN层已包含）
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs, out_chs, kernel_size, stride,
                padding=(kernel_size - 1) // 2,  # 计算对称填充
                groups=groups, bias=False
            )

        self.bn = nn.BatchNorm2d(out_chs)  # 批量归一化层
        self.act = nn.ReLU() if use_act else nn.Identity()  # 激活函数（可选ReLU或恒等）
        self.lab = LearnableAffineBlock() if (use_act and use_lab) else nn.Identity()  # 可选仿射块

    def forward(self, x):
        x = self.conv(x)  # 卷积
        x = self.bn(x)  # 批量归一化
        x = self.act(x)  # 激活函数
        x = self.lab(x)  # 仿射变换
        return x

class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,       # 输入通道数
            out_chs,      # 输出通道数
            kernel_size,  # 深度卷积核大小
            groups=1,     # 分组数（默认1，深度卷积时为out_chs）
            use_lab=False # 是否使用仿射块
    ):
        super().__init__()
        # 逐点卷积（1x1卷积）降维
        self.conv1 = ConvBNAct(
            in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab
        )
        # 深度卷积（分组卷积）提取空间特征
        self.conv2 = ConvBNAct(
            out_chs, out_chs, kernel_size=kernel_size, groups=out_chs,  # groups=out_chs为深度卷积
            use_act=True, use_lab=use_lab
        )

    def forward(self, x):
        x = self.conv1(x)  # 1x1卷积降维
        x = self.conv2(x)  # 深度卷积+BN+激活+仿射
        return x


class StemBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        # 第一层卷积：3x3步长2，下采样
        self.stem1 = ConvBNAct(
            in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab
        )
        # 分支a：2x2卷积（步长1，无下采样）
        self.stem2a = ConvBNAct(
            mid_chs, mid_chs//2, kernel_size=2, stride=1, use_lab=use_lab
        )
        # 分支b：2x2卷积（步长1，恢复通道数）
        self.stem2b = ConvBNAct(
            mid_chs//2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab
        )
        # 第三层卷积：3x3步长2，下采样并融合分支
        self.stem3 = ConvBNAct(
            mid_chs*2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab
        )
        # 1x1卷积调整通道数
        self.stem4 = ConvBNAct(
            mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)  # 最大池化（保留尺寸）

    def forward(self, x):
        x = self.stem1(x)                # 第一层卷积，尺寸减半
        x = F.pad(x, (0, 1, 0, 1))       # 填充右侧和下侧，保持尺寸一致
        x2 = self.stem2a(x)              # 分支a：通道减半
        x2 = F.pad(x2, (0, 1, 0, 1))     # 填充
        x2 = self.stem2b(x2)             # 分支b：通道恢复
        x1 = self.pool(x)                # 池化分支x，保持尺寸
        x = torch.cat([x1, x2], dim=1)   # 拼接两个分支（通道数翻倍）
        x = self.stem3(x)                # 卷积下采样，通道恢复
        x = self.stem4(x)                # 调整通道数
        return x

class EseModule(nn.Module):
    def __init__(self, chs):  # chs: 输入通道数
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1)  # 1x1卷积生成注意力权重
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活生成0-1权重

    def forward(self, x):
        identity = x  # 保存输入特征
        # 全局平均池化：压缩空间维度，保留通道维度
        x = x.mean((2, 3), keepdim=True)  # 输出形状：(B, C, 1, 1)
        x = self.conv(x)                  # 1x1卷积生成权重
        x = self.sigmoid(x)               # 归一化为注意力权重
        return torch.mul(identity, x)     # 权重与输入特征相乘，增强关键区域
class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,  # 输入通道数
            mid_chs,  # 中间通道数（瓶颈通道）
            out_chs,  # 输出通道数
            layer_num,  # 卷积层数
            kernel_size=3,  # 卷积核大小
            residual=False,  # 是否使用残差连接
            light_block=False,  # 是否使用轻量级卷积（LightConvBNAct）
            use_lab=False,  # 是否使用仿射块
            agg='ese',  # 特征聚合方式（'se'或'ese'）
            aggstartlayer= 2,
            addconv=None, #自定义卷积
            drop_path=0.,  # 随机失活率（用于正则化）
    ):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()

        # 构建卷积层序列
        for i in range(layer_num):
            if light_block:
                # 使用轻量级卷积块（1x1+深度卷积）
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,  # 第一层用输入通道，其余用中间通道
                        mid_chs, kernel_size, use_lab=use_lab
                    )
                )
            elif addconv in {'HSLConv'}:

                module_conv = globals()[addconv]  # 从全局变量中获取
                if addconv in {'FADConv'}:
                    self.layers.append(
                        module_conv(in_channels=in_chs if i == 0 else mid_chs, out_channels= mid_chs, kernel_size=3, stride=1 )
                    )
                else:
                    self.layers.append(
                        module_conv(
                            in_chs if i == 0 else mid_chs,  # 第一层用输入通道，其余用中间通道
                            mid_chs
                        )
                    )
            else:
                # 使用标准卷积块（ConvBNAct）
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs, kernel_size, use_lab=use_lab
                    )
                )
        # 特征聚合模块（SE或ESE）
        total_chs = in_chs + layer_num * mid_chs  # 输入通道+所有中间层通道之和
        if agg == 'se':
            # SE模块：压缩-激励机制
            self.aggregation = nn.Sequential(
                ConvBNAct(total_chs, out_chs // 2, kernel_size=1, use_lab=use_lab),
                ConvBNAct(out_chs // 2, out_chs, kernel_size=1, use_lab=use_lab),
            )

        elif agg in {'MSDAM'}:

            module_agg = globals()[agg]  # 从全局变量中获取
            self.aggregation = nn.Sequential(
                    ConvBNAct(total_chs, out_chs, kernel_size=1, use_lab=use_lab),
                    module_agg(out_chs),
                   )
        else:
            # ESE模块：增强空间注意力
            self.aggregation = nn.Sequential(
                ConvBNAct(total_chs, out_chs, kernel_size=1, use_lab=use_lab),
                EseModule(out_chs),  # 应用空间注意力
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()  # 随机失活

    def forward(self, x):
        identity = x  # 保存残差连接的输入
        output = [x]  # 保存所有层的输出（包括输入）

        # 逐层卷积，收集特征
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)  # 拼接所有层的特征图（通道数累加）
        x = self.aggregation(x)  # 特征聚合与注意力机制

        if self.residual:
            x = self.drop_path(x) + identity  # 残差连接+随机失活
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,  # 输入通道数
            mid_chs,  # 中间通道数
            out_chs,  # 输出通道数
            block_num,  # 块数
            layer_num,  # 每块的卷积层数
            downsample=True,  # 是否下采样（阶段起始是否降维）
            light_block=False,  # 是否使用轻量级块
            kernel_size=3,  # 卷积核大小
            use_lab=False,  # 是否使用仿射块
            agg='se',  # 聚合方式
            addconv=None,  # 自定义卷积
            drop_path=0.,  # 随机失活率（可为列表，每块独立设置）
    ):
        super().__init__()
        self.downsample = downsample

        # 下采样模块（3x3深度卷积，步长2）
        if downsample:
            self.downsample = ConvBNAct(
                in_chs, in_chs, kernel_size=3, stride=2, groups=in_chs, use_act=False, use_lab=use_lab
            )
        else:
            self.downsample = nn.Identity()  # 不下采样时直接通过

        # 构建多个HG_Block
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,  # 第一块输入为in_chs，后续为out_chs（残差连接）
                    mid_chs, out_chs, layer_num,
                    kernel_size=kernel_size,
                    residual=False if i == 0 else True,  # 第一块无残差，后续块有
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    addconv=addconv,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)  # 组合块为序列

    def forward(self, x):
        x = self.downsample(x)  # 下采样（若需要）
        x = self.blocks(x)  # 通过多个HG_Block
        return x



@register()
class HGNetv2(nn.Module):

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],  # Stem块输入/中间/输出通道
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3], # 输入通道, 中间通道, 输出通道, 块数, 是否下采样, 是否轻量块, 卷积核大小, 层数
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 agg= 'ese',
                 addconv=None,  # 自定义卷积
                 local_model_dir='weight/hgnetv2/'):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab)

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num \
                = stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    agg,
                    addconv=addconv))

        # 冻结参数
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)  # 冻结stem
            if not freeze_stem_only:
                # 冻结指定阶段
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        # 冻结归一化层
        if freeze_norm:
            self._freeze_norm(self)

        # 加载预训练权重
        pretrained = pretrained
        print('加载预训练权重:',pretrained)

        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                model_path = os.path.join(local_model_dir, f'PPHGNetV2_{name}_stage1.pth')
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        if torch.distributed.get_rank() == 0:
                            print(GREEN + "Downloading pretrained HGNetV2..." + RESET)
                            state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu',
                                                                       model_dir=local_model_dir)
                            torch.distributed.barrier()
                        else:
                            torch.distributed.barrier()
                            state = torch.load(model_path, map_location='cpu')
                    else:
                        # 非分布式情况下直接下载
                        print(GREEN + "Downloading pretrained HGNetV2..." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu',model_dir=local_model_dir)

                # 🔥加载部分预训练参数（仅 stem 和 stage1）
                self.load_pretrained_filtered(state, keep_keys=['stem.*', 'stages.0.*'])

            except Exception as e:
                print(f"{str(e)}")
                logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                logging.error(GREEN + f"Please manually download from {download_url} to {local_model_dir}" + RESET)

    def load_pretrained_filtered(self,
                                 state_dict,
                                 keep_keys=['stem.*', 'stages.0.*'],
                                 verbose=False):
        import re

        # Step 1: 只保留匹配的 key
        keys_to_keep = []
        for k in list(state_dict.keys()):
            if any(re.match(pat, k) for pat in keep_keys):
                keys_to_keep.append(k)

        filtered_state = {}
        model_dict = self.state_dict()  # 这里改成self.state_dict()，拿当前模型参数字典

        for k in keys_to_keep:
            if k in model_dict and state_dict[k].shape == model_dict[k].shape:
                filtered_state[k] = state_dict[k]
            else:
                if verbose:
                    print(f"❌ Skip loading: {k}")
                    if k in model_dict:
                        print(f"   Shape mismatch: expected {model_dict[k].shape}, got {state_dict[k].shape}")
                    else:
                        print(f"   Key not found in model")

        # Step 2: 加载参数，必须self调用
        for key in filtered_state.keys():
            print(key)

        self.load_state_dict(filtered_state, strict=False)
        print(f"🚀 Loaded {len(filtered_state)} matched parameters into model.")
    def _freeze_norm(self, m: nn.Module):
        """递归冻结BN层，替换为FrozenBatchNorm2d"""
        for name, child in m.named_children():
            if isinstance(child, nn.BatchNorm2d):
                # 替换为冻结BN层
                frozen_bn = FrozenBatchNorm2d(child.num_features)
                frozen_bn.load_state_dict(child.state_dict())
                setattr(m, name, frozen_bn)
            else:
                self._freeze_norm(child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        """冻结模块的所有参数"""
        for p in m.parameters():
            p.requires_grad = False
    def forward(self, x):
        """前向传播"""
        x = self.stem(x)  # 通过Stem块，尺寸缩小4倍
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)  # 通过各个阶段，每个阶段可能下采样
            if idx in self.return_idx:  # 收集指定阶段的输出
                outs.append(x)
        return outs  # 返回多阶段特征图（如stage1-stage3的输出）


