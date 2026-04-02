"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from ..core import register

__all__ = ['DFINETransformer']



class MLP(nn.Module):
    """
    多层感知机模块，用于实现非线性映射
    输入维度 -> 隐藏层维度 -> 输出维度
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 创建线性层序列：input_dim -> hidden_dim -> ... -> hidden_dim -> output_dim
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)  # 获取激活函数

    def forward(self, x):
        # 前向传播，除最后一层外应用激活函数
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    """
    多尺度可变形注意力模块，自适应地从不同尺度特征图上采样关键点
    """

    def __init__(
            self,
            embed_dim=256,  # 嵌入维度
            num_heads=8,  # 注意力头数
            num_levels=4,  # 特征层级数
            num_points=4,  # 每个层级的采样点数
            method='default',  # 采样方法
            offset_scale=0.5,  # 偏移量缩放因子
    ):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        # 处理每个层级的采样点数量
        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        # 注册采样点缩放因子为缓冲区
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim必须能被num_heads整除"

        # 线性层预测采样偏移量和注意力权重
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        # 绑定可变形注意力核心函数
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()  # 初始化参数

        # 如果使用离散方法，固定采样偏移量参数
        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # 初始化采样偏移量参数
        init.constant_(self.sampling_offsets.weight, 0)
        # 为每个注意力头设置初始方向
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # 初始化注意力权重参数
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(self,
                query: torch.Tensor,  # 查询特征 [bs, query_length, C]
                reference_points: torch.Tensor,  # 参考点 [bs, query_length, n_levels, 2/4]
                value: torch.Tensor,  # 键值特征 [bs, value_length, C]
                value_spatial_shapes: List[int]):  # 每个层级的空间形状 [(H_0, W_0), ...]
        """
        计算多尺度可变形注意力
        """
        bs, Len_q = query.shape[:2]

        # 计算采样偏移量和注意力权重
        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)  # 对注意力权重进行softmax归一化

        # 根据参考点计算采样位置
        if reference_points.shape[-1] == 2:  # 参考点为2D坐标(x,y)
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1,2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:  # 参考点为4D坐标(x,y,w,h)
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "reference_points的最后一维必须是2或4，但得到{}".format(reference_points.shape[-1]))

        # 调用核心函数计算可变形注意力
        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights,
                                              self.num_points_list)

        return output

class Gate(nn.Module):
    """
    门控机制，用于自适应融合两个输入张量
    """

    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)  # 使用特定概率初始化偏置
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        # 计算门控值并融合输入
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)

class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层，包含自注意力、可变形交叉注意力和前馈网络
    """

    def __init__(self,
                 d_model=256,  # 模型维度
                 n_head=8,  # 注意力头数
                 dim_feedforward=1024,  # 前馈网络隐藏维度
                 dropout=0.,  # dropout率
                 activation='relu',  # 激活函数
                 n_levels=4,  # 特征层级数
                 n_points=4,  # 每个层级的采样点数
                 cross_attn_method='default',  # 交叉注意力方法
                 layer_scale=None

                 ):  # 层缩放因子
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 交叉注意力模块，使用多尺度可变形注意力
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)

        self.dropout2 = nn.Dropout(dropout)

        # 门控机制，用于融合不同来源的特征
        self.gateway = Gate(d_model)

        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()  # 初始化参数

    def _reset_parameters(self):
        # 使用Xavier初始化线性层权重
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        # 将位置编码添加到输入张量
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        # 前馈网络的前向传播
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,  # 目标特征
                reference_points,  # 参考点
                value,  # 键值特征
                spatial_shapes,  # 空间形状
                attn_mask=None,  # 注意力掩码
                query_pos_embed=None):  # 查询位置编码
        """
        Transformer解码器层的前向传播
        """
        # 自注意力计算
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)  # 残差连接
        # print("target!!!",target.shape)
        target = self.norm1(target)  # 层归一化

        # 交叉注意力计算，使用多尺度可变形注意力

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        # 通过门控机制融合特征
        target = self.gateway(target, self.dropout2(target2))

        # 前馈网络
        target2 = self.forward_ffn(target)


        target = target + self.dropout4(target2)  # 残差连接
        target = self.norm3(target.clamp(min=-65504, max=65504))  # 层归一化并限制范围

        return target




class Integral(nn.Module):
    """
    积分层，用于将分布转换为连续值
    计算方式：`sum{Pr(n) * W(n)}`，其中Pr(n)是概率分布，W(n)是权重函数
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)  # 将输入转换为概率分布
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)  # 通过线性投影转换为连续值
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    """
    定位质量评估模块，用于评估检测框的定位质量
    通过分析预测角点的分布特征来预测质量分数
    """

    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        # 使用MLP预测定位质量分数
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        # 计算预测角点的概率分布
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        # 获取top-k概率值
        prob_topk, _ = prob.topk(self.k, dim=-1)
        # 结合top-k概率和均值作为统计特征
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        # 预测定位质量分数并与分类分数融合
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class TransformerDecoder(nn.Module):
    """
    Transformer解码器，实现细粒度分布细化(FDR)
    通过多层迭代更新来优化目标检测预测
    """

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu'):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max

        # 创建解码器层列表，前eval_idx+1层使用标准层，其余使用宽层
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                                    + [copy.deepcopy(decoder_layer_wide) for _ in
                                       range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        预处理MSDeformableAttention的输入值
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        """
        转换为部署模式，移除不必要的层
        """
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,  # 目标特征
                ref_points_unact,  # 未激活的参考点
                memory,  # 编码器记忆
                spatial_shapes,  # 空间形状
                bbox_head,  # 边界框预测头
                score_head,  # 分数预测头
                query_pos_head,  # 查询位置编码头
                pre_bbox_head,  # 预边界框预测头
                integral,  # 积分层
                up,  # 上采样因子
                reg_scale,  # 回归缩放因子
                attn_mask=None,  # 注意力掩码
                memory_mask=None,  # 记忆掩码
                dn_meta=None):  # 去噪元数据
        """
        Transformer解码器的前向传播
        """
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []

        # 权重函数，用于将分布转换为连续值
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # 对于特定层调整尺度
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            # 通过解码器层处理
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                # 初始边界框预测，使用逆sigmoid函数细化
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # 细化边界框角点，整合前一层的修正
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            # 训练模式或特定层保存输出
            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # LQE评估定位质量，提升检测框准确性
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            # 更新状态用于下一层
            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
            torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores


@register()
class DFINETransformer(nn.Module):
    """
    D-FINE Transformer: 基于改进匹配机制的DETR变体，实现快速收敛和高精度目标检测
    """
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,  # 类别数
                 hidden_dim=256,  # 隐藏维度
                 num_queries=300,  # 查询向量数量
                 feat_channels=[512, 1024, 2048],  # 特征通道数
                 feat_strides=[8, 16, 32],  # 特征步长
                 num_levels=3,  # 特征层级数
                 num_points=4,  # 每个层级的采样点数
                 nhead=8,  # 注意力头数
                 num_layers=6,  # 解码器层数
                 dim_feedforward=1024,  # 前馈网络维度
                 dropout=0.,  # dropout率
                 activation="relu",  # 激活函数
                 num_denoising=100,  # 去噪训练样本数
                 label_noise_ratio=0.5,  # 标签噪声比率
                 box_noise_scale=1.0,  # 边界框噪声尺度
                 learn_query_content=False,  # 是否学习查询内容
                 eval_spatial_size=None,  # 评估时的空间大小
                 eval_idx=-1,  # 评估层索引
                 eps=1e-2,  # 小常数，防止数值不稳定
                 aux_loss=True,  # 是否使用辅助损失
                 cross_attn_method='default',  # 交叉注意力方法
                 query_select_method='default',  # 查询选择方法
                 reg_max=32,  # 回归分布的最大值
                 reg_scale=4.,  # 回归缩放因子
                 layer_scale=1,  # 层缩放因子
                 mlp_act='relu',  # MLP激活函数
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        # 扩展特征步长列表
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # 构建输入投影层
        self._build_input_proj_layer(feat_channels)

        # Transformer模块
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        # 创建解码器层
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
                                                activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
                                                     activation, num_levels, num_points,
                                                     cross_attn_method=cross_attn_method, layer_scale=layer_scale)


        # 创建解码器
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)

        # 去噪训练相关
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # 解码器嵌入
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)

        # 编码器输出处理
        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim, )),
        ]))

        # 编码器分数和边界框预测头
        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)  # 类别无关分数预测
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)  # 类别相关分数预测

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)  # 边界框预测

        # 解码器预测头
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
            + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in
               range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)  # 积分层，用于将分布转换为连续值

        # 初始化编码器输出锚点和有效掩码
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        """
        转换为部署模式，移除不必要的层
        """
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        """
        初始化模型参数
        """
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        # 初始化解码器预测头
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

        # 初始化输入投影层
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

#修改2
    def _build_input_proj_layer(self, feat_channels):
        """
        构建输入特征投影层，将不同通道的特征映射到统一维度
        """
        self.input_proj = nn.ModuleList()  # 用于存储输入投影层
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([  # 定义1x1卷积和批归一化
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))
                    ]))
                )

        # 扩展特征层级，如果特征通道数不匹配
        in_channels = feat_channels[-1]  # 获取最后一层的通道数
        for _ in range(self.num_levels - len(feat_channels)):  # 如果特征层级数不匹配，扩展
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([  # 定义3x3卷积和批归一化
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))
                    ]))
                )
                in_channels = self.hidden_dim
    def _get_encoder_input(self, feats: List[torch.Tensor]):
        """
        获取编码器输入，处理多尺度特征
        """
        # 调试信息：检查 feats 和 input_proj 的长度
        # print("Length of feats:", len(feats))
        # print("Length of input_proj:", len(self.input_proj))
        # 特征投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # 特征展平
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # 拼接所有层级的特征
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        """
        生成锚点（参考点）
        """
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))  # 转换为对数空间
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):
        """
        获取解码器输入，包括初始化查询和参考点
        """
        # 准备锚点和有效掩码
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # 应用有效掩码
        memory = valid_mask.to(memory.dtype) * memory

        # 处理编码器输出
        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        # 选择top-k查询
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
            self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)

        # 预测边界框
        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # 初始化查询内容
        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        # 如果有去噪训练数据，拼接上去
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor,
                     topk: int):
        """
        从编码器输出中选择top-k个查询
        """
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        # 根据索引选择top-k元素
        topk_anchors = outputs_anchors_unact.gather(dim=1, \
                                                    index=topk_ind.unsqueeze(-1).repeat(1, 1,
                                                                                        outputs_anchors_unact.shape[
                                                                                            -1]))

        topk_logits = outputs_logits.gather(dim=1, \
                                            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[
                                                -1])) if self.training else None

        topk_memory = memory.gather(dim=1, \
                                    index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        """
        模型的前向传播
        """
        # 输入投影和特征嵌入
        memory, spatial_shapes = self._get_encoder_input(feats)

        # 准备去噪训练数据
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                                                         self.num_classes,
                                                         self.num_queries,
                                                         self.denoising_class_embed,
                                                         num_denoising=self.num_denoising,
                                                         label_noise_ratio=self.label_noise_ratio,
                                                         box_noise_scale=1.0,
                                                         )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        # 获取解码器输入
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # 解码器前向传播
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta)

        # 处理去噪训练输出
        if self.training and dn_meta is not None:
            # 分离去噪训练和正常训练的输出
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)

            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)

            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)

        # 构建输出字典
        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        # 添加辅助损失输出
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                     out_corners[-1], out_logits[-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            # 添加去噪训练输出
            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                                                        dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """
        设置辅助损失的输出格式
        """
        # 这是一个解决torchscript兼容性问题的方法
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None):
        """
        设置更复杂的辅助损失输出格式
        """
        # 这是一个解决torchscript兼容性问题的方法
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                 'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]


