"""
Reference: https://github.com/google-research/deeplab2/blob/main/model/transformer_decoder/kmax.py
"""

from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

"DropPath?"
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_tf_ as trunc_normal_

import matplotlib.pyplot as plt
import math

# from models.pcapmap import pcapmap_cluster

import numpy as np

# TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
# TRANSFORMER_DECODER_REGISTRY.__doc__ = """
# Registry for transformer module.
# """
#
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:1" if USE_CUDA else "cpu")

class Flatten(nn.Module):
  def forward(self, x):
    return x.reshape(x.shape[0], -1)

class UnFlatten(nn.Module):
  def forward(self, x,c,h,w):
    return x.reshape(x.shape[0], c,h,w)

def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()

def leaky_relu(p=0.2):
  return nn.LeakyReLU(p, inplace=True)

def get_norm(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm=None, act=None,
                 conv_type='2d', conv_init='he_normal', norm_init=1.0):
        super().__init__()

        if conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels)
        # nn.init.constant_(self.norm.weight, norm_init)
        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)


        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.norm(out1)
        out3 = self.act(out2)
        return out3

# def build_transformer_decoder(cfg, input_shape_from_backbone):
#     """
#     Build a instance embedding branch from `cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NAME`.
#     """
#     name = cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NAME
#     return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, input_shape_from_backbone)
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv1d(in_channels=n, out_channels=k,kernel_size=1)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        ""
        "8*3*768*1"
        # b,c,h,w = x.shape
        # x_ = x.reshape(b,-1)
        x_ = x
        for i, layer in enumerate(self.layers):
            x_ = F.relu(layer(x_)) if i < self.num_layers - 1 else layer(x_)
        # x_ = x_.reshape(b,c,h,w)
        return x_

# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/decoder/max_deeplab.py#L60
def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
        (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L41
class AttentionOperation(nn.Module):
    def __init__(self, channels_v, num_heads):
        super().__init__()

        self._batch_norm_similarity = nn.BatchNorm2d(num_heads, eps=1e-3, momentum=0.01)
        self._batch_norm_retrieved_value = nn.BatchNorm1d(channels_v,eps=1e-3, momentum=0.01)

        # self._batch_norm_similarity = get_norm('syncbn', num_heads)
        # self._batch_norm_retrieved_value = get_norm('syncbn', channels_v)

    def forward(self, query, key, value):
        "主函数"
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('bhdl,bhdm->bhlm', query, key)
        similarity_logits = self._batch_norm_similarity(similarity_logits)

        with autocast(enabled=False):
            "weight map between objection Q and feature K"
            attention_weights = F.softmax(similarity_logits.float(), dim=-1)
        "reweight feature according to the dot similarity between Q and K" \
        "PLUS: Q is related with the cluster center C"
        retrieved_value = torch.einsum(
            'bhlm,bhdm->bhdl', attention_weights, value)
        "N:batchsize "
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = self._batch_norm_retrieved_value(
            retrieved_value)
        retrieved_value = F.gelu(retrieved_value)
        return retrieved_value


# https://github.com/google-research/deeplab2/blob/main/model/kmax_deeplab.py#L32
class kMaXPredictor(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=3):
        super().__init__()

        self.pixel_space_head_conv0bnact = ConvBN(256, 256, kernel_size=5,
                                                   groups=256, padding=2, bias=False,
                                                   norm=None, act='gelu', conv_init='xavier_uniform')
        self.pixel_space_head_conv1bnact = ConvBN(256, 256, kernel_size=1, bias=False, norm=None,
                                                   act='gelu')
        self.pixel_space_head_last_convbn = ConvBN(256, in_channel_pixel, kernel_size=1, bias=True, norm=None, act=None)
        trunc_normal_(self.pixel_space_head_last_convbn.conv.weight, std=0.01)

        self.transformer_mask_head = ConvBN(256, in_channel_pixel, kernel_size=1, bias=False, norm=None, act=None,
                                             conv_type='1d')
        self.transformer_class_head = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self.transformer_class_head.conv.weight, std=0.01)

        self.pixel_space_mask_batch_norm = get_norm(None, channels=1)
        # nn.init.constant_(self.pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        # mask_embeddings/class_embeddings: B x C x N
        # pixel feature: B x C x H x W
        pixel_space_feature = self.pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self.pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self.pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self.transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self.transformer_mask_head(mask_embeddings)
        "pixel_space_normalized_feature: 8,128,32,32; cluster_mask_kernel:8,128,1024"
        "bchw,bcn->bnhw"
        mask_logits = torch.einsum('bchw,bcn->bnhw',
                                   pixel_space_normalized_feature, cluster_mask_kernel)

        mask_logits = self.pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)

        # class_labels = torch.argmax(mask_logits, dim=1)
        #
        # # 创建一个简单的颜色映射
        # cmap = plt.get_cmap('tab10')  # 'tab10' 是一个常用的类别色彩图
        # batch_idx = 0  # 选择第一个样本进行可视化
        #
        # # 将类别标签映射到对应颜色
        # class_labels_np = class_labels[batch_idx].cpu().numpy()
        # colored_image = cmap(class_labels_np / np.max(class_labels_np))  # 归一化到 0-1
        #
        # # 绘制结果
        # plt.imshow(colored_image)
        # plt.axis('off')
        # plt.title('Class Prediction Visualization')
        # plt.show()

        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature}


class CT_conv1d(nn.Module):
    def __init__(
            self,in_channel_query=768):
        super().__init__()
        self.ct_conv1d_1 = Dynamic_conv1d(in_planes=in_channel_query, out_planes=in_channel_query * 2,
                                          kernel_size=3, padding=1)
        self.ct_conv1d_2 = Dynamic_conv1d(in_planes=in_channel_query * 2, out_planes=in_channel_query * 2,
                                          kernel_size=3, padding=1)
        self.ct_conv1d_3 = Dynamic_conv1d(in_planes=in_channel_query * 2, out_planes=in_channel_query * 2,
                                          kernel_size=3, padding=1)
        self.ct_conv1d_4 = Dynamic_conv1d(in_planes=in_channel_query * 2, out_planes=in_channel_query,
                                          kernel_size=3, padding=1)
        self.ct_conv1d_5 = Dynamic_conv1d(in_planes=in_channel_query, out_planes=in_channel_query,
                                          kernel_size=3, padding=1)
    def forward(self,feature,ct):
        out1 = F.relu(self.ct_conv1d_1(feature,ct))
        out2 = F.relu(self.ct_conv1d_2(out1, ct))
        out3 = F.relu(self.ct_conv1d_3(out2, ct))
        out4 = F.relu(self.ct_conv1d_4(out3, ct))
        out5 = self.ct_conv1d_5(out4, ct)
        return out5


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L107
class kMaXTransformerLayer(nn.Module):
    def __init__(
            self,
            num_classes=1536,
            in_channel_pixel=512,
            in_channel_query=3,
            base_filters=128,
            num_heads=8,
            bottleneck_expansion=2,
            key_expansion=1,
            value_expansion=2,
            drop_path_prob=0.0,
    ):
        super().__init__()

        self._num_classes = num_classes
        self._num_heads = num_heads
        self._in_channel_pixel = in_channel_pixel
        self._in_channel_query = in_channel_query
        "128*2"
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        "128*1"
        self._total_key_depth = int(round(base_filters * key_expansion))
        "128*2=256"
        self._total_value_depth = int(round(base_filters * value_expansion))

        # Per tf2 implementation, the same drop path prob are applied to:
        # 1. k-means update for object query
        # 2. self/cross-attetion for object query
        # 3. ffn for object query
        "DropPath"
        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        initialization_std = self._bottleneck_channels ** -0.5

        "MLP"
        # self.mlp_center = MLP(input_dim=3, hidden_dim=24, output_dim=3, num_layers=8)

        # self.mlp_q = MLP(input_dim=self._in_channel_query, hidden_dim=self._in_channel_query*4, output_dim=self._in_channel_query, num_layers=3)
        # self.mlp_k = MLP(input_dim=self._in_channel_query, hidden_dim=self._in_channel_query*4, output_dim=self._in_channel_query, num_layers=3)
        # self.mlp_v = MLP(input_dim=self._in_channel_query, hidden_dim=self._in_channel_query*4, output_dim=self._in_channel_query, num_layers=3)

        "256→256"
        # self.query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
        #                                   norm='syncbn', act='gelu', conv_type='1d')

        self.query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                         norm=None, act='gelu', conv_type='1d')

        "2048→256"
        # self.pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
        #                                   norm='none', act='gelu')

        self.pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                         norm=None, act='gelu')

        "256→128+256"
        # self.query_qkv_conv_bn = MLP(input_dim=self._num_classes,hidden_dim=self._in_channel_query*3,
        #                               output_dim=self._in_channel_query*3,num_layers=5)
        self.query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth,
                                         kernel_size=1, bias=False,
                                         norm=None, act=None, conv_type='1d')
        trunc_normal_(self.query_qkv_conv_bn.conv.weight, std=initialization_std)

        "256→256"
        self.pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                       norm=None, act=None)
        trunc_normal_(self.pixel_v_conv_bn.conv.weight, std=initialization_std)

        "attention: "
        self.query_self_attention = AttentionOperation(channels_v=self._bottleneck_channels, num_heads=num_heads)

        "256→256"
        # self.query_conv3_bn = MLP(input_dim=self._in_channel_query,hidden_dim=self._in_channel_query,
        #                            output_dim=self._in_channel_query,num_layers=3)
        self.query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                      norm=None, act=None, conv_type='1d', norm_init=0.0)

        "256→2048"
        self.query_fnn = MLP(input_dim=self._in_channel_query,hidden_dim=self._in_channel_query*4,
                              output_dim=self._in_channel_query,num_layers=3)

        self.query_ffn_conv1_bn_act = ConvBN(self._in_channel_query, 1024, kernel_size=1, bias=False,
                                              norm=None, act='gelu', conv_type='1d')
        "2048→256"
        self.query_ffn_conv2_bn = ConvBN(1024, self._in_channel_query, kernel_size=1, bias=False,
                                          norm=None, act=None, conv_type='1d', norm_init=0.0)

        self.predcitor = kMaXPredictor(in_channel_pixel=self._in_channel_pixel,
                                        in_channel_query=self._in_channel_query, num_classes=self._in_channel_query)


        self.kmeans_query_batch_norm_retrieved_value = nn.BatchNorm1d(self._bottleneck_channels)
        # self._kmeans_query_batch_norm_retrieved_value = get_norm('none', self._total_value_depth)
        "256→256"
        self.kmeans_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                             norm=None, act=None, conv_type='1d', norm_init=0.0)
        # self.kmeans_query_conv3_bn = MLP(input_dim=self._num_classes,hidden_dim=self._num_classes,
        #                                   output_dim=self._num_classes,num_layers=3)



    def forward(self,pixel_feature, query_feature):
        ""
        "centers=cluster_centers,"
        "      center_feature=content_ct, pixel_feature=content_uv"
        "pixel_feature: 中间特征: 8,512,1024"
        "query_feature: 聚类中心: 8,3,1024"

        "8*512*32*32"

        N, C, H, W = pixel_feature.shape
        _, D, L = query_feature.shape
        pixel_space = self.pixel_conv1_bn_act(F.gelu(pixel_feature))  # N C H W
        query_space = self.query_conv1_bn_act(query_feature)  # N x C x L

        # k-means cross-attention.
        pixel_value = self.pixel_v_conv_bn(pixel_space)  # N C H W
        pixel_value = pixel_value.reshape(N, self._total_value_depth, H * W)
        # k-means assignment.
        prediction_result = self.predcitor(
            mask_embeddings=query_space, class_embeddings=query_space, pixel_feature=pixel_space)


        with torch.no_grad():
            "1024, 1024"
            clustering_result = prediction_result['mask_logits'].flatten(2).detach()  # N L HW

            # index = clustering_result.max(1, keepdim=True)[1]
            # clustering_result_ = torch.zeros_like(clustering_result,
            #                                      memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            # clustering_result_ = clustering_result_[0].cpu().detach().numpy()

            # class_labels = torch.argmax(clustering_result, dim=1)

            # 创建一个简单的颜色映射
            # cmap = plt.get_cmap('tab10')  # 'tab10' 是一个常用的类别色彩图
            # batch_idx = 0  # 选择第一个样本进行可视化
            #
            # # 将类别标签映射到对应颜色
            # class_labels_np = clustering_result[batch_idx].cpu().numpy()
            # colored_image = cmap(class_labels_np / np.max(class_labels_np))  # 归一化到 0-1
            #
            # # 绘制结果
            # plt.imshow(colored_image)
            # plt.axis('off')
            # plt.title('Class Prediction Visualization')
            # plt.show()

            # clustering_result11 = clustering_result[0].cpu().detach().numpy()
            # plt.imshow(clustering_result11)
            # plt.colorbar()
            # plt.show()

            # print('d')


        with autocast(enabled=False):
            # k-means update.
            kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float())  # N x C x L

        kmeans_update = self.kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self.kmeans_query_conv3_bn(kmeans_update)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        # query self-attention.
        query_qkv = self.query_qkv_conv_bn(query_space)
        query_q, query_k, query_v = torch.split(query_qkv,
                                                [self._total_key_depth, self._total_key_depth, self._total_value_depth],
                                                dim=1)
        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        query_k = query_k.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        query_v = query_v.reshape(N, self._num_heads, self._total_value_depth // self._num_heads, L)
        self_attn_update = self.query_self_attention(query_q, query_k, query_v)
        self_attn_update = self.query_conv3_bn(self_attn_update)

        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)

        # FFN.
        ffn_update = self.query_ffn_conv1_bn_act(query_feature)
        ffn_update = self.query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return  query_feature, prediction_result['pixel_feature']

        # return query_feature, prediction_result

        # N, C, HW = pixel_feature.shape
        # "向量"
        # _, D, L = center_feature.shape
        #
        # "256→256"
        # "query_feature:聚类中心"
        # "求center与pixel之间的相似度"
        #
        # "MLP block: input_dim, hidden_dim, output_dim, num_layers"
        # "映射：1*1 一维卷积"
        # # center_feature = self.mlp_center(center_feature) .unsqueeze(3)
        # # Q_feature = self.mlp_q(center_feature)
        # # K_feature = self.mlp_k(pixel_feature)
        # # V_feature = self.mlp_v(pixel_feature)
        #
        # "Q→1536*256; K/V→512*256"
        # Q_feature = center_feature
        # K_feature = pixel_feature
        # V_feature = pixel_feature
        #
        #
        #
        # # k-means cross-attention.
        # # pixel_value = pixel_value.reshape(N, self._total_value_depth, H * W)
        # # k-means assignment.
        # "权重map: Qc × (Kp)T"
        # prediction_result = self._predcitor(
        #     mask_embeddings=Q_feature, class_embeddings=Q_feature, pixel_feature=K_feature)
        # # clustering_result = prediction_result['mask_logits'].flatten(2).detach()  # N L HW
        #
        # "clustering_argmax操作：argmax N (Qc × (Kp)T)"
        # with torch.no_grad():
        #     clustering_result = prediction_result.flatten(2).detach()  # N L HW
        #     index = clustering_result.max(1, keepdim=True)[1]
        #     "变成了0-1"
        #     clustering_result = torch.zeros_like(clustering_result,
        #                                          memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
        #
        # "argmax N (Qc × (Kp)T) × Vp"
        # "8*3*64 8*64*768"
        # # bv,cv,lv = V_feature.shape()
        # with autocast(enabled=False):
        #     # k-means update.
        #     kmeans_update = torch.einsum('blm,bmd->bld', clustering_result.float(), V_feature.float())  # N x C x L
        #
        # "get_norm: kmeans_update"
        # kmeans_update = self.kmeans_query_batch_norm_retrieved_value(kmeans_update)
        # "1*1卷积：256→256"
        # kmeans_update = self.kmeans_query_conv3_bn(kmeans_update)
        #
        # "drop_path_kmeans:类似于dropout"
        # # query_feature = center_feature + self.drop_path_kmeans(kmeans_update)
        #
        # centers = centers + self.drop_path_kmeans(kmeans_update)
        # # query_feature = centers
        #
        # # query self-attention.
        # "center之后作为Q，进行multi-head self attention"
        # "self attention 结构"
        # "1*1卷积：256→128+256"
        # query_qkv = self._query_qkv_conv_bn(centers)
        # "分离：query_feature:聚类中心"
        # query_q_, query_k_, query_v_ = torch.split(query_qkv,
        #                                         [self._in_channel_query, self._in_channel_query, self._in_channel_query],
        #                                         dim=1)
        #
        # "multi-attention transformer"
        # "_num_heads=8,_total_key_depth=128,L"
        # query_q = query_q_.reshape(N, self._num_heads, self._in_channel_query // self._num_heads, L)
        # query_k = query_k_.reshape(N, self._num_heads, self._in_channel_query // self._num_heads, L)
        # query_v = query_v_.reshape(N, self._num_heads, self._in_channel_query // self._num_heads, L)
        #
        # self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        # self_attn_update = self._query_conv3_bn(self_attn_update)
        #
        # query_q_ = query_q_ + self.drop_path_attn(self_attn_update)
        # # query_feature = query_feature + self.drop_path_attn(self_attn_update)
        # # query_feature = F.gelu(query_feature)
        #
        # # FFN.
        # ffn_update = self._query_fnn(query_q_)
        # # ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        # # ffn_update = self._query_ffn_conv2_bn(ffn_update)
        # query_q_ = query_q_ + self.drop_path_ffn(ffn_update)
        # query_q_ = F.gelu(query_q_)
        #
        # "query_feature → 更新后的color feature"
        # return centers,center_feature,query_q_

        # return query_feature, prediction_result

class attention1d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(3, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x,ct):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        batch_size, in_planes, height = x.size()

        softmax_attention = self.attention(ct)
        weight = self.weight.view(self.K, -1)
        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size,)

        x = x.view(1, -1, height, )# 变化成一个维度进行组卷积

        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-1))

        return output

# @TRANSFORMER_DECODER_REGISTRY.register()
class kMaXTransformerDecoder(nn.Module):

    def __init__(
            self,
            dec_layers=3,
            in_channels=1024,
            num_classes=1536,
            num_queries=1536,
            num_layers = 3,
            drop_path_prob=0.0,
            in_channel_pixel=512,
            in_channel_query=3,
            input_shape_from_backbone=192,   ## content_uv:8*192*16*16
    ):
        """
        NOTE: this interface is experimental.
        Args:
        """
        super().__init__()

        # define Transformer decoder here
        "需要定义panoptic_seg"

        self.kmax_transformer_layers = nn.ModuleList()
        self.ct_conv1d = nn.ModuleList()
        self._num_blocks = dec_layers
        self._num_layers = num_layers

        print(self._num_layers)


        "transformer 不同层: 修改"
        for _ in range(self._num_layers):
            self.kmax_transformer_layers.append(
                kMaXTransformerLayer(num_classes=num_classes,
                                 in_channel_pixel=in_channel_pixel,
                                 in_channel_query=in_channel_query,
                                 base_filters=128,
                                 num_heads=8,
                                 bottleneck_expansion=2,
                                 key_expansion=1,
                                 value_expansion=2,
                                 drop_path_prob=drop_path_prob)
            )

        # for _ in range(self._num_layers):
        #     self.ct_conv1d.append(CT_conv1d(in_channel_query=in_channel_query))


        self._num_queries = num_queries
        # learnable query features


        "不需要嵌入层,输入的center已经是连续向量表示了"
        "嵌入层，将离散元素映射到更加有意义的连续向量表示"
        self._cluster_centers = nn.Embedding(256, num_queries)
        "初始化模型参数"
        trunc_normal_(self._cluster_centers.weight, std=1.0)


        "ConvBN:卷积层+batchnorm"
        self.class_embedding_projection = ConvBN(3, 256, kernel_size=1, bias=False, norm=None, act='gelu',
                                                  conv_type='1d')

        self.mask_embedding_projection = ConvBN(3, 256, kernel_size=1, bias=False, norm=None, act='gelu',
                                                 conv_type='1d')

        self.color_embedding_projection = ConvBN(512, 256, kernel_size=1, bias=False, norm=None, act='gelu',
                                                conv_type='2d')

        self.predcitor = kMaXPredictor(in_channel_pixel=in_channel_pixel,
                                       in_channel_query=in_channel_query, num_classes=in_channel_query)

    def forward(self,cluster_centers,color_feature):
        ""

        current_transformer_idx = 0
        # ct_w = self.ct_embedding(ct_w.squeeze(3))
        # predictions_class = []
        # predictions_mask = []
        predictions_pixel_feature = []

        predictions_pixel_feature.append(color_feature)
        for _ in range(self._num_layers):
            cluster_centers, color_feature = self.kmax_transformer_layers[current_transformer_idx](
                pixel_feature=color_feature, query_feature=cluster_centers
            )
            # predictions_class.append(prediction_result['class_logits'])
            # predictions_mask.append(prediction_result['mask_logits'])

            # "-----------------------------------------------------------"
            # "可视化用，测试/训练得删掉"\
            # "pcamap: 色温中心"
            # pcapmap_cluster(cluster_centers, color_feature)
            # "-----------------------------------------------------------"


            # cc_ = cluster_centers.reshape(1,3,32,32)
            # cc_ = cc_.permute(0,2,3,1).cpu().detach().numpy()
            #
            # plt.imshow(cc_[0])
            # plt.show()

            # plt.figure(figsize=(3, 1))
            # for i in range(1):
            #     for j in range(3):
            #         plt.subplot(1, 3, i * 3 + j + 1)
            #         plt.imshow(cc_[i, j])
            #         plt.axis('off')
            # plt.show()


            current_transformer_idx += 1

        class_embeddings = self.class_embedding_projection(cluster_centers)
        "mask_embeddings:1024,32,32"
        mask_embeddings = self.mask_embedding_projection(cluster_centers)
        color_embeddings = self.color_embedding_projection(color_feature)



        # Final predictions.
        prediction_result = self.predcitor(
            class_embeddings=class_embeddings,
            mask_embeddings=mask_embeddings,
            pixel_feature=color_embeddings,
        )
        # predictions_class.append(prediction_result['class_logits'])
        # predictions_mask.append(prediction_result['mask_logits'])
        predictions_pixel_feature.append(prediction_result['pixel_feature'])


        return prediction_result['class_logits'], prediction_result['mask_logits'], prediction_result['pixel_feature'], predictions_pixel_feature


