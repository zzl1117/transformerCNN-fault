##初版实现的模型框架用于故障诊断
# In[1] 导入所需工具包
import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import time
from torch.nn import functional as F
from math import floor, ceil
import math
from collections import OrderedDict
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.utils import data
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
# import torchvision.transforms as transforms
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
import random
import math, copy
# In[1] 加载模型
# In[1]我们需要用一个普通的线性层来投影它们
# 创建一个PatchEmbedding类来保持代码整洁
'''
注意：在检查了最初的实现之后，我发现为了提高性能，作者使用了Conv2d层而不是线性层,
这是通过使用与“patch_size”相等的kernel_size和stride 来获得的。
直观地说，卷积运算分别应用于每个切片。
因此，我们必须首先应用conv层，然后将生成的图像展平
'''

# In[1]下一步是添加cls标记和位置嵌入。cls标记只是每个序列（切片）中的一个数字。
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 用卷积层代替线性层->性能提升
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # 先用大小为切片大小，步长为切片步长的卷积核来提取特征图，然后将特征图展平
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))
        # self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        # print(torch.randn((img_size // patch_size) **2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:#x:torch.Size([64, 1, 84, 84])
        b, _, _, _ = x.shape#b:64
        x = self.projection(x)#torch.Size([64, 144, 512])
        # print(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) #cls_tokens:torch.Size([64, 1, 512])
        # 在输入前添加cls标记
        x = torch.cat([cls_tokens, x], dim=1) #torch.Size([64, 145, 512])
        # print(self.positions.size())
        # 加位置嵌入
        x += self.positions
        # print(x,x.size())
        # print("PatchEmbedding:", x.size())
        return x

# print(PatchEmbedding()(x).shape)
# In[1]用nn.MultiHadAttention或实现我们自己的。为了完整起见，我将展示它的样子
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h=8, d_model=512, dropout=0.2):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0  # 剖析点1
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 纬度
        # shape:query=key=value--->:[batch_size,max_legnth,embedding_dim=512]

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 剖析点2
        nbatches = query.size(0)

        # 第一步：将q,k,v分别与Wq，Wk，Wv矩阵进行相乘
        # shape:Wq=Wk=Wv----->[512,512]
        # 第二步：将获得的Q、K、V在第三个纬度上进行切分
        # shape:[batch_size,max_length,8,64]
        # 第三部：填充到第一个纬度
        # shape:[batch_size,8,max_length,64]
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]  # 剖析点3

        # 进入到attention之后纬度不变，shape:[batch_size,8,max_length,64]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 将纬度进行还原
        # 交换纬度：[batch_size,max_length,8,64]
        # 纬度还原：[batch_size,max_length,512]
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)  # 剖析点4

        # 最后与WO大矩阵相乘 shape:[512,512]
        return self.linears[-1](x)

class MultiHeadedDilatedConv(nn.Module):
    def __init__(self, h=8, d_model=512, dropout=0.2):
        "Take in model size and number of heads."
        super(MultiHeadedDilatedConv, self).__init__()
        assert d_model % h == 0  # 剖析点1
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1,
                          dilation=1)
        # self.conv1 = nn.Conv1d(in_channels=1152, out_channels=1152, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=2,
                           dilation=2)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=3,
                               dilation=3)

    def forward(self, query, key, value, mask=None):
        # 纬度
        # shape:query=key=value--->:[batch_size,max_legnth,embedding_dim=512]

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 剖析点2
        nbatches = query.size(0)

        query=query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        query1 = self.conv1(query)
        query2 = self.conv2(query)
        query3 = self.conv3(query)
        query = query1+query2+query3

        key1 = self.conv1(key)
        key2 = self.conv2(key)
        key3 = self.conv3(key)
        key = key1 + key2 + key3

        value1 = self.conv1(value)
        value2 = self.conv2(value)
        value3 = self.conv3(value)
        value = value1 + value2 + value3

        x = query+key+value
        # 将纬度进行还原
        # 交换纬度：[batch_size,max_length,8,64]
        # 纬度还原：[batch_size,max_length,512]
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)  # 剖析点4

        # 最后与WO大矩阵相乘 shape:[512,512]
        return self.linears[-1](x)

# print(MultiHeadAttention()(patches_embedded).shape)

# In[1]反馈组件
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion: int = 3, drop_p: float = 0.2):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# In[1]创建Transformer编码器块
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.2,
                 **kwargs):
        super(TransformerEncoderBlock, self).__init__()

        # First sub-block
        self.norm1 = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(**kwargs)
        self.dropout1 = nn.Dropout(drop_p)

        self.norm3 = nn.LayerNorm(emb_size)
        self.multiDilate = MultiHeadedDilatedConv(**kwargs)
        self.dropout3 = nn.Dropout(drop_p)
        # Second sub-block
        self.norm2 = nn.LayerNorm(emb_size)
        self.feedforward = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self, x):
        # First sub-block
        residual1 = x
        x = self.norm1(x)
        x = self.attention(x,x,x)
        #可以在这个地方加另一个注意力
        x = self.dropout1(x)
        x = x + residual1  # Residual connection

        residual3 = x
        x = self.norm3(x)
        x = self.multiDilate(x,x,x)
        # 可以在这个地方加另一个注意力
        x = self.dropout3(x)
        x = x + residual3
        # Second sub-block
        residual2 = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = x + residual2  # Residual connection
        x = x + residual1
        return x


# patches_embedded = PatchEmbedding()(x)
# print(TransformerEncoderBlock()(patches_embedded).shape)
# In[1]创建Transformer编码器

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

# In[1]分类器
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),#transformer模型最后送到mlp中做预测的只有cls_token的输出结果（如上图红框所示），而其他的图像块的输出全都不要了
            # Reduce('b n e -> b e', reduction='first'),
            # nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

    def forward(self, x: Tensor) -> Tensor:
        # Take only the first element along the second dimension
        # x = x[:, 0, :]
        x = super(ClassificationHead, self).forward(x)
        return x


# In[1]ViT架构
# 组成PatchEmbedding、TransformerEncoder和ClassificationHead来创建最终的ViT架构
class ViTNET(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 21,
                 emb_size: int = 512,
                 img_size: int = 105,
                 depth: int = 4,
                 n_classes: int = 10,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# batch_size = 64
# in_channels = 1
# img_size = 105
# patch_size = 21
# emb_size = 512
# depth = 4
# n_classes = 10
# # from torchviz import make_dot
# vit_model = ViTNET(in_channels, patch_size, emb_size, img_size, depth, n_classes)
# # 将模型移到GPU上，如果可用
# vit_model.to(device)
# # 创建一个虚拟输入
# dummy_input = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
# # 生成网络输出的图形
# output = vit_model(dummy_input)
# g = make_dot(output)
# 保存图形为PDF文件
# g.render(filename='graph',format='pdf', view=False)  # 保存为 graph.pdf，参数view表示是否打开pdf
#
# # 生成随机输入数据
# x = torch.randn(batch_size, in_channels, img_size, img_size)
#
# # 运行模型进行前向传播
# output = vit_model(x)
#
# # 输出模型的结构和输出的大小
# print(vit_model)
# print("Input size:", x.size())
# print("Output size:", output.size())
