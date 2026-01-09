import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)
# from function import normal, normal_style
# from function import calc_mean_std
# import scipy.stats as stats
# from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from ViT_helper import DropPath, to_2tuple, trunc_normal_
# from models.uv_hist import RGBuvHistBlock
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# import DWB.arch.splitNetworks as splitter
# from DWB.arch.deep_wb_model import deepWBNet
# import models.transformer_OR as transformer
import transformer_OR as transformer
from pcapmap import pcapmap_cluster

class Flatten(nn.Module):
  def forward(self, x):
    return x.reshape(x.shape[0], -1)

class UnFlatten(nn.Module):
  def forward(self, x,c,h,w):
    return x.reshape(x.shape[0], c,h,w)

def leaky_relu(p=0.2):
  return nn.LeakyReLU(p, inplace=True)

def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

class HistVectorizer(nn.Module):
  "32,192,8,device"
  def __init__(self, insize=16, emb=192, depth=8):
    super().__init__()
    self.flatten = Flatten()
    self.unflatten = UnFlatten()

    # self.insize = insize

    fc_layers = []
    " insize * insize * 3 "
    for i in range(depth-1):
      if i == 0:
        fc_layers.extend(
          [nn.Linear(emb*insize*insize, emb * 2), leaky_relu()])
      elif i == 1:
        fc_layers.extend([nn.Linear(emb * 2, emb), leaky_relu()])
      else:
        fc_layers.extend([nn.Linear(emb, emb), leaky_relu()])
    fc_layers.extend([nn.Linear(emb, emb*insize*insize), leaky_relu()])

    self.fcs = nn.Sequential(*fc_layers)


    "RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x196608 and 3072x384)"


  def forward(self, x):
    # x = x.cuda()
    B,C,H,W = x.shape
    x1 = self.flatten(x)
    "多cuda问题"
    x2 = self.fcs(x1)
    x3 = self.unflatten(x2,C,H,W)
    return x3

class CTBANK(nn.Module):
  "32,192,8,device"
  def __init__(self, insize=3, emb=768, depth=6):
    super().__init__()
    self.flatten = Flatten()
    self.unflatten = UnFlatten()

    fc_layers = []
    " insize * insize * 3 "
    for i in range(depth-1):
      if i == 0:
        fc_layers.extend(
          [nn.Linear(emb*insize, emb * 2), leaky_relu()])
      elif i == 1:
        fc_layers.extend([nn.Linear(emb * 2, emb), leaky_relu()])
      else:
        fc_layers.extend([nn.Linear(emb, emb), leaky_relu()])
    fc_layers.extend([nn.Linear(emb, emb*insize), leaky_relu()])

    self.fcs = nn.Sequential(*fc_layers)


    "RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x196608 and 3072x384)"


  def forward(self, x):
    # x = x.cuda()
    B,C,H,W = x.shape
    x1 = self.flatten(x)
    "多cuda问题"
    x2 = self.fcs(x1)
    x3 = self.unflatten(x2,C,H,W)
    return x3


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=128, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)

        return x



class decoder(nn.Module):
    def __init__(self, inchannel=512):
        super(decoder, self).__init__()
        # self.model = nn.Sequential(
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(inchannel, inchannel, (3, 3)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(inchannel, inchannel, (3, 3)),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(inchannel, inchannel, (3, 3)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(inchannel, int(inchannel//4), (3, 3)),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(int(inchannel//4), int(inchannel//4), (3, 3)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(int(inchannel//4), 64, (3, 3)),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(64, 64, (3, 3)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(64, 3, (3, 3)),
        # )
        "FOR BASE_Trans_3_8_256_OR_awl3_bz8"
        self.model = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1)),  # Adjusted to match pretrained model
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1)),  # Final output adjusted
        )

    def forward(self, stylized_feature):
        x = self.model(stylized_feature)
        return x


class decoder_ex(nn.Module):
    def __init__(self, inchannel=512):
        super(decoder_ex, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(inchannel, inchannel, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(inchannel, inchannel, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(inchannel, inchannel, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(inchannel, int(inchannel//4), (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(int(inchannel//4), int(inchannel//4), (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(int(inchannel//4), 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 32, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 3, (3, 3)),
        )

    def forward(self, stylized_feature):
        x = self.model(stylized_feature)
        return x


# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



"主函数"

class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    "args,device,dwb,embedding,histencoder,ctbank,kmaxtrans,Trans,decoder)"
    def __init__(self,device):
        super().__init__()


        "UV-HIS block"
        self.insize = 16
        intensity_scale = True
        # histogram_size = 128
        # max_input_size = 128
        # histogram_size = 16
        max_input_size = 16
        method = 'inverse-quadratic'  # options:'thresholding','RBF','inverse-quadratic'

        "create a histogram block"
        "extract the uv-hist of each block"
        # self.histogram_block = RGBuvHistBlock(insz=max_input_size, h=histogram_size,
        #                                  intensity_scale=intensity_scale,
        #                                  method=method,
        #                                  device=device)

        self.device = device
        # self.mse_loss = nn.MSELoss()
        self.transformer = transformer.Transformer()
        # self.transformer_forcontent = transformer.Transformer_forcontent()
        # hidden_dim = transformer.d_model
        self.decoder = decoder()
        # self.decoder = decoder_ex()
        self.embedding_img = PatchEmbed()
        self.embedding_hist = PatchEmbed()

        # embedding centers：
        self.num_centers = 3
        self.cluster_centers = nn.Embedding(1024, self.num_centers) #32*32
        trunc_normal_(self.cluster_centers.weight, std=1.0)


        ".permute(2,0,1)"
        "uv_prior_img_128"


        "loss设置"
        # self.log_vars = nn.Parameter(torch.zeros((3)), requires_grad=True)

        "pooling设置"
        self.pool = nn.AdaptiveAvgPool2d(16)
        self.encoder_color = nn.Sequential(nn.Conv2d(448, 448, 3, 1, 1),
                                             nn.ReLU(),
                                             nn.Conv2d(448, 1024, 3, 1, 1),
                                             nn.ReLU(),
                                             nn.Conv2d(1024, 1024, 3, 1, 1),
                                             nn.ReLU())




    # def forward(self,samples_c: NestedTensor,samples_s: NestedTensor):
    "测试/训练的时候，filename得删掉"
    def forward(self, samples_c):
        ""

        "分别用两个patch_embedding来提取颜色和内容信息 → "
        "content_feat: [-1],[-2]"
        "style_feat: [0-5]"
        ### Linear projection
        "B*C*H*W → B*(C*H*W/P^2)*P*P"
        content = self.embedding_img(samples_c)
        color = self.embedding_hist(samples_c)

        B = color.shape[0]
        cluster_centers = self.cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1)

        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None

        "-----------------------------------------------------------"
        "可视化用，测试/训练得删掉"\
        "pcamap: 色温中心"
        # filename_in = filename.split('.')[0]+"_in.jpg"
        # pcapmap_cluster(cluster_centers, color, filename_in)
        "-----------------------------------------------------------"

        mask = None
        "先可视化输入，cluster_centers 和 content，再可视化中间量"
        hs, style_class,style_feature = self.transformer(color, cluster_centers, mask, content, pos_s)
        Ics = self.decoder(hs)


        "-----------------------------------------------------------"
        "可视化用，测试/训练得删掉" \
        "pcamap: 色温中心"
        # filename_out = filename.split('.')[0] + "_out.jpg"
        # pcapmap_cluster(style_class, style_feature, filename_out)
        "-----------------------------------------------------------"


        return Ics