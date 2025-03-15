import math
import os
import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath

from models.arcface_model import Backbone
from models.temporal_convolutional_model import TemporalConvNet
from models.transformer import MultimodalTransformerEncoder, IntraModalTransformerEncoder, InterModalTransformerEncoder
from models.backbone import VisualBackbone, AudioBackbone


import torch
from torch import nn

import numpy as np

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module
import torch.nn.functional as F

root_dir = dirname(dirname((abspath(__file__))))
sys.path.append(root_dir)

import constants
from torch.nn import Identity

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import ViTImageProcessor, ViTForImageClassification

import torch.nn as nn
from transformers import ViTModel


class ViTFeatureExtractor(nn.Module):
    def __init__(self,
                 model_name="google/vit-base-patch16-224",
                 input_size=40,
                 output_dim=512,
                 freeze_backbone=True,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        # 1. 输入预处理模块 (保持通道优先顺序)
        self.input_adapter = nn.Sequential(
            nn.Upsample(size=224, mode='bilinear'),  # 40x40 -> 224x224
            nn.BatchNorm2d(3)  # 输入形状 [B,3,224,224]
        )

        # 2. 加载ViT模型 (无需调整通道顺序)
        self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=False)

        # 3. 标准化参数 (ImageNet标准)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # 添加投影层
        self.projection = nn.Linear(768, output_dim)
        # 4. 冻结参数
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad_(False)

        self.device = device
        self.to(device)

    def preprocess(self, x):
        """输入预处理流水线"""
        x = self.input_adapter(x)  # 上采样到224x224
        x = (x - self.mean) / self.std  # 标准化
        return x

    def forward(self, x):
        """
        输入:
            x - 形状为 (B, 3, 40, 40) 的Tensor
        输出:
            features - 形状为 (B, 768) 的CLS特征向量
        """
        # 输入预处理 (保持通道优先)
        x = self.preprocess(x.to(self.device))  # 输出形状 [B,3,224,224]

        # 直接输入ViT (无需调整维度顺序)
        outputs = self.vit(pixel_values=x, output_hidden_states=False)

        return outputs.last_hidden_state[:, 0, :]


def vit_pre():
    # 处理器，用于将图像转换为模型所需的格式
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    # 加载预训练的 ViT 模型
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    for param in model.parameters():
        param.requires_grad = True

    model.classifier = Identity()

    return model, processor
from PIL import Image
class vit(nn.Module):
    def __init__(self):
        super(vit,self).__init__()
        # 处理器，用于将图像转换为模型所需的格式
        self.resize = nn.Sequential(
            nn.AdaptiveAvgPool2d((224, 224)),
            nn.Conv2d(3, 3, kernel_size=1)  # 通道数适配
        )
        self.model,self.processor = vit_pre()

        self.model.classifier = Identity()


    def forward(self, x):

        #print(x.shape)
        resized_tensor = self.resize(x)
        normalized_tensor = (resized_tensor - resized_tensor.min()) / (resized_tensor.max() - resized_tensor.min())
        images = []
        for img in normalized_tensor:
            img = img.permute(1, 2, 0).cpu().numpy()  # 调整通道顺序
            img = (img * 255).astype(np.uint8)  # 转换为 [0, 255] 范围
            img = Image.fromarray(img)
            images.append(img)

        x=self.processor(images=images, return_tensors="pt")
        x = x.to(self.model.device)  # 将输入数据移动到模型所在设备
        x=self.model(**x).logits
        #print(type(x))
        return x


class my_res50(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=8,
                 use_pretrained=True,
                 state_dict_name='',
                 root_dir='',
                 mode="ir",
                 embedding_dim=512
                 ):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels,
                                 num_layers=50,
                                 drop_ratio=0.4,
                                 mode=mode
                                 )
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')

            if "backbone" in list(state_dict.keys())[0]:

                self.backbone.output_layer = Sequential(
                    BatchNorm2d(embedding_dim),
                    Dropout(0.4),
                    Flatten(),
                    Linear(embedding_dim * 5 * 5, embedding_dim),
                    BatchNorm1d(embedding_dim)
                )

                new_state_dict = {}
                for key, value in state_dict.items():

                    if "logits" not in key:
                        new_key = key[9:]
                        new_state_dict[new_key] = value

                self.backbone.load_state_dict(new_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.output_layer = Sequential(
            BatchNorm2d(embedding_dim),
            Dropout(0.4),
            Flatten(),
            Linear(embedding_dim * 5 * 5, embedding_dim),
            BatchNorm1d(embedding_dim)
        )

        self.logits = nn.Linear(in_features=embedding_dim,
                                out_features=num_classes)

    def forward(self, x, extract_cnn=False):
        x = self.backbone(x)

        if extract_cnn:
            return x

        x = self.logits(x)
        return x


class LeaderFollowerAttentionNetwork(nn.Module):
    def __init__(self,
                 backbone_state_dict,
                 modality=['frame'],
                 kernel_size=5,
                 example_length=300,
                 tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128],
                              'cnn_res50': [512, 256, 256, 128],
                              'mfcc':[32, 32, 32, 32],
                              'vggish': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,
                                'bert': 768,
                                'cnn_res50': 512,
                                'mfcc': 39,
                                'vggish': 128,
                                'egemaps': 23},
                 encoder_dim={'video': 128,
                              'bert': 128,
                              'cnn_res50': 128,
                              'mfcc': 32,
                              'vggish': 32,
                              'egemaps': 32},
                 root_dir='',
                 device='cuda'):
        super().__init__()
        self.backbone_state_dict = backbone_state_dict
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.encoderQ, self.encoderK, self.encoderV = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()

        self.final_dim = self.encoder_dim[self.modality[0]] + 32*len(self.modality)
        self.spatial = None

    def init(self):
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

            self.encoderQ[modal] = nn.Linear(self.encoder_dim[modal], 32)
            self.encoderK[modal] = nn.Linear(self.encoder_dim[modal], 32)
            self.encoderV[modal] = nn.Linear(self.encoder_dim[modal], 32)

        self.ln = nn.LayerNorm([len(self.modality), 32])
        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()
            x[modal] = x[modal].contiguous().view(batch_size * self.example_length, -1)

        Q = [self.encoderQ[modal](x[modal]) for modal in self.modality]
        K = [self.encoderK[modal](x[modal]) for modal in self.modality]
        V = [self.encoderV[modal](x[modal]) for modal in self.modality]

        Q = torch.stack(Q, dim=-2)
        K = torch.stack(K, dim=-2)
        V = torch.stack(V, dim=-2)

        QT = Q.permute(0, 2, 1)
        scores = torch.matmul(K, QT) / math.sqrt(32)
        scores = nn.functional.softmax(scores, dim=-1)

        follower = torch.matmul(scores, V)
        follower = self.ln(follower + V)
        follower = follower.view(follower.size()[0], -1)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

        return x


class LeaderFollowerAttentionNetworkWithMultiHead(nn.Module):
    def __init__(self, backbone_state_dict, modality=['frame'],
                 kernel_size=5, example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128],
                              'cnn_res50': [512, 256, 256, 128],
                              'mfcc':[32, 32, 32, 32],
                              'vggish': [32, 32, 32, 32]},
                 embedding_dim={'video': 512, 'bert': 768, 'cnn_res50': 512,
                                'mfcc': 39, 'vggish': 128, 'egemaps': 23},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128,
                              'mfcc': 32, 'vggish': 32, 'egemaps': 32},
                 modal_dim=32, num_heads=2,
                 root_dir='', device='cuda'):
        super().__init__()
        self.backbone_state_dict = backbone_state_dict
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.fusion = nn.ModuleDict(), None
        self.num_heads = num_heads
        self.modal_dim = modal_dim
        self.final_dim = self.encoder_dim[self.modality[0]] + self.modal_dim*len(self.modality)
        self.spatial = None

    def init(self):
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()

        follower = self.fusion(x)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

        return x


class LeaderFollowerAttentionNetworkWithMultiHead(nn.Module):
    def __init__(self, backbone_state_dict, modality=['frame'], kernel_size=5,
                 example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128],
                              'cnn_res50': [512, 256, 256, 128],
                              'mfcc':[32, 32, 32, 32],
                              'vggish': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512,
                                'mfcc': 39, 'vggish': 128, 'egemaps': 23},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128,
                              'mfcc': 32, 'vggish': 32, 'egemaps': 32},
                 modal_dim=32, num_heads=2,
                 root_dir='', device='cuda'):
        super().__init__()
        self.backbone_state_dict = backbone_state_dict
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.fusion = nn.ModuleDict(), None
        self.num_heads = num_heads
        self.modal_dim = modal_dim
        self.final_dim = self.encoder_dim[self.modality[0]] + self.modal_dim*len(self.modality)
        self.spatial = None

    def init(self):
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()

        follower = self.fusion(x)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

        return x


class LFAN(nn.Module):
    def __init__(self,
                 backbone_settings,
                 output_dim: int,
                 task: str,
                 modality=['frame'],
                 kernel_size=5,
                 example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128],
                              'cnn_res50': [512, 256, 256, 128],
                              'mfcc':[32, 32, 32, 32],
                              'vggish': [32, 32, 32, 32],
                              'logmel': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512,
                                'mfcc': 39, 'vggish': 128, 'logmel': 128,
                                'egemaps': 88},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128,
                              'mfcc': 32, 'vggish': 32, 'logmel': 32,
                              'egemaps': 32},
                 modal_dim=32,
                 num_heads=2,
                 root_dir='',
                 device='cuda'):
        super().__init__()

        # 新增特征融合层参数
        self.fusion_layers = nn.ModuleDict()

        assert task in constants.TASKS, task
        self.task = task

        self.output_dim = output_dim
        self.backbone_settings = backbone_settings
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.fusion = nn.ModuleDict(), None
        self.num_heads = num_heads
        self.modal_dim = modal_dim

        self.final_dim = self.encoder_dim[self.modality[0]
                         ] + self.modal_dim * len(self.modality)
        self.spatial = nn.ModuleDict()
        self.bn = nn.ModuleDict()

    def load_visual_backbone(self, backbone_settings):

        resnet = VisualBackbone(mode='ir', use_pretrained=False)


        state_dict = torch.load(join(self.root_dir,
                                     backbone_settings['visual_state_dict'] + ".pth"),
                                map_location='cpu')
        resnet.load_state_dict(state_dict)

        # ViT特征提取器
        vit_model = ViTFeatureExtractor(output_dim=512)


        for model in [resnet, vit_model]:
            for param in model.parameters():
                param.requires_grad = False

        return nn.ModuleDict({'resnet': resnet, 'vit': vit_model})  # 返回双模型

    def load_audio_backbone(self, backbone_settings):

        vggish = AudioBackbone()
        state_dict = torch.load(join(self.root_dir,
                                     backbone_settings['audio_state_dict'] + ".pth"),
                                map_location='cpu')
        vggish.backbone.load_state_dict(state_dict)

        for param in vggish.parameters():
            param.requires_grad = False


        return vggish

    def init(self):
        # self.output_dim = 1

        # 视觉特征初始化
        if 'video' in self.modality:
            self.root_dir = self.root_dir
            self.spatial["video"] = self.load_visual_backbone(
                backbone_settings=self.backbone_settings)

            # 新增特征融合层
            self.fusion_layers['video'] = nn.Sequential(
                nn.Linear(512 + 768, 512),  # ResNet(512)+ViT(768)→512
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        """if 'video' in self.modality:
            self.root_dir = self.root_dir
            self.spatial["visual"] = self.load_visual_backbone(
                backbone_settings=self.backbone_settings)"""

        if 'logmel' in self.modality:
            self.root_dir = self.root_dir
            self.spatial["audio"] = self.load_audio_backbone(
                backbone_settings=self.backbone_settings)

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(
                num_inputs=self.embedding_dim[modal],
                max_length=self.example_length,
                num_channels=self.tcn_channel[modal],
                attention=self.tcn_attention,
                kernel_size=self.kernel_size,
                dropout=0.1
            ).to(self.device)
            self.bn[modal] = BatchNorm1d(self.tcn_channel[modal][-1])

        self.fusion = MultimodalTransformerEncoder(
            modalities=self.modality,
            input_dim=self.encoder_dim,
            modal_dim=self.modal_dim,
            num_heads=self.num_heads,
            dropout=0.1
        )

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, X):
        if 'video' in X:
            batch_size, length, c, w, h = X['video'].shape
            video_input = X['video'].view(-1, c, w, h)

            # 并行提取双特征
            resnet_feat = self.spatial['video']['resnet'](video_input)  # [B*T, 512]
            vit_feat = self.spatial['video']['vit'](video_input) # [B*T, 768]

            # 特征融合
            fused_feat = torch.cat([resnet_feat, vit_feat], dim=-1)
            fused_feat = self.fusion_layers['video'](fused_feat)  # [B*T, 512]

            # 恢复时序结构
            X['video'] = fused_feat.view(batch_size, length, -1).unsqueeze(1)
        """ if 'video' in X:
            batch_size, length, channel, width, height = X['video'].shape
            X['video'] = X['video'].view(-1, channel, width, height)
            # [batch x length, channel, width, height]

            X['video'] = self.spatial.visual(X['video'])
            _, feature_dim = X['video'].shape
            X['video'] = X['video'].view(batch_size, length, feature_dim
                                         ).unsqueeze(1)
            # [batch, 1, length, feature_dim]"""

        if 'logmel' in X:
            batch_size, height, length, width = X['logmel'].shape
            X['logmel'] = X['logmel'].permute((0, 2, 3, 1)).contiguous()
            X['logmel'] = X['logmel'].view(-1, width, height)
            # [batch x length, channel, width, height]
            X['logmel'] = self.spatial.audio(X['logmel'])
            _, feature_dim = X['logmel'].shape
            X['logmel'] = X['logmel'].view(batch_size, length, feature_dim
                                           ).unsqueeze(1)
            # [batch, 1, length, feature_dim]

        for modal in X:
            batch_size = X[modal].shape[0]
            X[modal] = X[modal].squeeze(1).transpose(1, 2)
            X[modal] = self.temporal[modal](X[modal])
            X[modal] = self.bn[modal](X[modal]).transpose(1, 2)

        follower = self.fusion(X)

        X = torch.cat((X[self.modality[0]], follower), dim=-1)
        X = self.regressor(X)
        X = X.view(batch_size, self.example_length, -1)

        if self.task == constants.REGRESSION:
            X = torch.tanh(X)

        return X


class AttentionFusion(nn.Module):
    """ Fuse modalities using attention. """

    def __init__(self,
                 num_feats_modality: list,
                 num_out_feats: int = 256):
        """ Instantiate attention fusion instance.

        Args:
            num_feats_modality (list): Number of features per modality.
            num_out_feats (int): Number of output features.
        """

        super(AttentionFusion, self).__init__()

        self.attn = nn.ModuleList([])
        for num_feats in num_feats_modality:
            self.attn.append(
                nn.Linear(num_feats, num_out_feats))

        self.weights = nn.Linear(num_out_feats * len(num_feats_modality), num_out_feats * len(num_feats_modality))
        self.num_features = num_out_feats * len(num_feats_modality)

    def forward(self, x: list):
        """ Forward pass

        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """

        proj_m = []
        for i, m in enumerate(x.values()):
            proj_m.append(self.attn[i](m.transpose(1, 2)))

        attn_weights = F.softmax(
            self.weights(torch.cat(proj_m, -1)), dim=-1)

        out_feats = attn_weights * torch.cat(proj_m, -1)

        return out_feats


class CAN(nn.Module):
    def __init__(self,
                 task: str,
                 modalities,
                 tcn_settings,
                 backbone_settings,
                 output_dim,
                 root_dir,
                 device
                 ):
        super().__init__()
        assert task in constants.TASKS, task
        self.task = task

        self.device = device

        self.temporal = nn.ModuleDict()
        self.up_sample = nn.ModuleDict()
        self.bn = nn.ModuleDict()

        self.spatial = nn.ModuleDict()


        for modal in modalities:
            self.temporal[modal] = TemporalConvNet(
                num_inputs=tcn_settings[modal]['input_dim'],
                num_channels=tcn_settings[modal]['channel'],
                kernel_size=tcn_settings[modal]['kernel_size']
            )
            self.bn[modal] = BatchNorm1d(tcn_settings[modal]['channel'][-1])


        feas_modalities = [tcn_settings[modal]['channel'][-1] for modal in modalities]
        self.fuse = AttentionFusion(num_feats_modality=feas_modalities,
                                    num_out_feats=128)

        self.conv_c = nn.Conv1d(128 * len(modalities), 128, 1)

        self.bn1 = BatchNorm1d(128 * len(modalities))
        self.fc1 = Linear(128 * len(modalities), 128 * len(modalities))
        self.fc2 = Linear(128 * len(modalities), output_dim)


        if 'video' in modalities:
            self.root_dir = root_dir
            self.spatial["visual"] = self.load_visual_backbone(
                backbone_settings=backbone_settings)

        if 'logmel' in modalities:
            self.root_dir = root_dir
            self.spatial["audio"] = self.load_audio_backbone(
                backbone_settings=backbone_settings)

    def load_visual_backbone(self, backbone_settings):

        resnet = VisualBackbone(mode='ir', use_pretrained=False)
        state_dict = torch.load(join(self.root_dir, backbone_settings['visual_state_dict'] + ".pth"),
                                map_location='cpu')
        resnet.load_state_dict(state_dict)

        for param in resnet.parameters():
            param.requires_grad = False


        return resnet

    def load_audio_backbone(self, backbone_settings):

        vggish = AudioBackbone()
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['audio_state_dict'] + ".pth"),
                                map_location='cpu')
        vggish.backbone.load_state_dict(state_dict)

        for param in vggish.parameters():
            param.requires_grad = False


        return vggish


    def forward(self, X):

        x = {}

        if 'video' in X:
            batch_size, length, channel, width, height = X['video'].shape
            X['video'] = X['video'].view(-1, channel, width, height) # [batch x length, channel, width, height]
            X['video'] = self.spatial.visual(X['video'])
            _, feature_dim = X['video'].shape
            X['video'] = X['video'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        if 'logmel' in X:
            batch_size, height, length, width = X['logmel'].shape
            X['logmel'] = X['logmel'].permute((0, 2, 3, 1)).contiguous()
            X['logmel'] = X['logmel'].view(-1, width, height) # [batch x length, channel, width, height]
            X['logmel'] = self.spatial.audio(X['logmel'])
            _, feature_dim = X['logmel'].shape
            X['logmel'] = X['logmel'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        for modal in X:
            x[modal] = X[modal].squeeze(1).transpose(1, 2)
            x[modal] = self.temporal[modal](x[modal])
            x[modal] = self.bn[modal](x[modal])

        c = self.fuse(x)
        c = self.fc1(c).transpose(1, 2)
        c = self.bn1(c).transpose(1, 2)
        c = F.leaky_relu(c)
        c = self.fc2(c)

        if self.task == constants.REGRESSION:
            c = torch.tanh(c)

        return c
## My code
    
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(input_dim, 2)

        self.out_layer1 = nn.Linear(512, 256)
        self.out_layer2 = nn.Linear(256, 64)
        self.out_layer3 = nn.Linear(64, 2)

    def forward(self, x):
        q = self.W(x)
        attn_weights = torch.softmax(self.V(self.tanh(q)), dim=1)
        attended_x = attn_weights * x
        out = self.out_layer1(attended_x)
        out = self.out_layer2(out)
        out = self.out_layer3(out)
        return out


class SequentialEncoder(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = SequentialEncoder(
            *[TransformerEncoderLayer(input_dim, num_heads, hidden_dim)
              for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x

class JMT_old(nn.Module):
    def __init__(self,modalities, tcn_settings, backbone_settings, output_dim,
                 root_dir, device,    visual_dim, audio_dim, num_heads,
                 hidden_dim,
                 num_layers, output_format: str):
        super(JMT, self).__init__()

        assert output_format in ['FC', 'SELF_ATTEN'], output_format
        self.output_format = output_format

        # Encoder blocks
        self.visual_encoder = TransformerEncoderBlock(128, num_heads,
                                                      hidden_dim, num_layers)
        self.physiological_encoder = TransformerEncoderBlock(128, num_heads,
                                                             hidden_dim,
                                                             num_layers)
        self.joint_representation_encoder = TransformerEncoderBlock(128,
                                                                    num_heads,
                                                                    hidden_dim,
                                                                    num_layers)
        self.final_encoder = TransformerEncoderBlock(128*6, num_heads,
                                                     hidden_dim, num_layers)

        # Cross attention
        self.cross_attention_v = nn.MultiheadAttention(128, num_heads)
        self.cross_attention_p = nn.MultiheadAttention(128, num_heads)
        self.cross_attention_pv = nn.MultiheadAttention(128, num_heads)

        # Fully connected layer for joint representation
        self.out_layer_pv = nn.Linear(128*2, 128)

        if output_format == 'FC':
            self.out_layer1 = nn.Linear(128*6, 1024)

        elif output_format == 'SELF_ATTEN':
            self.final_visual_encoder = TransformerEncoderBlock(128,
                                                                num_heads,
                                                                hidden_dim,
                                                                num_layers)
            self.final_self_attention = nn.MultiheadAttention(512, num_heads)

        else:
            raise NotImplementedError(output_format)

    def forward(self, visual_features, physiological_features):
        # Concatenate the visual and physiological features
        joint_representation = torch.cat(
            (visual_features, physiological_features), dim=2)

        # Decrease the dimensionality of the joint representation
        joint_representation = self.out_layer_pv(joint_representation)

        # Permute dimension from (batch, seq, feature) to (seq, batch, feature)
        visual_features = visual_features.permute(1, 0, 2)
        physiological_features = physiological_features.permute(1, 0, 2)
        joint_representation = joint_representation.permute(1, 0, 2)

        # Pass the visual, physiological and joint representation features through their respective encoders
        visual_encoded = self.visual_encoder(visual_features)
        physiological_encoded = self.physiological_encoder(
            physiological_features)
        joint_representation_encoded = self.joint_representation_encoder(
            joint_representation)
        # visual_encoded = visual_encoded.permute(1, 0, 2)
        # physiological_encoded = physiological_encoded.permute(1, 0, 2)
        # joint_representation_encoded = joint_representation_encoded.permute(1, 0, 2)

        # Do all the cross-attention between the visual encoded and physio encoded features
        cross_attention_output_v_p, _ = self.cross_attention_v(visual_encoded,
                                                               physiological_encoded,
                                                               physiological_encoded)

        # Do all the cross-attention between the physio encoded and visio encoded features
        cross_attention_output_p_v, _ = self.cross_attention_p(
            physiological_encoded, visual_encoded, visual_encoded)

        # Do all the cross-attention between the joint representation encoded and visio encoded features
        cross_attention_output_pv_v, _ = self.cross_attention_pv(
            joint_representation_encoded, visual_encoded, visual_encoded)

        # Do all the cross-attention between the visio encoded and joint representation encoded features
        cross_attention_output_v_pv, _ = self.cross_attention_v(visual_encoded,
                                                                joint_representation_encoded,
                                                                joint_representation_encoded)

        # Do all the cross-attention between the joint representation encoded and physio encoded features
        cross_attention_output_pv_p, _ = self.cross_attention_pv(
            joint_representation_encoded, physiological_encoded,
            physiological_encoded)

        # Do all the cross-attention between the physio encoded and joint representation encoded features
        cross_attention_output_p_pv, _ = self.cross_attention_p(
            physiological_encoded, joint_representation_encoded,
            joint_representation_encoded)

        if self.output_format == "SELF_ATTEN":
            '''
             --- [Start] Final Attention module ---
            '''
            stack_attention = torch.stack((cross_attention_output_v_p,
                                           cross_attention_output_p_v,
                                           cross_attention_output_pv_v,
                                           cross_attention_output_v_pv,
                                           cross_attention_output_pv_p,
                                           cross_attention_output_p_pv), dim=2)
            stack_attention = stack_attention.permute(1, 0, 2, 3)
            stack_attention_flatten = stack_attention.flatten(0, 1).permute(1, 0, 2)
            stack_attention_flatten = stack_attention_flatten
            b_size = stack_attention.shape[0]
            seq_size = stack_attention.shape[1]
            final_encoded = self.final_visual_encoder(stack_attention_flatten)

            final_attention, _ = self.final_self_attention(final_encoded,
                                                           final_encoded,
                                                           final_encoded)
            final_attention = final_attention.permute(1, 0, 2)
            final_attention_unflatten = final_attention.unflatten(0, (
            b_size, seq_size))

            final_attention_unflatten = final_attention_unflatten[:, :, -1, :]
            # bsz, seq, 512.
            '''
             --- [End] Final Attention module ---
            '''

            return final_attention_unflatten

        elif self.output_format == 'FC':
            # Concatenate Cross-attention outputs
            concat_attention = torch.cat((cross_attention_output_v_p,
                                          cross_attention_output_p_v,
                                          cross_attention_output_pv_v,
                                          cross_attention_output_v_pv,
                                          cross_attention_output_pv_p,
                                          cross_attention_output_p_pv), dim=2)
            out = self.out_layer1(concat_attention)  # bsz, seq, 1024

            return out

        else:
            raise NotImplementedError(self.output_format)


class JMTFusion(nn.Module):
    """ Fuse modalities using JMT Fusion. """

    def __init__(self,
                 num_feats_modality: list,
                 num_out_feats: int = 256):
        """ Instantiate attention fusion instance.

        Args:
            num_feats_modality (list): Number of features per modality.
            num_out_feats (int): Number of output features.
        """

        super(JMTFusion, self).__init__()

        # Encoder blocks
        self.visual_encoder = TransformerEncoderBlock(128, 1,128,1)
        self.audio_encoder = TransformerEncoderBlock(128,1,128,1)
        self.jr_encoder = TransformerEncoderBlock(128,1,128,1)
        self.final_encoder = TransformerEncoderBlock(128*6, 1,128,1)

        # Cross attention blocks
        self.CA_va = nn.MultiheadAttention(128, 1)
        self.CA_av = nn.MultiheadAttention(128, 1)
        self.CA_jra = nn.MultiheadAttention(128, 1)
        self.CA_ajr = nn.MultiheadAttention(128, 1)
        self.CA_vjr = nn.MultiheadAttention(128, 1)
        self.CA_jrv = nn.MultiheadAttention(128, 1)

        self.reduce_feats_dim = nn.Linear(128*2, 128)
        self.augment_audio_feats_dim = nn.Linear(64, 128)


        # Self attention
        self.final_encoder = TransformerEncoderBlock(128,1,128,1)
        self.final_self_attention = nn.MultiheadAttention(128, 1)


    def forward(self, x: list):
        """ Forward pass

        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """

        visual_feats = x['video'].permute(2,0,1) # [length, batch, feature_dim]
        audio_feats = x['vggish'].permute(2,0,1) # [length, batch, feature_dim]

        audio_feats = self.augment_audio_feats_dim(audio_feats) 

        jr_feats = torch.cat((visual_feats, audio_feats), dim=2)
        jr_feats = self.reduce_feats_dim(jr_feats) # [length, batch, feature_dim]

        # Encoders
        encoded_visual = self.visual_encoder(visual_feats)
        encoded_audio = self.audio_encoder(audio_feats)
        encoded_jr = self.jr_encoder(jr_feats)

        # Cross-attention
        CA_va_feats, _ = self.CA_va(encoded_visual, encoded_audio,
                                    encoded_audio)
        CA_av_feats, _ = self.CA_av(encoded_audio, encoded_visual,
                                    encoded_visual)

        CA_jrv_feats, _ = self.CA_jrv(encoded_jr, encoded_visual,
                                      encoded_visual)
        CA_vjr_feats, _ = self.CA_vjr(encoded_visual, encoded_jr, encoded_jr)

        CA_jra_feats, _ = self.CA_jra(encoded_jr, encoded_audio, encoded_audio)
        CA_ajr_feats, _ = self.CA_ajr(encoded_audio, encoded_jr, encoded_jr)

        # Self attention
        stack_attention = torch.stack((CA_va_feats, CA_av_feats, CA_jrv_feats, CA_vjr_feats, CA_jra_feats, CA_ajr_feats), dim=2) # [length, batch, 6, feature_dim]
        length, batch_size, num_stack, feature_dim = stack_attention.shape
        stack_attention = stack_attention.view(-1, num_stack, feature_dim) # [length*batch, 6, feature_dim]
        
        stack_attention_encoded = self.final_encoder(stack_attention)
        out_feats, _ = self.final_self_attention(stack_attention_encoded,stack_attention_encoded,stack_attention_encoded)
        
        out_feats = out_feats.view(length, batch_size, num_stack, feature_dim)
        out_feats = out_feats[:, :, -1, :] # [length, batch, feature_dim]
        
        out_feats = out_feats.permute(1,0,2) # [batch, length, feature_dim]

        return out_feats


class MTFusion(nn.Module):
    """ Fuse modalities using JMT Fusion. """

    def __init__(self,
                 num_feats_modality: list,
                 num_out_feats: int = 256):
        """ Instantiate attention fusion instance.

        Args:
            num_feats_modality (list): Number of features per modality.
            num_out_feats (int): Number of output features.
        """

        super(MTFusion, self).__init__()

        # Encoder blocks
        self.visual_encoder = TransformerEncoderBlock(128, 1,128,1)
        self.audio_encoder = TransformerEncoderBlock(128,1,128,1)
        self.final_encoder = TransformerEncoderBlock(128*6, 1,128,1)

        # Cross attention blocks
        self.CA_va = nn.MultiheadAttention(128, 1)
        self.CA_av = nn.MultiheadAttention(128, 1)

        self.reduce_feats_dim = nn.Linear(128*2, 128)
        self.augment_audio_feats_dim = nn.Linear(64, 128)


        # Self attention
        self.final_encoder = TransformerEncoderBlock(128,1,128,1)
        self.final_self_attention = nn.MultiheadAttention(128, 1)


    def forward(self, x: list):
        """ Forward pass

        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """

        visual_feats = x['video'].permute(2,0,1) # [length, batch, feature_dim]
        audio_feats = x['vggish'].permute(2,0,1) # [length, batch, feature_dim]

        audio_feats = self.augment_audio_feats_dim(audio_feats) 

        # Encoders
        encoded_visual = self.visual_encoder(visual_feats)
        encoded_audio = self.audio_encoder(audio_feats)

        # Cross-attention
        CA_va_feats, _ = self.CA_va(encoded_visual,encoded_audio,encoded_audio)
        CA_av_feats, _ = self.CA_av(encoded_audio,encoded_visual,encoded_visual)

        # Self attention
        stack_attention = torch.stack((CA_va_feats, CA_av_feats), dim=2) # [length, batch, 2, feature_dim]
        length, batch_size, num_stack, feature_dim = stack_attention.shape
        stack_attention = stack_attention.view(-1, num_stack, feature_dim) # [length*batch, 2, feature_dim]
        
        stack_attention_encoded = self.final_encoder(stack_attention)
        out_feats, _ = self.final_self_attention(stack_attention_encoded,stack_attention_encoded,stack_attention_encoded)
        
        out_feats = out_feats.view(length, batch_size, num_stack, feature_dim)
        out_feats = out_feats[:, :, -1, :] # [length, batch, feature_dim]
        
        out_feats = out_feats.permute(1,0,2) # [batch, length, feature_dim]

        return out_feats
        
        
class JMT(nn.Module):
    def __init__(self,
                 task: str,
                 modalities,
                 tcn_settings,
                 backbone_settings,
                 output_dim,
                 root_dir,
                 device,
                 model_name
                 ):
        super().__init__()

        self.device = device
        assert task in constants.TASKS, task
        self.task = task

        self.temporal = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        self.spatial = nn.ModuleDict()

        for modal in modalities:
            self.temporal[modal] = TemporalConvNet(
                num_inputs=tcn_settings[modal]['input_dim'],
                num_channels=tcn_settings[modal]['channel'],
                kernel_size=tcn_settings[modal]['kernel_size']
            )
            self.bn[modal] = BatchNorm1d(tcn_settings[modal]['channel'][-1])


        feas_modalities = [tcn_settings[modal]['channel'][-1] for modal in modalities]
        
        if model_name == constants.JMT:
            self.fuse = JMTFusion(num_feats_modality=feas_modalities,
                                  num_out_feats=128)
        elif model_name == constants.MT:
            self.fuse = MTFusion(num_feats_modality=feas_modalities,
                                 num_out_feats=128)
        else:
            raise NotImplementedError(model_name)

        self.bn1 = BatchNorm1d(128 * 1)
        self.fc1 = Linear(128 * 1, 128 * 1)
        self.fc2 = Linear(128 * 1, output_dim)


        if 'video' in modalities:
            self.root_dir = root_dir
            self.spatial["visual"] = self.load_visual_backbone(
                backbone_settings=backbone_settings)

        if 'logmel' in modalities:
            self.root_dir = root_dir
            self.spatial["audio"] = self.load_audio_backbone(
                backbone_settings=backbone_settings)

    def load_visual_backbone(self, backbone_settings):

        resnet = VisualBackbone(mode='ir', use_pretrained=False)
        state_dict = torch.load(join(self.root_dir,
                                     backbone_settings['visual_state_dict'] + ".pth"),
                                map_location='cpu')
        resnet.load_state_dict(state_dict)

        for param in resnet.parameters():
            param.requires_grad = False


        return resnet

    def load_audio_backbone(self, backbone_settings):

        vggish = AudioBackbone()
        state_dict = torch.load(join(self.root_dir,
                                     backbone_settings['audio_state_dict'] + ".pth"),
                                map_location='cpu')
        vggish.backbone.load_state_dict(state_dict)

        for param in vggish.parameters():
            param.requires_grad = False

        return vggish

    def forward(self, X):

        x = {}

        if 'video' in X:
            batch_size, length, channel, width, height = X['video'].shape
            X['video'] = X['video'].view(-1, channel, width, height) # [batch x length, channel, width, height]
            X['video'] = self.spatial.visual(X['video'])
            _, feature_dim = X['video'].shape
            X['video'] = X['video'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        if 'logmel' in X:
            batch_size, height, length, width = X['logmel'].shape
            X['logmel'] = X['logmel'].permute((0, 2, 3, 1)).contiguous()
            X['logmel'] = X['logmel'].view(-1, width, height) # [batch x length, channel, width, height]
            X['logmel'] = self.spatial.audio(X['logmel'])
            _, feature_dim = X['logmel'].shape
            X['logmel'] = X['logmel'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        for modal in X:
            x[modal] = X[modal].squeeze(1).transpose(1, 2)
            x[modal] = self.temporal[modal](x[modal])
            x[modal] = self.bn[modal](x[modal])

        c = self.fuse(x)
        c = self.fc1(c).transpose(1, 2)
        c = self.bn1(c).transpose(1, 2)
        c = F.leaky_relu(c)
        c = self.fc2(c)

        if self.task == constants.REGRESSION:
            c = torch.tanh(c)

        return c
if __name__ == '__main__':
    model=LFAN(backbone_settings={
        "visual_state_dict": "res50_ir_0.887",
        "audio_state_dict": "vggish"
    },output_dim=8,task='CLASSIFICATION')
    vistual_model=model.load_visual_backbone()
    x = torch.randn(1, 48, 48, 3)
    x=vistual_model(x)
    print(x.shape)

