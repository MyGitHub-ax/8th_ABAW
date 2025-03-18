from models.arcface_model import Backbone


import torch
from torch import nn


from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            # nn.ReLU(True)
        )

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        return x


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


class VGGish(VGG):
    def __init__(self):
        super().__init__(make_layers())

    def forward(self, x, fs=None):
        x = torch.tensor(x)[:, None, :, :].float()
        x = VGG.forward(self, x)
        return x


class VisualBackbone(nn.Module):
    """主要用于特征提取和分类任务。它通过预训练的骨干网络（Backbone）提取特征，并通过自定义的输出层和分类层进行微调或分类"""
    def __init__(self, input_channels=3, num_classes=8, use_pretrained=True, state_dict_path="", mode="ir",
                 embedding_dim=512):
        """功能：初始化模型参数，构建骨干网络、输出层和分类层。

            参数：

            input_channels：输入图像通道数（默认 3，RGB）。

            num_classes：分类任务类别数（默认 8）。

            use_pretrained：是否使用预训练权重（默认是）。

            state_dict_path：预训练权重路径。

            mode：骨干网络模式（如残差块类型 "ir"）。

            embedding_dim：嵌入向量维度（默认 512）"""
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, num_layers=50, drop_ratio=0.4, mode=mode)

        """初始化骨干网络（如 ResNet-50），配置输入通道、层数、Dropout 比例和模式。"""
        if use_pretrained:
            #从指定路径加载预训练权重到 CPU。
            state_dict = torch.load(state_dict_path, map_location='cpu')

            if "backbone" in list(state_dict.keys())[0]:

                self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                        Dropout(0.4),
                                                        Flatten(),
                                                        Linear(embedding_dim * 5 * 5, embedding_dim),
                                                        BatchNorm1d(embedding_dim))

                new_state_dict = {}
                for key, value in state_dict.items():

                    if "logits" not in key:
                        new_key = key[9:]
                        new_state_dict[new_key] = value

                self.backbone.load_state_dict(new_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)
            #冻结骨干网络所有参数（包括输出层），阻止其参与训练
            for param in self.backbone.parameters():
                param.requires_grad = False
        # 重定义输出层,此操作覆盖了加载的预训练权重，导致输出层参数被重新初始化
        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

        from torch.nn.init import xavier_uniform_, constant_

        for m in self.backbone.output_layer.modules():
            if isinstance(m, nn.Linear):
                m.weight = xavier_uniform_(m.weight)
                m.bias = constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        self.logits.weight = xavier_uniform_(self.logits.weight)
        self.logits.bias = constant_(self.logits.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def extract(self, x):
        x = self.backbone(x)
        return x


class AudioBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = VGGish()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x, extract_vggish=False):
        x = self.backbone(x)

        return x