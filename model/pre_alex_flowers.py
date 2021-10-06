import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


class Three_Task_Model(nn.Module):
    def __init__(self, is_fine_tuning, is_GB, is_WWA, num_classes, num_classes2, is_scale=True):
        super(Three_Task_Model, self).__init__()
        self.is_scale = is_scale
        self.is_GB = is_GB
        self.is_WWA = is_WWA
        self.pretrained_model = ParseNet_Flowers(is_fine_tuning, is_GB, is_WWA)

        if not is_fine_tuning:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.classifier2 = nn.Linear(4096 + 2048, num_classes2)

        if is_GB:  # WWA-CNN from "Growing a Brain" paper.
            if is_WWA:
                self.augClassifier2_1 = nn.Sequential(
                    nn.Linear(256 * 6 * 6, 1024),
                    nn.ReLU(inplace=True),
                )
                self.augClassifier2_2 = nn.Sequential(
                    nn.Linear(4096 + 1024, 2048),
                    nn.ReLU(inplace=True),
                )
            else:
                self.augClassifier2 = nn.Sequential(
                    nn.Linear(4096, 2048),
                    nn.ReLU(inplace=True),
                )
        else:
            if is_WWA:
                self.augClassifier2 = nn.Linear(256 * 6 * 6 + 4096, 2048)  # Both features and fc6.
            else:
                self.augClassifier2 = nn.Linear(4096, 2048)  # Only fc6.

        if self.is_scale:
            self.scale2 = nn.Parameter(torch.randn(4096 + 2048).cuda(), requires_grad=True)  # scale = {h^k, h^k+}

    def forward(self, x):
        features, fc6, fc7, prev_outputs, prev_augOutputs = self.pretrained_model(x)

        if self.is_GB:
            if self.is_WWA:
                fc6_plus2 = self.augClassifier2_1(features)
                fc7_plus2 = self.augClassifier2_2(torch.cat((fc6, fc6_plus2), 1))  # Both features and fc6.
            else:
                fc7_plus2 = self.augClassifier2(fc6)  # Only fc6.
        else:
            if self.is_WWA:
                fc7_plus2 = self.augClassifier2(torch.cat((features, fc6), 1))  # Both features and fc6.
            else:
                fc7_plus2 = self.augClassifier2(fc6)  # Only fc6.

        if self.is_scale:
            # ParseNet Normalization.
            norm_fc7 = fc7.div(torch.norm(fc7, 2, 1, keepdim=True).expand_as(fc7))
            norm_fc7_plus2 = fc7_plus2.div(torch.norm(fc7_plus2, 2, 1, keepdim=True).expand_as(fc7_plus2))
            outputs = self.classifier2(torch.cat((self.scale2[:4096].expand_as(norm_fc7) * norm_fc7,
                                                  self.scale2[4096:].expand_as(norm_fc7_plus2) * norm_fc7_plus2), 1))

            zero_inputs = Variable(torch.zeros(norm_fc7.size()).cuda(), requires_grad=False)
            augOutputs = self.classifier2(torch.cat((zero_inputs,
                                                     self.scale2[4096:].expand_as(norm_fc7_plus2) * norm_fc7_plus2), 1))
        else:
            outputs = self.classifier2(torch.cat((fc7, fc7_plus2), 1))

        return features, fc6, fc7, prev_outputs, prev_augOutputs, outputs, augOutputs


class ParseNet_Flowers(nn.Module):
    def __init__(self, is_fine_tuning, is_GB, is_WWA, num_classes=102, is_scale=True):
        super(ParseNet_Flowers, self).__init__()
        self.is_scale = is_scale
        self.is_GB = is_GB
        self.is_WWA = is_WWA
        pretrained_model = models.alexnet(pretrained=True)

        if not is_fine_tuning:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.features = pretrained_model.features
        self.fc6 = nn.Sequential(*list(pretrained_model.classifier.children())[:3])
        self.fc7 = nn.Sequential(*list(pretrained_model.classifier.children())[3:6])
        self.classifier = nn.Linear(4096 + 2048, num_classes)

        if is_GB:  # WWA-CNN from "Growing a Brain" paper.
            if is_WWA:
                self.augClassifier1_1 = nn.Sequential(
                    nn.Linear(256 * 6 * 6, 1024),
                    nn.ReLU(inplace=True),
                )
                self.augClassifier1_2 = nn.Sequential(
                    nn.Linear(4096 + 1024, 2048),
                    nn.ReLU(inplace=True),
                )
            else:
                self.augClassifier = nn.Sequential(
                    nn.Linear(4096, 2048),
                    nn.ReLU(inplace=True),
                )
        else:
            if is_WWA:
                self.augClassifier = nn.Linear(256 * 6 * 6 + 4096, 2048)  # Both features and fc6.
            else:
                self.augClassifier = nn.Linear(4096, 2048)  # Only fc6.

        if self.is_scale:
            self.scale = nn.Parameter(torch.randn(4096 + 2048).cuda(), requires_grad=True)  # scale = {h^k, h^k+}

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        fc6 = self.fc6(features)
        fc7 = self.fc7(fc6)
        if self.is_GB:
            if self.is_WWA:
                fc6_plus = self.augClassifier1_1(features)
                fc7_plus = self.augClassifier1_2(torch.cat((fc6, fc6_plus), 1))  # Both features and fc6.
            else:
                fc7_plus = self.augClassifier(fc6)  # Only fc6.
        else:
            if self.is_WWA:
                fc7_plus = self.augClassifier(torch.cat((features, fc6), 1))  # Both features and fc6.
            else:
                fc7_plus = self.augClassifier(fc6)  # Only fc6.

        if self.is_scale:
            # ParseNet Normalization.
            norm_fc7 = fc7.div(torch.norm(fc7, 2, 1, keepdim=True).expand_as(fc7))
            norm_fc7_plus = fc7_plus.div(torch.norm(fc7_plus, 2, 1, keepdim=True).expand_as(fc7_plus))
            outputs = self.classifier(torch.cat((self.scale[:4096].expand_as(norm_fc7) * norm_fc7,
                                                 self.scale[4096:].expand_as(norm_fc7_plus) * norm_fc7_plus), 1))

            zero_inputs = Variable(torch.zeros(norm_fc7.size()).cuda(), requires_grad=False)
            augOutputs = self.classifier(torch.cat((zero_inputs,
                                                    self.scale[4096:].expand_as(norm_fc7_plus) * norm_fc7_plus), 1))
        else:
            outputs = self.classifier(torch.cat((fc7, fc7_plus), 1))

        return features, fc6, fc7, outputs, augOutputs


class PNN_Three_Task_Model(nn.Module):
    def __init__(self, is_fine_tuning, num_classes=102, num_classes2=200):
        super(PNN_Three_Task_Model, self).__init__()
        pretrained_model = models.alexnet(pretrained=True)

        if not is_fine_tuning:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.pretrained_model = PNN_AlexNet_Flowers(is_fine_tuning)

        self.features = pretrained_model.features
        self.fc6 = nn.Sequential(*list(pretrained_model.classifier.children())[:3])
        self.fc7 = nn.Sequential(*list(pretrained_model.classifier.children())[3:6])
        self.classifier = nn.Linear(4096 * 2, num_classes2)

        self.fc6._modules['1'] = nn.Linear(256 * 6 * 6 * 2, 4096)
        self.fc7._modules['1'] = nn.Linear(4096 * 2, 4096)

    def forward(self, x):
        prev_features, prev_fc6, prev_fc7, _ = self.pretrained_model(x)

        features = self.features(x)
        features = features.view(features.size(0), -1)

        fc6 = self.fc6(torch.cat((prev_features, features), 1))
        fc7 = self.fc7(torch.cat((prev_fc6, fc6), 1))

        outputs = self.classifier(torch.cat((prev_fc7, fc7), 1))

        return prev_features, prev_fc6, prev_fc7, features, fc6, fc7, outputs


class PNN_AlexNet_Flowers(nn.Module):
    def __init__(self, is_fine_tuning, num_classes=102):
        super(PNN_AlexNet_Flowers, self).__init__()
        pretrained_model = models.alexnet(pretrained=True)

        if not is_fine_tuning:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.features = pretrained_model.features
        self.fc6 = nn.Sequential(*list(pretrained_model.classifier.children())[:3])
        self.fc7 = nn.Sequential(*list(pretrained_model.classifier.children())[3:6])
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        fc6 = self.fc6(features)
        fc7 = self.fc7(fc6)

        outputs = self.classifier(fc7)

        return features, fc6, fc7, outputs


class EWC_Three_Task_Model(nn.Module):
    def __init__(self, is_fine_tuning, num_classes=102, num_classes2=200):
        super(EWC_Three_Task_Model, self).__init__()
        pretrained_model = models.alexnet(pretrained=True)

        if not is_fine_tuning:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.pretrained_model = EWC_AlexNet_Flowers(is_fine_tuning)
        self.classifier = nn.Linear(4096, num_classes2)

    def forward(self, x):
        fc7, prev_outputs = self.pretrained_model(x)

        outputs = self.classifier(fc7)

        return fc7, prev_outputs, outputs


class EWC_AlexNet_Flowers(nn.Module):
    def __init__(self, is_fine_tuning, num_classes=102):
        super(EWC_AlexNet_Flowers, self).__init__()
        pretrained_model = models.alexnet(pretrained=True)

        if not is_fine_tuning:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.features = pretrained_model.features
        self.fc6 = nn.Sequential(*list(pretrained_model.classifier.children())[:3])
        self.fc7 = nn.Sequential(*list(pretrained_model.classifier.children())[3:6])
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        fc6 = self.fc6(features)
        fc7 = self.fc7(fc6)

        outputs = self.classifier(fc7)

        return fc7, outputs


class Pre_AlexNet_Flowers(nn.Module):
    def __init__(self, is_fine_tuning, num_classes=102):
        super(Pre_AlexNet_Flowers, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True)

        if not is_fine_tuning:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        self.pretrained_model.classifier._modules['6'] = \
            nn.Linear(self.pretrained_model.classifier._modules['6'].in_features, num_classes)

    def forward(self, x):
        return self.pretrained_model(x)


def pre_alex_flowers(pretrained=False, is_fine_tuning=False, memory=True, is_GB=False, is_WWA=True,
                     PNN=False, EWC=False, num_classes=102, num_classes2=200, task_index=1, **kwargs):
    """Pre-trained AlexNet for Flowers dataset.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet and Flowers, sequentially.
        is_fine_tuning (bool): If True, returns a fine-tuning model, otherwise a feature-extraction model.
        memory (bool): If True, returns a ParseNet model pre-trained on ImageNet and Flowers, sequentially.
    """
    if memory:
        if task_index is 3:  # [num_classes, num_classes2] should be defined.
            model = Three_Task_Model(is_fine_tuning, is_GB, is_WWA, num_classes, num_classes2, **kwargs)
        else:
            model = ParseNet_Flowers(is_fine_tuning, is_GB, is_WWA, **kwargs)
    else:
        if PNN:
            if task_index is 3:
                model = PNN_Three_Task_Model(is_fine_tuning, num_classes, num_classes2)
            else:
                model = PNN_AlexNet_Flowers(is_fine_tuning, **kwargs)
        elif EWC:
            if task_index is 3:
                model = EWC_Three_Task_Model(is_fine_tuning, num_classes, num_classes2)
            else:
                model = EWC_AlexNet_Flowers(is_fine_tuning, **kwargs)
        else:
            model = Pre_AlexNet_Flowers(is_fine_tuning, **kwargs)

    if pretrained:
        if memory:
            if task_index is 3:
                if num_classes2 == 67:
                    checkpoint = torch.load('mem_flowers_scenes_model')
                elif num_classes2 == 200:
                    checkpoint = torch.load('mem_flowers_birds_model')
            else:
                checkpoint = torch.load('mem_flowers_model')
        else:
            if PNN:
                if task_index is 3:
                    if num_classes2 == 67:
                        checkpoint = torch.load('pnn_flowers_scenes_model')
                    elif num_classes2 == 200:
                        checkpoint = torch.load('pnn_flowers_birds_model')
                else:
                    checkpoint = torch.load('pnn_alex_flowers_model')
            if EWC:
                if task_index is 3:
                    if num_classes2 == 67:
                        checkpoint = torch.load('ewc_flowers_scenes_model')
                    elif num_classes2 == 200:
                        checkpoint = torch.load('ewc_flowers_birds_model')
                else:
                    checkpoint = torch.load('ewc_alex_flowers_model')
            else:
                checkpoint = torch.load('pre_alex_flowers_model')
        model.load_state_dict(checkpoint['state_dict'])  # Containing ['bias', 'weight'].
    return model
