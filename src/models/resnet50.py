import torch
import torch.nn as nn
import timm
import sys


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, num_classes=9):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes
        """
        def resnet50(pretrained: bool = False, **kwargs) -> ResNet:
            model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
            return _create_resnet('resnet50', pretrained, **dict(model_args, **kwargs))
        """

        self.feature_layers = nn.Sequential(*list(self.model.children())[:-3])  # no linear layers  —2

    def get_grad_cam_target_layer(self):
        return self.model.layer4[-1]

    def forward(self, x):
        return self.model(x)
    
    def get_feature(self, x, feat_type='top', flatten=True):
        if feat_type == 'x':
            if flatten == False:
                return x
            return x.view(x.size(0), -1)
        elif feat_type == 'top':
            feat = self.feature_layers(x)
            if flatten == False:
                return feat
            return feat.view(feat.size(0), -1)
        else:
            sys.exit(1)
    
    def get_grad_loss(self, x):
        return 
