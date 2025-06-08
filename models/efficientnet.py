import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class EfficientNet(nn.Module):
    def __init__(self, encoder='efficientnet-b0', decoder='unet', weight=None):
        super(EfficientNet, self).__init__()
        if decoder == 'unet':
            self.model = smp.Unet(
                encoder_name=f'timm-{encoder}',
                encoder_weights=weight if weight else None,
                in_channels=3,
                classes=19,
            )
        elif decoder == 'fpn':
            self.model = smp.FPN(
                encoder_name=f'timm-{encoder}',
                encoder_weights=weight if weight else None,
                in_channels=3,
                classes=19
            )
        elif decoder == 'deeplabv3':
            self.model = smp.DeepLabV3(
                encoder_name=f'timm-{encoder}',
                encoder_weights=weight if weight else None,
                in_channels=3,
                classes=19
            )
        elif decoder == 'segformer':
            self.model = smp.Segformer(
                encoder_name=f'timm-{encoder}',
                encoder_weights=weight if weight else None,
                in_channels=3,
                classes=19
            )
        else:
            raise ValueError(f"Unsupported decoder: {decoder}. Supported decoders are 'unet', 'fpn', and 'deeplabv3'.")
            

    def forward(self, x, return_features=False):
        if return_features:
            features = self.model.encoder(x)
            output = self.model.decoder(features)
            output = self.model.segmentation_head(output)
            return output, features
        else:
            return self.model(x)