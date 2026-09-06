import torch.nn as nn
import segmentation_models_pytorch as smp

class StudentModel():
    def __init__(self, encoder='timm-efficientnet-b0', weight='imagenet', dropout=0.0):
        self.encoder = encoder
        self.weight = weight
        
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weight if weight else None,
            in_channels=3,
            classes=19,
        )
        model.decoder.dropout = nn.Dropout(dropout)

        self.feature_hook = []
        self.features = []
        
        # Register hooks to capture features
        for encoder_stage in self.model.encoder.stages:
            hook = encoder_stage.register_forward_hook(self.save_features)
            self.feature_hooks.append(hook)
    
    def save_features(self, module, input, output):
        self.features.append(output)
        
    def forward(self, x):
        self.features = []
        output = self.model(x)
        return output, self.features