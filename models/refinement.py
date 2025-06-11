import torch
import torch.nn as nn

class EncoderPRN(nn.Module):
    def __init__(self, in_channels, num_classes, mid_channels=32):
        super().__init__()
        self.prn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1)
        )
    def forward(self, images, teacer_logits):
        prn_in = torch.cat([images, teacer_logits], dim=1)
        refined_logits = self.prn(prn_in)
        return refined_logits