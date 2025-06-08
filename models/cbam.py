import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, model='student'):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.model = model
        if model == 'student':
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        
        linear_avg = self.mlp(avg_pool)
        linear_max = self.mlp(max_pool)
        
        channel_att_sum = linear_avg + linear_max

        if self.model == 'student':
            scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        else:
            scale = F.sigmoid(channel_att_sum)
        return x * scale
    

class SpatialGate(nn.Module):
    def __init__(self, model='student', kernel_size=7):
        super(SpatialGate, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1) // 2, bias=False),
            nn.BatchNorm2d(1),
        )
        self.model = model

    def forward(self, x):
        compress = torch.cat((
            torch.max(x, dim=1)[0].unsqueeze(1), 
            torch.mean(x, dim=1).unsqueeze(1)
        ), dim=1)
        
        if self.model == 'student':
            spatial_att = self.spatial(compress)
        else:
            spatial_att = torch.sum(compress, dim=1, keepdim=True)
            
        scale = F.sigmoid(spatial_att)
        return x * scale
    
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, model='student'):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio, model)
        self.spatial_gate = SpatialGate(model)

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x