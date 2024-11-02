import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
    
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class KeypointDetectionModel(nn.Module):
    def __init__(self, num_keypoints=17):
        super(KeypointDetectionModel, self).__init__()
        self.backbone = models.resnet50()
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.conv = nn.Conv2d(2048, num_keypoints, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=6, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.upsample(x)
        return x

if __name__ == "__main__":
    model = KeypointDetectionModel()
    sample_input = torch.randn(1, 3, 256, 256)
    output = model(sample_input)
    print(output.shape)  # Should be [1, num_keypoints, 32, 32]
    # print(model)  # Should print the model architecture