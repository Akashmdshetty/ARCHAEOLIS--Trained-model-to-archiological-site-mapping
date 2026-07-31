import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        # Pad upsampled x to match skip dimensions if necessary
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MultiTaskArchaeologist(nn.Module):
    """
    Multi-task head for:
    1. Segmentation (Ruins/Vegetation)
    2. Detection (Artifact Bounding Boxes)
    3. Regression (Erosion Risk Heatmap)
    4. Land Fault Detection (Purple Mask)
    """
    def __init__(self, encoder_channels=[64, 128, 256, 512]):
        super(MultiTaskArchaeologist, self).__init__()
        
        # 1. Segmentation Decoder (U-Net style)
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        
        # Final Segmentation Head (3 classes: Background, Ruins, Vegetation)
        self.seg_head = nn.Conv2d(64, 3, kernel_size=1)
        
        # 2. Erosion Risk Head (Heatmap)
        self.erosion_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. Artifact Detection Head (Simplified: Center Heatmap + Size)
        # In a real scenario, this would be a more complex head like YOLO or Faster R-CNN
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 5, kernel_size=1) # [confidence, x, y, w, h]
        )
        
        # 4. Land Fault Head
        self.fault_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, intermediate_features):
        x1, x2, x3, x4 = intermediate_features
        
        # Decoding for Segmentation and Erosion
        d3 = self.dec4(x4, x3)
        d2 = self.dec3(d3, x2)
        d1 = self.dec2(d2, x1)
        
        # Upsample d1 to input size (assuming 224x224 and encoder output at layer1 is 56x56)
        d_final = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=True)
        
        seg_mask = self.seg_head(d_final)
        erosion_heatmap = self.erosion_head(d_final)
        fault_mask = self.fault_head(d_final)
        
        # Detection output from deep features
        detection = self.detection_head(x4)
        
        return {
            'segmentation': seg_mask,
            'erosion': erosion_heatmap,
            'detection': detection,
            'faults': fault_mask
        }
