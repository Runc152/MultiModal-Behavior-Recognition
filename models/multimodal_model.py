import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_slowfast import RGBIRSlowFast
from .ClassificationHead import ClassificationHead


# ──────────────────────────────────────────────
# Temporal aggregator (unchanged)
# ──────────────────────────────────────────────
class TemporalAggregator(nn.Module):
    """
    Lightweight temporal aggregation.
    Spatial pooling first → learnable depthwise temporal conv → adaptive pool → [B, C].
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            feature_dim, feature_dim,
            kernel_size=5, padding=2, groups=feature_dim, bias=False,
        )
        self.pw_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, C, T]
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.act(x)
        x = self.pool(x)        # [B, C, 1]
        return x.squeeze(-1)    # [B, C]


# ──────────────────────────────────────────────
# Pixel quality network
# ──────────────────────────────────────────────
class PixelQualityNet(nn.Module):
    """
    Lightweight quality assessment from raw video frames.

    Analyses low-level statistics (brightness, texture, edge density)
    WITHOUT learning semantics — two spatial conv layers only.

    Input:  [B, C, T, H, W]  — raw-ish frames (normalized, any resolution)
    Output: [B, Q]           — quality descriptor per sample
    """

    def __init__(self, in_channels, quality_dim=64):
        super().__init__()
        # Spatial analysis: two conv layers capture luminance + texture
        # stride=4 each → 1/16 spatial resolution, temporal stride=1
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, 16,
                      kernel_size=(1, 7, 7), stride=(1, 4, 4), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32,
                      kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 2, 2),
                      bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        # Collapse spatial dims, keep temporal
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Aggregate temporal with learned conv
        self.temporal = nn.Sequential(
            nn.Conv1d(32, quality_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.spatial(x)                              # [B, 32, T, H/16, W/16]
        x = self.spatial_pool(x)                         # [B, 32, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1)                    # [B, 32, T]
        x = self.temporal(x)                             # [B, Q, 1]
        return x.squeeze(-1)                             # [B, Q]


# ──────────────────────────────────────────────
# Quality-guided fusion
# ──────────────────────────────────────────────
class QualityGuidedFusion(nn.Module):
    """
    Fuse backbone features using pixel-level quality signals.

    Quality weights come from PixelQualityNet (independent of backbone),
    so the model can detect degraded modality BEFORE the features are computed.

    quality_rgb, quality_ir → MLP → [w_rgb, w_ir]
    rgb_feat,  ir_feat      → Linear → v_rgb, v_ir
    fused = w_rgb * v_rgb + w_ir * v_ir
    """

    def __init__(self, feature_dim, hidden_dim, quality_dim=64):
        super().__init__()
        # Quality → modality weight
        self.weight_net = nn.Sequential(
            nn.Linear(quality_dim * 2, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )
        # Feature projection
        self.v_rgb = nn.Linear(feature_dim, hidden_dim)
        self.v_ir  = nn.Linear(feature_dim, hidden_dim)

    def forward(self, rgb_feat, ir_feat, quality_rgb, quality_ir):
        # rgb_feat, ir_feat: [B, C]
        # quality_rgb, quality_ir: [B, Q]
        q = torch.cat([quality_rgb, quality_ir], dim=1)  # [B, 2Q]
        weights = F.softmax(self.weight_net(q), dim=1)    # [B, 2]
        w_rgb = weights[:, 0:1]   # [B, 1]
        w_ir  = weights[:, 1:2]   # [B, 1]

        v_rgb = self.v_rgb(rgb_feat)   # [B, H]
        v_ir  = self.v_ir(ir_feat)     # [B, H]

        fused = w_rgb * v_rgb + w_ir * v_ir   # [B, H]
        return fused, w_rgb, w_ir


# ──────────────────────────────────────────────
# Multi-modal model
# ──────────────────────────────────────────────
class MultiModalModel(nn.Module):
    def __init__(
        self,
        model_type='slowfast',
        model_variant=None,
        rgb_weight=None,
        ir_weight=None,
        feature_dim=2304,
        hidden_dim=512,
        num_classes=60,
        quality_dim=64,
    ):
        super().__init__()

        self.model_type = model_type.lower()

        # ── backbone ──
        self.Slowfast = RGBIRSlowFast(
            model_type=model_type,
            model_variant=model_variant,
            rgb_weight=rgb_weight,
            ir_weight=ir_weight,
        )

        # ── pixel quality (lightweight, backbone-independent) ──
        self.pixel_quality_rgb = PixelQualityNet(in_channels=3,  quality_dim=quality_dim)
        self.pixel_quality_ir  = PixelQualityNet(in_channels=1,  quality_dim=quality_dim)

        # ── per-modality temporal aggregation ──
        self.temporal_rgb = TemporalAggregator(feature_dim)
        self.temporal_ir  = TemporalAggregator(feature_dim)

        # ── quality-guided fusion ──
        self.fusion = QualityGuidedFusion(feature_dim, hidden_dim, quality_dim)

        # ── classifier ──
        self.classifier = ClassificationHead(
            input_dim=hidden_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )

    def forward(self, rgb, ir):
        # ── Step 0: extract fast frames for pixel quality ──
        # rgb: [rgb_slow, rgb_fast], ir: [ir_slow, ir_fast]
        rgb_fast = rgb[1]                      # [B, 3, T_fast, H, W]
        ir_fast  = ir[1]                       # [B, 1 or 3, T_fast, H, W]
        # Ensure single-channel IR (backbone may have repeated it to 3)
        if ir_fast.shape[1] != 1:
            ir_fast = ir_fast[:, :1, ...]       # [B, 1, T_fast, H, W]

        quality_rgb = self.pixel_quality_rgb(rgb_fast)   # [B, Q]
        quality_ir  = self.pixel_quality_ir(ir_fast)     # [B, Q]

        # ── Step 1: backbone ──
        rgb_feat, ir_feat = self.Slowfast(rgb, ir)       # [B, C, T, h, w]

        # ── Step 2: spatial pool (keep temporal) ──
        rgb_feat = rgb_feat.mean(dim=[3, 4])              # [B, C, T]
        ir_feat  = ir_feat.mean(dim=[3, 4])

        # ── Step 3: temporal aggregation ──
        rgb_feat = self.temporal_rgb(rgb_feat)            # [B, C]
        ir_feat  = self.temporal_ir(ir_feat)

        # ── Step 4: quality-guided fusion ──
        fused, w_rgb, w_ir = self.fusion(
            rgb_feat, ir_feat, quality_rgb, quality_ir)   # [B, H]

        # ── Step 5: classify ──
        out = self.classifier(fused)                      # [B, num_classes]

        return out
