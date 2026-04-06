import torch
import torch.nn as nn
from .multimodal_slowfast import RGBIRSlowFast
from .Fusion import CrossModalAttention, Reliability
from .ClassificationHead import ClassificationHead

class MultiModalModel(nn.Module):
    def __init__(self, use_reliability=True, rgb_weight=None, ir_weight=None, feature_dim=2304, hidden_dim=512):
        super().__init__()

        self.use_reliability = use_reliability
        self.Slowfast = RGBIRSlowFast(rgb_weight=rgb_weight,ir_weight=ir_weight)

        if self.use_reliability:
            self.reliability = Reliability(feature_dim=feature_dim)  # 根据实际特征维度调整
        else:
            self.reliability = None

        self.cross_attention = CrossModalAttention(feature_dim=feature_dim,hidden_dim=hidden_dim)
        self.classifier = ClassificationHead(feature_dim=2*hidden_dim,hidden_dim=hidden_dim, num_classes=60)  # 根据实际类别数调整

    def forward(self, rgb, ir):
        # ===== 2. 特征提取 =====
        rgb_feat, ir_feat = self.Slowfast(rgb, ir)
        rgb_feat = rgb_feat.mean(dim=[2,3,4])  # 全局平均池化
        ir_feat = ir_feat.mean(dim=[2,3,4])    # 全局平均池化

        # ===== 3. reliability（可选）=====
        if self.use_reliability:
            w_rgb, w_ir = self.reliability(rgb_feat, ir_feat)
        else:
            w_rgb, w_ir = None, None
        # ===== 4. 融合 =====
        fused = self.cross_attention(rgb_feat, ir_feat, w_rgb, w_ir)

        # ===== 5. 分类 =====
        out = self.classifier(fused)

        return out