import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# CrossModalAttention / Reliability (legacy)
# ──────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    """
    Cross-modal attention fusion (2-token).
    Input: pooled features [B, feature_dim] per modality.
    """

    def __init__(self, feature_dim=2304, hidden_dim=1024):
        super().__init__()
        self.q = nn.Linear(feature_dim, hidden_dim)
        self.k = nn.Linear(feature_dim, hidden_dim)
        self.v = nn.Linear(feature_dim, hidden_dim)
        self.rel_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb_feat, ir_feat, w_rgb, w_ir):
        tokens = torch.stack([rgb_feat, ir_feat], dim=1)  # [B, 2, C]
        Q = self.q(tokens)
        K = self.k(tokens)
        V = self.v(tokens)
        attn = torch.matmul(Q, K.transpose(-2, -1))
        attn = attn / (Q.shape[-1] ** 0.5)
        if w_ir is not None and w_rgb is not None:
            reliability = torch.cat([w_rgb, w_ir], dim=1)
            reliability = torch.log(reliability + 1e-6)
            reliability = reliability.unsqueeze(1)
            attn = attn + self.rel_scale * reliability
        attn = F.softmax(attn, dim=-1)
        fused = torch.matmul(attn, V)
        fused = fused.flatten(start_dim=1)
        return fused


class Reliability(nn.Module):
    """
    Global reliability scoring (single weight per modality).
    Input: pooled features [B, feature_dim] per modality.
    """

    def __init__(self, feature_dim=2304):
        super().__init__()
        input_dim = feature_dim * 2 + 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 577),
            nn.ReLU(),
            nn.Linear(577, 2)
        )

    def forward(self, rgb_feat, ir_feat):
        var_rgb, ent_rgb = feature_statistics(rgb_feat)
        var_ir, ent_ir = feature_statistics(ir_feat)
        x = torch.cat([rgb_feat, ir_feat, var_rgb, var_ir, ent_rgb, ent_ir], dim=1)
        score = self.mlp(x)
        weight = torch.softmax(score, dim=1)
        w_rgb = weight[:, 0:1]
        w_ir = weight[:, 1:2]
        return w_rgb, w_ir


def feature_statistics(feat):
    variance = torch.var(feat, dim=1, keepdim=True)
    p = torch.softmax(feat, dim=1)
    entropy = -torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return variance, entropy


# ──────────────────────────────────────────────
# Per‑frame Quality Gate (new)
# ──────────────────────────────────────────────
class PerFrameQualityGate(nn.Module):
    """
    Dynamic per‑frame quality assessment and modal fusion.

    Instead of one global weight per modality (Reliability),
    this module outputs a weight *per frame*, allowing the model
    to say: "frames 0‑10 look good in RGB, frames 30‑47 are black,
    so switch to IR for those."

    Input:  [B, C, T]  per modality (spatially pooled backbone output)
    Output: [B, C, T]  fused features (per‑frame weighted combination)
            + per‑frame weights for inspection
    """

    def __init__(self, feature_dim):
        super().__init__()
        # --- quality scoring branch ---
        # looks at both modalities jointly with local temporal context
        self.quality_conv = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim // 2, 2, kernel_size=1),   # (score_rgb, score_ir)
        )

        # --- value projection ---
        self.value_proj = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, rgb_feat, ir_feat):
        # rgb_feat, ir_feat: [B, C, T]

        # quality scores with local temporal context  (kernel=5 → ~250 ms)
        concat = torch.cat([rgb_feat, ir_feat], dim=1)   # [B, 2C, T]
        scores = self.quality_conv(concat)                # [B, 2, T]

        # per‑frame softmax → weights sum to 1 at each time step
        weights = F.softmax(scores, dim=1)               # [B, 2, T]
        w_rgb = weights[:, 0:1, :]                        # [B, 1, T]
        w_ir  = weights[:, 1:2, :]                        # [B, 1, T]

        # project values and fuse
        v_rgb = self.value_proj(rgb_feat)                 # [B, C, T]
        v_ir  = self.value_proj(ir_feat)                  # [B, C, T]

        fused = w_rgb * v_rgb + w_ir * v_ir               # [B, C, T]

        return fused, w_rgb, w_ir
