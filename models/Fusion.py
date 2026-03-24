import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合模块，输入 RGB 和 IR 的特征以及它们的权重，输出融合后的特征
    """

    def __init__(self, feature_dim=2304, hidden_dim=1024):

        super().__init__()

        self.q = nn.Linear(feature_dim, hidden_dim)
        self.k = nn.Linear(feature_dim, hidden_dim)
        self.v = nn.Linear(feature_dim, hidden_dim)

        self.rel_scale = nn.Parameter(torch.tensor(0.5))  # 初始值可以小一些

        self.out = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, rgb_feat, ir_feat, w_rgb, w_ir):

        tokens = torch.stack([rgb_feat, ir_feat], dim=1)
        # B × 2 × C

        Q = self.q(tokens)
        K = self.k(tokens)
        V = self.v(tokens)

        attn = torch.matmul(Q, K.transpose(-2,-1))
        attn = attn / (Q.shape[-1] ** 0.5)

        reliability = torch.cat([w_rgb, w_ir], dim=1)
        reliability = torch.log(reliability + 1e-6)           # log(probability)
        reliability = reliability.unsqueeze(1)

        attn = attn + self.rel_scale * reliability

        attn = F.softmax(attn, dim=-1)

        fused = torch.matmul(attn, V)

        fused = fused.flatten(start_dim=1)

        fused = self.out(fused)

        return fused


class Reliability(nn.Module):
    """
    可靠性评估模块，输入 RGB 和 IR 的特征，输出两者的权重
    """

    def __init__(self, feature_dim=2304):

        super().__init__()

        input_dim = feature_dim*2 + 4

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 577),
            nn.ReLU(),
            nn.Linear(577, 2)
        )

    def forward(self, rgb_feat, ir_feat):

        var_rgb, ent_rgb = feature_statistics(rgb_feat)
        var_ir, ent_ir = feature_statistics(ir_feat)

        x = torch.cat([
            rgb_feat,
            ir_feat,
            var_rgb,
            var_ir,
            ent_rgb,
            ent_ir
        ], dim=1)

        score = self.mlp(x)

        weight = torch.softmax(score, dim=1)

        w_rgb = weight[:,0:1]
        w_ir = weight[:,1:2]

        return w_rgb, w_ir

def feature_statistics(feat):

    variance = torch.var(feat, dim=1, keepdim=True)

    p = torch.softmax(feat, dim=1)

    entropy = -torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)

    return variance, entropy