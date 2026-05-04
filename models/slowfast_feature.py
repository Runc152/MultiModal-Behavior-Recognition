import torch
import torch.nn as nn
from pytorchvideo.models.slowfast import create_slowfast


class SlowFastFeatureExtractor(nn.Module):

    def __init__(self, weight_path=None, device="cuda", freeze=False):

        super().__init__()

        self.device = torch.device(device)

        self.model = create_slowfast(
            model_num_class=400,
            slowfast_channel_reduction_ratio=8,
            slowfast_conv_channel_fusion_ratio=2,
            head_pool=torch.nn.AdaptiveAvgPool3d,  # 改为自适应池化，支持任意帧数
            head_output_size=(1, 1, 1),
        )


        # 加载权重
        if weight_path is not None:

            checkpoint = torch.load(weight_path, map_location=self.device)

            if "model_state" in checkpoint:
                checkpoint = checkpoint["model_state"]

            self.model.load_state_dict(checkpoint, strict=False)

        #去掉分类头
        self.model.blocks[-1] = nn.Identity()

        # 冻结参数
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model = self.model.to(self.device)

    def forward(self, x):

        x = [i.to(self.device) for i in x]

        feat = self.model(x)

        return feat