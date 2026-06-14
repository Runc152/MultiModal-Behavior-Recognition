import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.models.x3d import create_x3d


# ──────────────────────────────────────────────
# X3D variants metadata
# ──────────────────────────────────────────────
X3D_META = {
    'xs': {'feature_dim': 192, 'clip_length': 4,  'crop_size': 160,
           'width_factor': 2.0, 'depth_factor': 2.2},
    's':  {'feature_dim': 192, 'clip_length': 13, 'crop_size': 160,
           'width_factor': 2.0, 'depth_factor': 2.2},
    'm':  {'feature_dim': 192, 'clip_length': 16, 'crop_size': 224,
           'width_factor': 2.0, 'depth_factor': 2.2},
    'l':  {'feature_dim': 480, 'clip_length': 16, 'crop_size': 312,
           'width_factor': 5.0, 'depth_factor': 5.0},
}


# ──────────────────────────────────────────────
# SlowFast
# ──────────────────────────────────────────────
class SlowFastFeatureExtractor(nn.Module):

    def __init__(self, weight_path=None, device="cuda", freeze=False):

        super().__init__()

        self.device = torch.device(device)

        self.model = create_slowfast(
            model_num_class=400,
            slowfast_channel_reduction_ratio=8,
            slowfast_conv_channel_fusion_ratio=2,
            head_pool=torch.nn.AdaptiveAvgPool3d,
            head_output_size=(1, 1, 1),
        )

        # load pretrained weights
        if weight_path is not None:
            checkpoint = torch.load(weight_path, map_location=self.device)
            if "model_state" in checkpoint:
                checkpoint = checkpoint["model_state"]
            self.model.load_state_dict(checkpoint, strict=False)

        # remove classification head
        self.model.blocks[-1] = nn.Identity()

        # freeze if needed
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model = self.model.to(self.device)

    def forward(self, x):
        x = [i.to(self.device) for i in x]
        feat = self.model(x)
        return feat


# ──────────────────────────────────────────────
# X3D
# ──────────────────────────────────────────────
class X3DFeatureExtractor(nn.Module):
    """
    X3D feature extractor.
    Supports variants: xs, s, m, l.

    Input:  single tensor [B, 3, T, H, W] or list [slow, fast] (uses fast path).
            If T != clip_length: uniform temporal subsampling.
            If H,W != crop_size: spatial resize via trilinear interpolation.
    Output: feature map [B, feature_dim, T', H', W']
    """

    def __init__(self, variant='m', weight_path=None, device="cuda", freeze=False):
        super().__init__()

        self.device = torch.device(device)
        self.variant = variant.lower()
        assert self.variant in X3D_META, \
            f"Unsupported X3D variant: {variant}, options: {list(X3D_META.keys())}"

        meta = X3D_META[self.variant]
        self.clip_length = meta['clip_length']
        self.crop_size = meta['crop_size']
        self.feature_dim = meta['feature_dim']

        # build backbone (head will be removed later)
        self.model = create_x3d(
            input_channel=3,
            input_clip_length=self.clip_length,
            input_crop_size=self.crop_size,
            model_num_class=400,
            width_factor=meta['width_factor'],
            depth_factor=meta['depth_factor'],
        )

        # load pretrained weights
        if weight_path is not None:
            # local file
            checkpoint = torch.load(weight_path, map_location=self.device)
            if "model_state" in checkpoint:
                checkpoint = checkpoint["model_state"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            self.model.load_state_dict(checkpoint, strict=False)
        else:
            # try torch hub
            try:
                hub_model = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    f'x3d_{self.variant}',
                    pretrained=True,
                    verbose=False,
                )
                self.model.load_state_dict(hub_model.state_dict(), strict=False)
                del hub_model
            except Exception as e:
                print(f"Warning: X3D-{self.variant.upper()} pretrained weights "
                      f"not loaded, using random init: {e}")

        # remove classification head (blocks[-1] is ResNetBasicHead)
        self.model.blocks[-1] = nn.Identity()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model = self.model.to(self.device)

    def forward(self, x):
        # accept list [slow, fast] -> use fast path (higher temporal resolution)
        if isinstance(x, (list, tuple)):
            x = x[1]  # [B, 3, fast_num_frames, H, W]

        x = x.to(self.device)
        B, C, T, H, W = x.shape

        # spatial: resize to crop_size
        if H != self.crop_size or W != self.crop_size:
            x = F.interpolate(
                x,
                size=(T, self.crop_size, self.crop_size),
                mode='trilinear',
                align_corners=False,
            )

        feat = self.model(x)  # [B, feature_dim, T', H', W']
        return feat
