import torch
import torch.nn as nn
from .slowfast_feature import SlowFastFeatureExtractor, X3DFeatureExtractor, X3D_META


class RGBIRSlowFast(nn.Module):

    def __init__(self,
                 model_type='slowfast',
                 model_variant=None,
                 rgb_weight=None,
                 ir_weight=None,
                 device="cuda"):

        super().__init__()

        self.model_type = model_type.lower()

        if self.model_type == 'slowfast':
            # RGB SlowFast (frozen)
            self.rgb_extractor = SlowFastFeatureExtractor(
                weight_path=rgb_weight,
                device=device,
                freeze=True
            )
            # IR SlowFast (trainable)
            self.ir_extractor = SlowFastFeatureExtractor(
                weight_path=ir_weight,
                device=device,
                freeze=False
            )

        elif self.model_type == 'x3d':
            variant = (model_variant or 'm').lower()

            # RGB X3D — partial fine-tune: freeze stem + early stages,
            # unfreeze later stages for domain adaptation
            self.rgb_extractor = X3DFeatureExtractor(
                variant=variant,
                weight_path=rgb_weight,
                device=device,
                freeze=False,   # will freeze selectively below
            )
            # Freeze blocks[0:3] (stem + stage1 + stage2), keep blocks[3:4] trainable
            for i in range(3):
                for param in self.rgb_extractor.model.blocks[i].parameters():
                    param.requires_grad = False

            # IR X3D — full fine-tune
            self.ir_extractor = X3DFeatureExtractor(
                variant=variant,
                weight_path=ir_weight,
                device=device,
                freeze=False,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}, "
                             f"expected 'slowfast' or 'x3d'")

    def forward(self, rgb, ir):
        # IR single-channel -> repeat to 3 channels
        if self.model_type == 'slowfast':
            ir = [x.repeat(1, 3, 1, 1, 1) if x.shape[1] == 1 else x for x in ir]
        else:
            # x3d forward handles both list and single tensor
            if isinstance(ir, (list, tuple)):
                ir = [x.repeat(1, 3, 1, 1, 1) if x.shape[1] == 1 else x for x in ir]
            else:
                if ir.shape[1] == 1:
                    ir = ir.repeat(1, 3, 1, 1, 1)

        rgb_feat = self.rgb_extractor(rgb)
        ir_feat = self.ir_extractor(ir)

        return rgb_feat, ir_feat
