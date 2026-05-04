import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda
import math

class NTUDataset(Dataset):
    """
    多模态视频数据集：RGB + IR
    整个视频加载 → 输出 slow(32fps) + fast(8fps) 两路
    """
    def __init__(
        self,
        rgb_dir,
        ir_dir,
        slow_num_frames=8,   # slow 路径 8 帧
        fast_num_frames=32,    # fast 路径 32 帧
        side_size=256,
    ):
        self.rgb_dir = Path(rgb_dir)
        self.ir_dir = Path(ir_dir)
        self.slow_num_frames = slow_num_frames
        self.fast_num_frames = fast_num_frames
        self.side_size = side_size

        # 匹配 RGB-IR 对
        self.rgb_files = sorted(self.rgb_dir.glob("*_rgb.avi"))
        self.ir_files = sorted(self.ir_dir.glob("*_ir.avi"))
        self.pairs = self._match_pairs()

        # 标签
        self.label_map = self._build_label_map()

    def _match_pairs(self):
        ir_dict = {f.stem.replace("_ir", ""): f for f in self.ir_files}
        pairs = []
        for rgb in self.rgb_files:
            pre = rgb.stem.replace("_rgb", "")
            if pre in ir_dict:
                pairs.append((rgb, ir_dict[pre]))
        return pairs

    def _build_label_map(self):
        label_map = {}
        for i, (rgb_path, _) in enumerate(self.pairs):
            name = rgb_path.stem
            a = int(name.split("A")[-1].split("_")[0])
            label_map[i] = a - 1  # 标签从 0 开始
        return label_map

    def _get_transform(self, num_frames, is_rgb=True, is_slow=True):
        if is_rgb:
            mean = [0.45, 0.45, 0.45]
            std  = [0.225, 0.225, 0.225]
        else:
            mean = [0.5]
            std  = [0.5]

        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                ShortSideScale(size=self.side_size)
            ])
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, ir_path = self.pairs[idx]
        label = self.label_map[idx]

        try:
            # 1. 加载整个视频
            rgb_video = EncodedVideo.from_path(str(rgb_path))
            ir_video  = EncodedVideo.from_path(str(ir_path))

            # 整个视频时长
            rgb_dur = int(rgb_video.duration)
            ir_dur = int(ir_video.duration)
            duration = min(rgb_dur, ir_dur)

            # 2. 整个视频取 clip
            rgb_clip = rgb_video.get_clip(0, duration)
            ir_clip  = ir_video.get_clip(0, duration)

            # 变换
            transform_slow = self._get_transform(self.slow_num_frames, is_slow=True)
            transform_fast = self._get_transform(self.fast_num_frames, is_slow=False)

            # 3. slow / fast 两路采样
            rgb_slow = transform_slow(rgb_clip)["video"]
            rgb_fast = transform_fast(rgb_clip)["video"]
            ir_slow  = transform_slow(ir_clip)["video"]
            ir_fast  = transform_fast(ir_clip)["video"]

            return {
                "rgb_slow": rgb_slow,
                "rgb_fast": rgb_fast,
                "ir_slow":  ir_slow,
                "ir_fast":  ir_fast,
                "label":    torch.tensor(label, dtype=torch.long)
            }

        except Exception as e:
            print(f"Error idx {idx}: {e}")
            return {
                "rgb_slow": torch.zeros(3, self.slow_num_frames, self.side_size, self.side_size),
                "rgb_fast": torch.zeros(3, self.fast_num_frames, self.side_size, self.side_size),
                "ir_slow":  torch.zeros(1, self.slow_num_frames, self.side_size, self.side_size),
                "ir_fast":  torch.zeros(1, self.fast_num_frames, self.side_size, self.side_size),
                "label":    torch.tensor(-1)
            }