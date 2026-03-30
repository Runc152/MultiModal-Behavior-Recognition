#!/usr/bin/env python
# train.py

import argparse
import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

# 导入项目自定义模块
from datasets.ntu_dataset import NTUDataset
from models.multimodal_slowfast import RGBIRSlowFast
from models.Fusion import Reliability, CrossModalAttention
from models.ClassificationHead import ClassificationHead
from utils.metrics import accuracy



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config):
    """创建训练和验证 DataLoader"""
    dataset = NTUDataset(
        rgb_dir=config['data']['rgb_dir'],
        ir_dir=config['data']['ir_dir'],
        slow_num_frames=config['data']['slow_num_frames'],
        fast_num_frames=config['data']['fast_num_frames'],
        side_size=config['data']['side_size']
    )

    # 按比例划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    return train_loader, val_loader


def create_model(config, device):
    """实例化所有模型模块并移动到设备"""
    # 双模态特征提取器
    rgb_extractor = RGBIRSlowFast(
        rgb_weight=config['model']['rgb_weight'],
        ir_weight=config['model']['ir_weight'],
        device=device
    )

    # 可靠性评估模块
    reliability = Reliability(feature_dim=config['model']['feature_dim']).to(device)

    # 跨模态注意力融合
    cross_attn = CrossModalAttention(feature_dim=config['model']['feature_dim']).to(device)

    # 分类头
    classifier = ClassificationHead(
        input_dim=config['model']['feature_dim'],
        num_classes=config['model']['num_classes'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    return rgb_extractor, reliability, cross_attn, classifier


def train_one_epoch(model, reliability, cross_attn, classifier,
                    dataloader, optimizer, criterion, scaler, device, config):
    """单轮训练"""
    model.train()
    reliability.train()
    cross_attn.train()
    classifier.train()

    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        # 将数据移到设备
        rgb_slow = batch['rgb_slow'].to(device)
        rgb_fast = batch['rgb_fast'].to(device)
        ir_slow = batch['ir_slow'].to(device)
        ir_fast = batch['ir_fast'].to(device)
        labels = batch['label'].to(device)

        # 混合精度上下文
        with autocast(enabled=config['train']['use_amp']):
            # 提取特征
            rgb_feat, ir_feat = model(rgb=[rgb_slow, rgb_fast], ir=[ir_slow, ir_fast])

            # 可靠性评估
            w_rgb, w_ir = reliability(rgb_feat, ir_feat)

            # 跨模态融合
            fused_feat = cross_attn(rgb_feat, ir_feat, w_rgb, w_ir)

            # 分类
            logits = classifier(fused_feat)

            # 损失
            loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        if config['train']['use_amp']:
            scaler.scale(loss).backward()
            if config['train']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(reliability.parameters()) +
                    list(cross_attn.parameters()) + list(classifier.parameters()),
                    config['train']['grad_clip']
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config['train']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(reliability.parameters()) +
                    list(cross_attn.parameters()) + list(classifier.parameters()),
                    config['train']['grad_clip']
                )
            optimizer.step()

        # 记录指标
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        total_loss += loss.item()
        total_acc1 += acc1.item()
        total_acc5 += acc5.item()

        pbar.set_postfix(loss=loss.item(), acc1=acc1.item())

    avg_loss = total_loss / len(dataloader)
    avg_acc1 = total_acc1 / len(dataloader)
    avg_acc5 = total_acc5 / len(dataloader)
    return avg_loss, avg_acc1, avg_acc5


@torch.no_grad()
def validate(model, reliability, cross_attn, classifier,
             dataloader, criterion, device, config):
    """验证"""
    model.eval()
    reliability.eval()
    cross_attn.eval()
    classifier.eval()

    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0

    pbar = tqdm(dataloader, desc='Validation', leave=False)
    for batch in pbar:
        rgb_slow = batch['rgb_slow'].to(device)
        rgb_fast = batch['rgb_fast'].to(device)
        ir_slow = batch['ir_slow'].to(device)
        ir_fast = batch['ir_fast'].to(device)
        labels = batch['label'].to(device)

        with autocast(enabled=config['train']['use_amp']):
            rgb_feat, ir_feat = model(rgb=[rgb_slow, rgb_fast], ir=[ir_slow, ir_fast])
            w_rgb, w_ir = reliability(rgb_feat, ir_feat)
            fused_feat = cross_attn(rgb_feat, ir_feat, w_rgb, w_ir)
            logits = classifier(fused_feat)
            loss = criterion(logits, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        total_loss += loss.item()
        total_acc1 += acc1.item()
        total_acc5 += acc5.item()

    avg_loss = total_loss / len(dataloader)
    avg_acc1 = total_acc1 / len(dataloader)
    avg_acc5 = total_acc5 / len(dataloader)
    return avg_loss, avg_acc1, avg_acc5


def main(config_path):
    # 加载配置
    config = load_config(config_path)

    # 初始化 wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=config['wandb']['name'],
        config=config
    )

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据加载
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 模型
    model, reliability, cross_attn, classifier = create_model(config, device)

    # 优化器（只训练需要梯度的参数）
    params = (list(model.parameters()) +
              list(reliability.parameters()) +
              list(cross_attn.parameters()) +
              list(classifier.parameters()))
    optimizer = AdamW(params, lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])

    # 学习率调度（预热 + 余弦衰减）
    warmup_epochs = config['train']['warmup_epochs']
    total_epochs = config['train']['epochs']
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 混合精度 scaler
    scaler = GradScaler(enabled=config['train']['use_amp'])

    # 保存目录
    os.makedirs(config['train']['save_dir'], exist_ok=True)

    best_acc1 = 0.0
    for epoch in range(1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")

        # 训练
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, reliability, cross_attn, classifier,
            train_loader, optimizer, criterion, scaler, device, config
        )
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc1': train_acc1,
            'train_acc5': train_acc5,
            'lr': optimizer.param_groups[0]['lr']
        })

        # 验证
        if epoch % config['train']['eval_interval'] == 0:
            val_loss, val_acc1, val_acc5 = validate(
                model, reliability, cross_attn, classifier,
                val_loader, criterion, device, config
            )
            wandb.log({
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'val_acc5': val_acc5
            })
            print(f"Val Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")

            # 保存最佳模型
            if val_acc1 > best_acc1:
                best_acc1 = val_acc1
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'reliability_state_dict': reliability.state_dict(),
                    'cross_attn_state_dict': cross_attn.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                    'config': config
                }
                torch.save(checkpoint, os.path.join(config['train']['save_dir'], 'best_model.pth'))
                print(f"New best model saved with Acc@1: {best_acc1:.2f}%")

        # 更新学习率
        scheduler.step()

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MultiModal Behavior Recognition')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)