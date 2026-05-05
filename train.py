import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

# Import project-specific modules
from datasets.ntu_dataset import NTUDataset
from models.multimodal_model import MultiModalModel
from utils.metrics import accuracy


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config):
    """Create train and validation DataLoaders."""
    dataset = NTUDataset(
        rgb_dir=config['data']['rgb_dir'],
        ir_dir=config['data']['ir_dir'],
        slow_num_frames=config['data']['slow_num_frames'],
        fast_num_frames=config['data']['fast_num_frames'],
        side_size=config['data']['side_size']
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=filter_invalid_collate 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=filter_invalid_collate  
    )
    return train_loader, val_loader


def create_model(config, device):
    """Create the model and move it to the target device."""
    model = MultiModalModel(
        use_reliability=config['model'].get('use_reliability', True),
        rgb_weight=config['model']['rgb_weight'],
        ir_weight=config['model']['ir_weight'],
        feature_dim=config['model']['feature_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes']
    ).to(device)
    return model


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_acc1, config, tb_log_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_acc1': best_acc1,
        'config': config,
        'tb_log_dir': tb_log_dir
    }
    torch.save(checkpoint, path)


def load_checkpoint(resume_path, model, optimizer=None, scheduler=None, scaler=None,
                    device='cpu', load_optimizer=True):
    """Load a checkpoint, optionally restoring optimizer state."""
    checkpoint = torch.load(resume_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    start_epoch = 1
    best_acc1 = 0.0
    tb_log_dir = None

    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
    if 'best_acc1' in checkpoint:
        best_acc1 = checkpoint['best_acc1']

    # Optionally restore optimizer, scheduler, and AMP scaler
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if load_optimizer and scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if load_optimizer and scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    
    if 'tb_log_dir' in checkpoint:
        tb_log_dir = checkpoint['tb_log_dir']

    print(f"Successfully loaded checkpoint: {resume_path}")
    print(f"Resuming from epoch {start_epoch}, current best_acc1 = {best_acc1:.2f}")

    return start_epoch, best_acc1, tb_log_dir


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, config):
    """Train the model for one epoch."""
    model.train()

    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        # Move batch data to the device
        rgb_slow = batch['rgb_slow'].to(device)
        rgb_fast = batch['rgb_fast'].to(device)
        ir_slow = batch['ir_slow'].to(device)
        ir_fast = batch['ir_fast'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        with autocast(enabled=config['train']['use_amp']):
            # Get multi-modal feature predictions
            logits = model(rgb=[rgb_slow, rgb_fast], ir=[ir_slow, ir_fast])

            # Compute loss
            loss = criterion(logits, labels)

        # Backpropagate gradients
        if config['train']['use_amp']:
            scaler.scale(loss).backward()
            if config['train']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['train']['grad_clip']
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config['train']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['train']['grad_clip']
                )
            optimizer.step()

        # Record metrics
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        total_loss += loss.item()
        total_acc1 += acc1.item()
        total_acc5 += acc5.item()

        pbar.set_postfix(loss=loss.item(), acc1=acc1.item(), acc5=acc5.item())

    avg_loss = total_loss / len(dataloader)
    avg_acc1 = total_acc1 / len(dataloader)
    avg_acc5 = total_acc5 / len(dataloader)
    return avg_loss, avg_acc1, avg_acc5


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    """Validate the model."""
    model.eval()

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
            logits = model(rgb=[rgb_slow, rgb_fast], ir=[ir_slow, ir_fast])
            loss = criterion(logits, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        total_loss += loss.item()
        total_acc1 += acc1.item()
        total_acc5 += acc5.item()

    avg_loss = total_loss / len(dataloader)
    avg_acc1 = total_acc1 / len(dataloader)
    avg_acc5 = total_acc5 / len(dataloader)
    return avg_loss, avg_acc1, avg_acc5


def filter_invalid_collate(batch):
    """自定义 collate 函数：过滤掉 label==-1 的无效样本"""
    if not batch:
        return {}
    
    # 过滤出有效样本
    valid_samples = [sample for sample in batch if sample['label'] != -1]
    
    if not valid_samples:
        # 如果 batch 中没有有效样本，返回空字典
        return {}
    
    # 将有效样本堆叠成张量
    keys = valid_samples[0].keys()
    collated_batch = {}
    for key in keys:
        collated_batch[key] = torch.stack([sample[key] for sample in valid_samples])
    
    return collated_batch


def main(config_path, resume_path=None, load_optimizer=True):
    # Load configuration
    config = load_config(config_path)

    seed = config['train'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tb_log_dir = config['tensorboard']['log_dir']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Model
    model = create_model(config, device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # Learning rate schedule (warmup + cosine annealing)
    warmup_epochs = config['train']['warmup_epochs']
    total_epochs = config['train']['epochs']
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Mixed precision scaler
    scaler = GradScaler(enabled=config['train']['use_amp'])

    # Create save directory
    os.makedirs(config['train']['save_dir'], exist_ok=True)

    # Default training state
    start_epoch = 1
    best_acc1 = 0.0

    # If no resume path is provided, try the one in config
    if resume_path is None:
        resume_path = config['train'].get('resume_path', None)

    # Resume training / load checkpoint
    if resume_path:
        start_epoch, best_acc1, tb_log_dir = load_checkpoint(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            load_optimizer=load_optimizer
        )
        # If checkpoint doesn't have tb_log_dir, use the one from config
        if tb_log_dir is None:
            tb_log_dir = config['tensorboard']['log_dir']

    tb_writer = SummaryWriter(log_dir=tb_log_dir)

    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")

        # Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, config
        )
        
        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train_top1', train_acc1, epoch)
        tb_writer.add_scalar('Accuracy/Train_top5', train_acc5, epoch)

        # Validate
        if epoch % config['train']['eval_interval'] == 0:
            val_loss, val_acc1, val_acc5 = validate(
                model, val_loader, criterion, device, config
            )

            tb_writer.add_scalar('Loss/Val', val_loss, epoch)
            tb_writer.add_scalar('Accuracy/Val_top1', val_acc1, epoch)
            tb_writer.add_scalar('Accuracy/Val_top5', val_acc5, epoch)
            print(f"Val Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")

            # Always save the latest checkpoint during validation
            latest_path = os.path.join(config['train']['save_dir'], 'latest_model.pth')
            save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, best_acc1, config, tb_log_dir)

            # Save the best checkpoint
            if val_acc1 > best_acc1:
                best_acc1 = val_acc1
                best_path = os.path.join(config['train']['save_dir'], 'best_model.pth')
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_acc1, config, tb_log_dir)
                print(f"New best model saved with Acc@1: {best_acc1:.2f}%")

        # Update scheduler
        scheduler.step()

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MultiModal Behavior Recognition')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--only_load_weights', action='store_true', help='Only load model weights')
    args = parser.parse_args()

    main(
        config_path=args.config,
        resume_path=args.resume,
        load_optimizer=not args.only_load_weights
    )