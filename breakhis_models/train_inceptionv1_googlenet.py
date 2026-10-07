import argparse
import os

import torch
import torch.nn as nn
from torchvision import models

from common.dataset import build_dataloaders
from common.engine import train_model, test_report


def build_model(num_classes: int, pretrained: bool = True):
    model = models.googlenet(
        weights=models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None,
        aux_logits=False, init_weights=not pretrained,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./runs/inceptionv1_googlenet")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile_mode", type=str, default="default",
                         choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_mixup", action="store_true", help="disable MixUp/CutMix")
    parser.add_argument("--no_ema", action="store_true", help="disable EMA weight averaging")
    parser.add_argument("--no_compile", action="store_true", help="disable torch.compile")
    parser.add_argument("--no_tta", action="store_true", help="disable test-time augmentation at eval")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not torch.cuda.is_available():
        print("[warn] No CUDA GPU detected — this script is tuned for A100/H100 "
              "(bf16/TF32/channels_last/compile). It will still run on CPU but slowly "
              "and some flags (bf16 autocast) are silently no-ops without CUDA.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    train_loader, val_loader, test_loader, class_names, class_weights = build_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Classes: {class_names} | train batches/epoch: {len(train_loader)}")

    model = build_model(num_classes=len(class_names), pretrained=not args.no_pretrained)

    model, best_val_acc = train_model(
        model, train_loader, val_loader, class_weights, device,
        backbone_param_prefixes=['conv1', 'conv2', 'conv3', 'inception3', 'inception4', 'inception5'],
        num_classes=len(class_names),
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        lr_head=args.lr_head, lr_finetune=args.lr_finetune, patience=args.patience,
        use_mixup=not args.no_mixup, use_ema=not args.no_ema,
        use_compile=not args.no_compile, compile_mode=args.compile_mode,
        batch_size=args.batch_size, reference_batch_size=128,
    )
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "inceptionv1_googlenet_best.pt"))
    test_report(model, test_loader, class_names, device, tta=not args.no_tta)


if __name__ == "__main__":
    main()
