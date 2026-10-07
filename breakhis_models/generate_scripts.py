import os

HEADER = '''"""
train_{fname}.py

Standalone training script for {display_name} on BreakHis (or any
ImageFolder-structured binary/multi-class histopathology dataset).

Tuned for A100/H100: bf16 autocast, TF32, channels_last, torch.compile,
MixUp/CutMix, label smoothing, EMA weights, TTA at test time.
See common/engine.py for what each of these does and why.

Usage:
    python train_{fname}.py --data_dir /path/to/data --epochs 40 --out_dir ./runs/{fname}

    # revert to the original conservative batch size if you hit OOM:
    python train_{fname}.py --data_dir /path/to/data --batch_size {batch_size_default}

    # disable any single lever if you want to ablate it:
    python train_{fname}.py --data_dir /path/to/data --no_mixup --no_ema --no_compile

Expected data_dir layout: see common/dataset.py docstring.

Speed notes (see common/engine.py for details):
  - Default batch size is now {batch_size_large} (up from the original
    {batch_size_default}) to use A100/H100 VRAM more fully and cut wall-clock
    time via fewer optimizer steps/epoch. LR is scaled proportionally
    (linear scaling rule) so this shouldn't change final accuracy — it's
    not "bigger batch, same LR, hope for the best."
  - torch.compile mode defaults to "default" instead of "max-autotune":
    on a dataset this small, max-autotune's 60-90s kernel search per model
    is overhead that a few minutes of training can't amortize. Use
    --compile_mode max-autotune if you're doing a long run (100+ epochs).
"""

import argparse
import os

import torch
import torch.nn as nn
{extra_imports}

from common.dataset import build_dataloaders
from common.engine import train_model, test_report


def build_model(num_classes: int, pretrained: bool = True):
{build_model_body}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./runs/{fname}")
    parser.add_argument("--img_size", type=int, default={img_size})
    parser.add_argument("--batch_size", type=int, default={batch_size_large})
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
        print(f"Using device: {{device}} ({{torch.cuda.get_device_name(0)}})")

    train_loader, val_loader, test_loader, class_names, class_weights = build_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Classes: {{class_names}} | train batches/epoch: {{len(train_loader)}}")

    model = build_model(num_classes=len(class_names), pretrained=not args.no_pretrained)

    model, best_val_acc = train_model(
        model, train_loader, val_loader, class_weights, device,
        backbone_param_prefixes={backbone_prefixes},
        num_classes=len(class_names),
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        lr_head=args.lr_head, lr_finetune=args.lr_finetune, patience=args.patience,
        use_mixup=not args.no_mixup, use_ema=not args.no_ema,
        use_compile=not args.no_compile, compile_mode=args.compile_mode,
        batch_size=args.batch_size, reference_batch_size={batch_size_default},
    )
    print(f"Best validation accuracy: {{best_val_acc:.4f}}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "{fname}_best.pt"))
    test_report(model, test_loader, class_names, device, tta=not args.no_tta)


if __name__ == "__main__":
    main()
'''

MODELS = [
    dict(
        fname="resnet18", display_name="ResNet-18", img_size=224,
        batch_size_default=128, batch_size_large=256,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model''',
        backbone_prefixes=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    ),
    dict(
        fname="resnet50", display_name="ResNet-50", img_size=224,
        batch_size_default=96, batch_size_large=192,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model''',
        backbone_prefixes=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    ),
    dict(
        fname="resnet101", display_name="ResNet-101", img_size=224,
        batch_size_default=64, batch_size_large=128,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model''',
        backbone_prefixes=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    ),
    dict(
        fname="densenet121", display_name="DenseNet-121", img_size=224,
        batch_size_default=64, batch_size_large=128,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model''',
        backbone_prefixes=["features"],
    ),
    dict(
        fname="densenet161", display_name="DenseNet-161", img_size=224,
        batch_size_default=48, batch_size_large=96,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model''',
        backbone_prefixes=["features"],
    ),
    dict(
        fname="densenet169", display_name="DenseNet-169", img_size=224,
        batch_size_default=64, batch_size_large=128,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model''',
        backbone_prefixes=["features"],
    ),
    dict(
        fname="densenet201", display_name="DenseNet-201", img_size=224,
        batch_size_default=48, batch_size_large=96,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model''',
        backbone_prefixes=["features"],
    ),
    dict(
        fname="inceptionv1_googlenet", display_name="InceptionNet-V1 (GoogLeNet)", img_size=224,
        batch_size_default=128, batch_size_large=256,
        extra_imports="from torchvision import models",
        build_model_body='''    # torchvision's GoogLeNet IS Inception-v1. aux_logits are disabled for
    # fine-tuning simplicity (they only help when training from scratch).
    model = models.googlenet(
        weights=models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None,
        aux_logits=False, init_weights=not pretrained,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model''',
        backbone_prefixes=["conv1", "conv2", "conv3", "inception3", "inception4", "inception5"],
    ),
    dict(
        fname="inceptionv3", display_name="InceptionNet-V3", img_size=299,
        batch_size_default=64, batch_size_large=128,
        extra_imports="from torchvision import models",
        build_model_body='''    # Inception-v3 requires 299x299 input (see --img_size default below).
    model = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
        aux_logits=True, init_weights=not pretrained,
    )
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model''',
        backbone_prefixes=["Conv2d", "Mixed"],
    ),
    dict(
        fname="mobilenetv1", display_name="MobileNet-V1", img_size=224,
        batch_size_default=192, batch_size_large=384,
        extra_imports="import timm",
        build_model_body='''    # torchvision has no MobileNetV1; timm's mobilenetv1_100 has real
    # ImageNet-pretrained weights and is the standard substitute.
    model = timm.create_model("mobilenetv1_100", pretrained=pretrained, num_classes=num_classes)
    return model''',
        backbone_prefixes=["conv_stem", "bn1", "blocks"],
    ),
    dict(
        fname="mobilenetv2", display_name="MobileNet-V2", img_size=224,
        batch_size_default=192, batch_size_large=384,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model''',
        backbone_prefixes=["features"],
    ),
    dict(
        fname="mobilenetv3", display_name="MobileNet-V3 (Large)", img_size=224,
        batch_size_default=192, batch_size_large=384,
        extra_imports="from torchvision import models",
        build_model_body='''    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model''',
        backbone_prefixes=["features"],
    ),
    dict(
        fname="xception", display_name="XceptionNet", img_size=299,
        batch_size_default=64, batch_size_large=128,
        extra_imports="import timm",
        build_model_body='''    # torchvision has no Xception; timm's legacy_xception carries the
    # standard ImageNet-pretrained weights used in the original paper's spirit.
    model = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=num_classes)
    return model''',
        backbone_prefixes=["conv1", "bn1", "conv2", "bn2", "block1", "block2", "block3", "block4",
                            "block5", "block6", "block7", "block8", "block9", "block10", "block11",
                            "block12", "conv3", "bn3", "conv4", "bn4"],
    ),
]

os.makedirs("/home/claude/breakhis_models", exist_ok=True)

for m in MODELS:
    content = HEADER.format(
        fname=m["fname"],
        display_name=m["display_name"],
        extra_imports=m["extra_imports"],
        build_model_body=m["build_model_body"],
        img_size=m["img_size"],
        batch_size_default=m["batch_size_default"],
        batch_size_large=m["batch_size_large"],
        backbone_prefixes=repr(m["backbone_prefixes"]),
    )
    path = f"/home/claude/breakhis_models/train_{m['fname']}.py"
    with open(path, "w") as f:
        f.write(content)
    print("wrote", path)
