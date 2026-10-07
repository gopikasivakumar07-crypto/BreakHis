import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from timm.data import Mixup
from timm.utils import ModelEmaV2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class WeightedSoftTargetCrossEntropy(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        per_sample_loss = -(target * logprobs).sum(dim=-1)
        per_sample_weight = (target * self.class_weights.to(target.device)).sum(dim=-1)
        return (per_sample_loss * per_sample_weight).sum() / per_sample_weight.sum()


def set_backbone_trainable(model, trainable: bool, backbone_attr_names):
    for name, param in model.named_parameters():
        if any(name.startswith(attr) for attr in backbone_attr_names):
            param.requires_grad = trainable


def get_amp_settings(device):
    if not torch.cuda.is_available():
        return torch.bfloat16, False, "CPU"
    cap = torch.cuda.get_device_capability(device)
    gpu_name = torch.cuda.get_device_name(device)
    use_bf16 = cap[0] >= 8
    return (torch.bfloat16 if use_bf16 else torch.float16), (not use_bf16), gpu_name


def maybe_compile(model, enable: bool, mode: str = "default"):
    if not enable:
        return model, False
    try:
        compiled = torch.compile(model, mode=mode)
        return compiled, True
    except Exception as e:
        print(f"[warn] torch.compile failed ({e}); continuing uncompiled.")
        return model, False


def train_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    device,
    backbone_param_prefixes,
    num_classes: int,
    epochs: int = 40,
    warmup_epochs: int = 5,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    patience: int = 8,
    use_mixup: bool = True,
    use_ema: bool = True,
    use_compile: bool = True,
    compile_mode: str = "default",
    label_smoothing: float = 0.1,
    batch_size: int = 32,
    reference_batch_size: int = 32,
    log_every: int = 1,
):
    lr_scale = batch_size / reference_batch_size
    lr_head = lr_head * lr_scale
    lr_finetune = lr_finetune * lr_scale
    if lr_scale != 1.0:
        print(f"[info] batch_size={batch_size} vs reference={reference_batch_size} "
              f"-> LR scaled x{lr_scale:.2f} (lr_head={lr_head:.2e}, lr_finetune={lr_finetune:.2e})")

    amp_dtype, use_scaler, gpu_name = get_amp_settings(device)
    print(f"[info] AMP precision: {'fp16 + GradScaler' if use_scaler else 'bf16'} on {gpu_name}")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    model = model.to(device, memory_format=torch.channels_last)
    model, compiled_ok = maybe_compile(model, use_compile, mode=compile_mode)

    ema_model = ModelEmaV2(model, decay=0.9995, device=device) if use_ema else None

    mixup_fn = None
    if use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
            mode="batch", label_smoothing=label_smoothing, num_classes=num_classes,
        )
        train_criterion = WeightedSoftTargetCrossEntropy(class_weights)
    else:
        train_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)

    eval_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_acc = 0.0
    best_state = copy.deepcopy((ema_model.module if ema_model else model).state_dict())
    epochs_no_improve = 0

    set_backbone_trainable(model, False, backbone_param_prefixes)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(warmup_epochs, 1))

    for epoch in range(epochs):
        if epoch == warmup_epochs:
            set_backbone_trainable(model, True, backbone_param_prefixes)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_finetune)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(epochs - warmup_epochs, 1)
            )

        model.train()
        running_loss, running_correct, n = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            hard_labels = labels
            if mixup_fn is not None and imgs.size(0) % 2 == 0:
                imgs, labels = mixup_fn(imgs, labels)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs, aux_outputs = outputs
                    loss = train_criterion(outputs, labels) + 0.4 * train_criterion(aux_outputs, labels)
                else:
                    loss = train_criterion(outputs, labels)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if ema_model is not None:
                ema_model.update(model)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1) if outputs.dim() > 1 else outputs
            running_correct += (preds == hard_labels).sum().item()
            n += imgs.size(0)
        scheduler.step()

        train_loss = running_loss / n
        train_acc = running_correct / n
        eval_model = ema_model.module if ema_model is not None else model
        val_loss, val_acc, _, _ = evaluate(eval_model, val_loader, eval_criterion, device)

        if epoch % log_every == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
                  f"{time.time()-t0:.1f}s/epoch"
                  f"{' [compiled]' if compiled_ok else ''}"
                  f"{' [ema]' if ema_model else ''}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(eval_model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
                break

    final_model = ema_model.module if ema_model is not None else model
    final_model.load_state_dict(best_state)
    return final_model, best_val_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    amp_dtype, _, _ = get_amp_settings(device)
    running_loss, running_correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        running_correct += (preds == labels).sum().item()
        n += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return running_loss / n, running_correct / n, all_preds, all_labels


@torch.no_grad()
def test_report(model, test_loader, class_names, device, tta: bool = True):
    model.eval()
    amp_dtype, _, _ = get_amp_settings(device)
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(device, memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            if tta:
                probs_hflip = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
                probs_vflip = torch.softmax(model(torch.flip(imgs, dims=[2])), dim=1)
                probs = (probs + probs_hflip + probs_vflip) / 3.0
        preds = probs.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    print(f"\n=== TEST SET REPORT (held-out{', with TTA' if tta else ''}) ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    if len(class_names) == 2:
        try:
            auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
            print(f"ROC-AUC: {auc:.4f}")
        except ValueError:
            pass

    return all_preds, all_labels, all_probs
