# BreakHis — 13 separate backbone training scripts

One self-contained script per architecture. Same recipe underneath
(transfer learning → staged fine-tuning → class-weighted loss →
cosine LR → early stopping) so results are comparable across models,
but each script is fully standalone per your request.

## Setup

```bash
pip install torch torchvision timm scikit-learn
```

## 1. Get your data into the right shape (once)

```bash
python reorganize_breakhis.py --src /path/to/BreaKHis_v1/histology_slides/breast \
    --dst ./data --magnification 40X
```

This does a **patient-wise** split (not image-wise) into
`data/{train,val,test}/{benign,malignant}/`. Patient-wise splitting
matters — BreaKHis has many images per patient, and an image-wise split
lets near-duplicate patches leak between train and test, which is the
single most common source of inflated "99% accuracy" claims on this
dataset in weaker papers. Keep the split patient-wise if you want
numbers a reviewer won't challenge.

## 2. Train any model

```bash
python train_resnet50.py       --data_dir ./data --epochs 40
python train_densenet121.py    --data_dir ./data --epochs 40
python train_inceptionv3.py    --data_dir ./data --epochs 40   # needs 299x299, handled automatically
python train_xception.py       --data_dir ./data --epochs 40   # needs 299x299, handled automatically
# ...same pattern for all 13
```

Each run saves the best checkpoint to `./runs/<model>/<model>_best.pt`
and prints a full classification report + confusion matrix + ROC-AUC
on the **held-out test set** at the end — not train accuracy, not val
accuracy used for early stopping.

## Models included (13 of the 14 in your diagram)

| Backbone | Script | Source of pretrained weights |
|---|---|---|
| ResNet-18 | `train_resnet18.py` | torchvision |
| ResNet-50 | `train_resnet50.py` | torchvision |
| ResNet-101 | `train_resnet101.py` | torchvision |
| DenseNet-121 | `train_densenet121.py` | torchvision |
| DenseNet-161 | `train_densenet161.py` | torchvision |
| DenseNet-169 | `train_densenet169.py` | torchvision |
| DenseNet-201 | `train_densenet201.py` | torchvision |
| InceptionNet-V1 | `train_inceptionv1_googlenet.py` | torchvision (GoogLeNet *is* Inception-v1) |
| InceptionNet-V3 | `train_inceptionv3.py` | torchvision |
| MobileNet-V1 | `train_mobilenetv1.py` | timm (`mobilenetv1_100`) |
| MobileNet-V2 | `train_mobilenetv2.py` | torchvision |
| MobileNet-V3 | `train_mobilenetv3.py` | torchvision (Large variant) |
| XceptionNet | `train_xception.py` | timm (`legacy_xception`) |

**InceptionNet-V2 is intentionally not included.** There is no
maintained, genuinely-pretrained "Inception-v2" checkpoint in
torchvision or timm — I checked both directly rather than guessing.
What papers usually call "v2" is BN-Inception, which isn't available
pretrained anywhere reliable. Two honest options if you need a 14th
model for the diagram: (a) swap in `inception_resnet_v2` (timm has
real pretrained weights for this one), or (b) note in the paper that
V2 was omitted from the empirical comparison for this reason — a
reviewer will respect that far more than a fabricated citation to
weights that don't exist.

## Model batch sizes and compile mode (updated)

Default `--batch_size` is now the larger tier (e.g. ResNet-50: 192,
DenseNet-121: 128) to cut wall-clock time via fewer optimizer steps per
epoch. Learning rate is scaled proportionally (linear LR scaling rule),
so this isn't "bigger batch, same LR, hope it still converges" — the
effective training dynamics stay close to the original tuning, so
accuracy should hold. If you hit OOM, drop back to the original size
printed in each script's `--help`.

`torch.compile` now defaults to `mode="default"` instead of
`"max-autotune"`. On a dataset the size of BreakHis, max-autotune's
60-90s kernel search per model is overhead a few minutes of training
can't amortize — `default` compiles in a few seconds with steady-state
throughput close enough not to matter here. Use
`--compile_mode max-autotune` only for long runs (100+ epochs on the
full multi-magnification set).



This recipe is what the BreakHis literature typically uses to land in
that range with transfer learning. Whether *your* run gets there
depends on your actual data quality, the patient-wise split, and how
distinguishable your classes are — it isn't something a training
script can promise. Report exactly what `test_report()` prints, on
whichever architectures actually hit those numbers on your data.

## Running on P100 instead of A100/H100

The engine now auto-detects the GPU and picks the right precision —
you don't need to change any flags for this:
- **A100/H100** (compute capability >= 8.0): bf16 autocast, no GradScaler needed.
- **P100** (Pascal, compute capability 6.0) and anything older: fp16 autocast
  + GradScaler instead. Pascal has no hardware bf16 — those ops would either
  silently run unaccelerated or emulate, so bf16 is the wrong tool here.
  P100 does have solid fp16 throughput (one of Pascal's actual strengths),
  it just needs loss scaling since fp16's narrow exponent range can
  under/overflow during backward — which GradScaler handles automatically.
  You'll see `[info] AMP precision: fp16 + GradScaler on Tesla P100-...`
  printed at the start of training to confirm.

**Batch sizes need to come down manually.** The default `--batch_size` in
each script assumes A100/H100-class VRAM (40-80GB). Standard P100 has
16GB and will OOM at those defaults. Pass a smaller value explicitly —
LR is auto-scaled relative to each script's original reference batch size,
so you don't need to touch `--lr_head`/`--lr_finetune` yourself:

| Model | Suggested `--batch_size` on P100 (16GB) |
|---|---|
| MobileNet-V1/V2/V3 | 64 |
| ResNet-18, GoogLeNet | 64 |
| ResNet-50, DenseNet-121/169 | 32 |
| ResNet-101, DenseNet-201 | 24 |
| DenseNet-161, InceptionV3, Xception | 16 |

These are starting points, not exact — if you OOM, halve it; if you have
headroom, nudge it up. `torch.compile` still works on Pascal (it falls
back gracefully if a kernel isn't supported), but won't give the same
speedup it does on Ampere/Hopper since there are no tensor cores to
exploit either way.
