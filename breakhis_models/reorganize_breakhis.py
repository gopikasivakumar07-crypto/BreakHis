import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _one_pass(patients, targets, trial_seed):
    order = list(patients)
    random.Random(trial_seed).shuffle(order)
    order.sort(key=lambda kv: -len(kv[1]))

    running = {"train": 0, "val": 0, "test": 0}
    assigned = {"train": set(), "val": set(), "test": set()}
    for pid, imgs in order:
        deficits = {s: (targets[s] - running[s]) / targets[s] if targets[s] > 0 else -1.0
                    for s in targets}
        split = max(deficits, key=deficits.get)
        assigned[split].add(pid)
        running[split] += len(imgs)
    return assigned, running


def stratified_patient_split(patient_images: dict, val_frac: float, test_frac: float,
                              seed: int, n_trials: int = 3000):
    total_images = sum(len(imgs) for imgs in patient_images.values())
    targets = {
        "test": total_images * test_frac,
        "val": total_images * val_frac,
        "train": total_images * (1 - val_frac - test_frac),
    }
    patients = list(patient_images.items())

    best_assigned, best_score = None, float("inf")
    rng = random.Random(seed)
    for _ in range(n_trials):
        trial_seed = rng.randint(0, 2**31 - 1)
        assigned, running = _one_pass(patients, targets, trial_seed)
        score = sum((running[s] - targets[s]) ** 2 for s in targets)
        if score < best_score:
            best_score, best_assigned = score, assigned
            if best_score == 0:
                break

    return best_assigned["train"], best_assigned["val"], best_assigned["test"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to BreaKHis_v1/histology_slides/breast")
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--magnification", type=str, default="40X",
                         choices=["40X", "100X", "200X", "400X", "all"])
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    split_counts = {"train": {}, "val": {}, "test": {}}

    for cls in ["benign", "malignant"]:
        patient_images = defaultdict(list)
        cls_root = src / cls / "SOB"
        for img_path in cls_root.rglob("*.png"):
            if args.magnification != "all" and f"{args.magnification}" not in img_path.parts:
                continue
            patient_id = img_path.parts[-2] if img_path.parts[-2] not in ("40X", "100X", "200X", "400X") \
                else img_path.parts[-3]
            patient_images[patient_id].append(img_path)

        train_patients, val_patients, test_patients = stratified_patient_split(
            patient_images, args.val_frac, args.test_frac, args.seed
        )

        for split_name, split_patients in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
            out_dir = dst / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            n_images = 0
            for pid in split_patients:
                for img_path in patient_images[pid]:
                    shutil.copy2(img_path, out_dir / f"{pid}_{img_path.name}")
                    n_images += 1
            split_counts[split_name][cls] = n_images

        print(f"{cls}: {len(train_patients)} train / {len(val_patients)} val / {len(test_patients)} test patients")

    print(f"\nDone. Data written to {dst}/{{train,val,test}}/{{benign,malignant}}/\n")

    print("=== Split balance check ===")
    for split_name in ["train", "val", "test"]:
        b = split_counts[split_name].get("benign", 0)
        m = split_counts[split_name].get("malignant", 0)
        total = b + m
        if total == 0:
            continue
        print(f"{split_name.upper()}")
        print(f"  Benign: {b}")
        print(f"  Malignant: {m}")
        print(f"  Total: {total}")
        print(f"  Benign ratio: {b/total*100:.2f}%")
        print(f"  Malignant ratio: {m/total*100:.2f}%\n")


if __name__ == "__main__":
    main()
