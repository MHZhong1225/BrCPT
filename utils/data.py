# utils/data.py

import os
import re
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import random

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


def get_dataloaders_imagefolder(
    dataset_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    mean: Tuple[float, float, float] = (0.78, 0.62, 0.76),
    std: Tuple[float, float, float] = (0.15, 0.19, 0.14),
    num_workers: int = 6,
) -> Dict[str, Any]:
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=80),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }

    return {
        'dataloaders': dataloaders,
        'sizes': {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)},
        'class_names': train_dataset.classes
    }

# -------------------------------
# BreakHis 
# -------------------------------

# 亚型映射（8类）
SUBTYPE_CODES = {
    # 良性
    "A": "Adenosis",
    "F": "Fibroadenoma",
    "PT": "Phyllodes Tumor",
    "TA": "Tubular Adenoma",
    # 恶性
    "DC": "Ductal Carcinoma",
    "LC": "Lobular Carcinoma",
    "MC": "Mucinous Carcinoma",
    "PC": "Papillary Carcinoma",
}
SUBTYPE_NAME_TO_CODE = {v.lower(): k for k, v in SUBTYPE_CODES.items()}
MULTI_LABELS = {code: i for i, code in enumerate(SUBTYPE_CODES.keys())}
BINARY_LABELS = {"benign": 0, "malignant": 1}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def _find_images(root: str) -> List[str]:
    paths: List[str] = []
    for ext in IMG_EXTS:
        paths.extend([str(p) for p in Path(root).rglob(f"*{ext}")])
    return paths

def _parse_magnification(path: str) -> Optional[str]:
    m = re.search(r"(40X|100X|200X|400X)", path, re.IGNORECASE)
    return m.group(1).upper() if m else None

def _parse_benign_malignant(path: str) -> Optional[int]:
    lower = path.lower()
    if "benign" in lower or "_b_" in lower:
        return BINARY_LABELS["benign"]
    if "malignant" in lower or "_m_" in lower:
        return BINARY_LABELS["malignant"]
    return None

def _parse_subtype_code(path: str) -> Optional[str]:
    fname = os.path.basename(path)
    m = re.search(r"_(?:B|M)_(A|F|PT|TA|DC|LC|MC|PC)\b", fname, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    parts = [p.lower() for p in Path(path).parts]
    for p in parts[::-1]:
        for full_name, code in SUBTYPE_NAME_TO_CODE.items():
            if full_name in p:
                return code.upper()
    return None

def _get_patient_id(path: str) -> Optional[str]:
    # 例:SOB_B_A-14-22549-40X-001.png -> patient id ~ "14-22549"
    fname = os.path.basename(path)
    m = re.search(r"_(?:B|M)_(?:A|F|PT|TA|DC|LC|MC|PC)-(.*?)-\d+", fname, re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: 上一级目录
    return os.path.basename(os.path.dirname(path)) or None

class BreakHisDataset(Dataset):
    def __init__(
        self,
        root: str,
        task: str = "multiclass",           # "binary" or "multiclass"
        magnifications: Optional[List[str]] = None,  # e.g., ["40X"], None for all
        transform: Optional[transforms.Compose] = None,
    ):
        files = _find_images(root)
        allowed = None if not magnifications else {m.upper() for m in magnifications}

        self.samples = []
        for fp in files:
            mag = _parse_magnification(fp)
            if allowed and (mag not in allowed):
                continue
            binlab = _parse_benign_malignant(fp)
            subcode = _parse_subtype_code(fp)
            pid = _get_patient_id(fp)
            self.samples.append({
                "path": fp,
                "magnification": mag,
                "binary": binlab,
                "subcode": subcode,
                "patient_id": pid,
            })

        if task == "binary":
            self.samples = [s for s in self.samples if s["binary"] is not None]
            self.num_classes = 2
            self.class_names = ["benign", "malignant"]
        elif task == "multiclass":
            self.samples = [s for s in self.samples if s["subcode"] in MULTI_LABELS]
            self.num_classes = len(MULTI_LABELS)
            self.class_names = [SUBTYPE_CODES[k] for k in MULTI_LABELS.keys()]
        else:
            raise ValueError("task must be 'binary' or 'multiclass'")

        if not self.samples:
            raise RuntimeError("未找到可用样本，请检查路径/倍数/命名。")

        self.task = task
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.task == "binary":
            label = s["binary"]
        else:
            label = MULTI_LABELS[s["subcode"]]
        return self.transform(img), label

def _split_by_patient(samples: List[dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    pid_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        pid = s.get("patient_id") or f"noid_{i}"
        pid_to_indices.setdefault(pid, []).append(i)

    pids = list(pid_to_indices.keys())
    random.Random(seed).shuffle(pids)

    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_pids = set(pids[:n_train])
    val_pids = set(pids[n_train:n_train + n_val])
    test_pids = set(pids[n_train + n_val:])

    def collect(target_pids):
        out = []
        for pid in target_pids:
            out.extend(pid_to_indices[pid])
        return out

    return collect(train_pids), collect(val_pids), collect(test_pids)

def get_dataloaders_breakhis(
    dataset_path: str,
    task: str = "multiclass",
    magnifications: Optional[List[str]] = None,     # 例如 ["40X"] 或 None=全部
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    num_workers: int = 6,
    seed: int = 2025,
    train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
) -> Dict[str, Any]:
    train_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_test_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_ds = BreakHisDataset(root=dataset_path, task=task, magnifications=magnifications, transform=train_tf)

    # 3) 按病人划分索引
    tr_idx, va_idx, te_idx = _split_by_patient(full_ds.samples, train_ratio, val_ratio, test_ratio, seed)

    val_ds = BreakHisDataset(dataset_path, task=task, magnifications=magnifications, transform=val_test_tf)
    test_ds = BreakHisDataset(dataset_path, task=task, magnifications=magnifications, transform=val_test_tf)

    train_set = Subset(full_ds, tr_idx)
    val_set   = Subset(val_ds,  va_idx)
    test_set  = Subset(test_ds, te_idx)

    # 5) DataLoaders
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        'val':   DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test':  DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }

    return {
        'dataloaders': dataloaders,
        'sizes': {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)},
        'class_names': full_ds.class_names,   # ["Adenosis", ...] 或 ["benign","malignant"]
        'num_classes': full_ds.num_classes
    }


def get_dataloaders(
    dataset_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    mean: Tuple[float, float, float] = (0.78, 0.62, 0.76),
    std: Tuple[float, float, float] = (0.15, 0.19, 0.14),
    # 下面是 BreakHis 专用可选项（有 train/val/test 时会被忽略）
    task: str = "multiclass",
    magnifications: Optional[List[str]] = None,
    num_workers: int = 6,
    seed: int = 2025,
    train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
) -> Dict[str, Any]:
    has_split = all(os.path.isdir(os.path.join(dataset_path, d)) for d in ["train", "val", "test"])
    if has_split:
        return get_dataloaders_imagefolder(
            dataset_path, image_size, batch_size, mean, std, num_workers
        )
    else:
        return get_dataloaders_breakhis(
            dataset_path, task, magnifications, image_size, batch_size,
            mean, std, num_workers, seed, train_ratio, val_ratio, test_ratio
        )


if __name__ == "__main__":
    import argparse

    def parse_triplet(s: str | None):
        if not s:
            return None
        parts = [p.strip() for p in s.split(",")]
        assert len(parts) == 3, "mean/std 需要形如 '0.5,0.5,0.5'"
        return tuple(float(x) for x in parts)

    def parse_pair(s: str | None):
        if not s:
            return (224, 224)
        parts = [p.strip() for p in s.split(",")]
        assert len(parts) == 2, "image_size 需要形如 '224,224'"
        return (int(parts[0]), int(parts[1]))

    parser = argparse.ArgumentParser(
        description="Check dataset splits and basic info for ImageFolder or BreakHis root."
    )
    parser.add_argument("--dataset-path", required=True, help="根目录:包含 train/val/test 或 BreakHis 的 breast 根目录")
    parser.add_argument("--task", default="multiclass", choices=["binary", "multiclass"],
                        help="BreakHis 专用:binary=良恶性;multiclass=8亚型")
    parser.add_argument("--magnifications", default="all",
                        help="BreakHis 专用:'all' 或 '40X,100X,200X,400X' 逗号分隔")
    parser.add_argument("--image-size", default="224,224", help="如 '224,224'")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=6)

    args = parser.parse_args()

    mags = None if args.magnifications.lower() == "all" else [m.strip().upper()
            for m in args.magnifications.split(",") if m.strip()]

    img_size = parse_pair(args.image_size)
    mean = parse_triplet(args.mean) if args.mean else (0.78, 0.62, 0.76)
    std  = parse_triplet(args.std)  if args.std  else (0.15, 0.19, 0.14)

    has_split = all(os.path.isdir(os.path.join(args.dataset_path, d)) for d in ["train", "val", "test"])
    print("\n--- Data Config ---")
    print(f"dataset_path : {args.dataset_path}")
    print(f"mode         : {'ImageFolder(train/val/test)' if has_split else 'BreakHis(auto split by patient)'}")
    print(f"task         : {args.task}")
    print(f"magnifications: {args.magnifications}")
    print(f"image_size   : {img_size}")
    print(f"batch_size   : {args.batch_size}")
    print(f"mean/std     : {mean} / {std}")
    if not has_split:
        print(f"ratios       : train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print("-------------------\n")

    data = get_dataloaders(
        dataset_path=args.dataset_path,
        image_size=img_size,
        batch_size=args.batch_size,
        mean=mean,
        std=std,
        task=args.task,
        magnifications=mags,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print("Dataset sizes:", data["sizes"])
    print("num_classes  :", data.get("num_classes", len(data["class_names"])))
    print("class_names  :", data["class_names"])
    print()