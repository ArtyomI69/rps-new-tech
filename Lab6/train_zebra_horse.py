# -*- coding: utf-8 -*-
"""
Классификация изображений: zebra vs bay_horse.
Обучение по train.csv, тестирование по test.csv.
"""
import os
import io
import csv
import random
import argparse
import datetime
import logging
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ----------------------------- transforms -----------------------------
class JpegCompression:
    """Случайное JPEG-сжатие (имитация потери качества)."""
    def __init__(self, quality_range=(40, 90)):
        self.quality_range = quality_range

    def __call__(self, img):
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ----------------------------- Dataset -----------------------------
CLASS_TO_IDX = {"bay_horse": 0, "zebra": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class ImageCSVDataset(Dataset):
    def __init__(self, csv_path, base_dir, transform=None, log=None):
        self.transform = transform
        self.samples = []

        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]

        skipped = 0
        for _, row in df.iterrows():
            abs_p = str(row["absolute_path"]).strip()
            rel_p = str(row["relative_path"]).strip().replace("\\", os.sep)
            label = str(row["class_label"]).strip().lower()
            if label not in CLASS_TO_IDX:
                skipped += 1
                continue

            path = abs_p if os.path.exists(abs_p) else os.path.join(base_dir, rel_p)
            if not os.path.exists(path):
                skipped += 1
                continue

            self.samples.append((path, CLASS_TO_IDX[label]))

        if log:
            log.info(f"{csv_path}: загружено {len(self.samples)} изображений, пропущено {skipped}")
            cnt = Counter(IDX_TO_CLASS[lbl] for _, lbl in self.samples)
            log.info(f"  распределение классов: {dict(cnt)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label, path


# ----------------------------- Model -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------- metrics -----------------------------
def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    metrics = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[IDX_TO_CLASS[i]] = dict(precision=prec, recall=rec, f1=f1, support=int(cm[i, :].sum()))
    return metrics


# ----------------------------- helpers -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{ts}.log")

    logger = logging.getLogger("zebra_horse")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_path, ts


def plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    p = os.path.join(out_dir, "training_curve.png")
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def evaluate(model, loader, device, criterion=None):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    y_true, y_pred, wrong_paths = [], [], []

    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            if criterion is not None:
                loss_sum += criterion(logits, labels).item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            for p, t, pr in zip(paths, labels.cpu().tolist(), preds.cpu().tolist()):
                if t != pr:
                    wrong_paths.append((p, IDX_TO_CLASS[t], IDX_TO_CLASS[pr]))

    acc = correct / total if total else 0.0
    avg_loss = loss_sum / total if total and criterion else None
    return acc, avg_loss, y_true, y_pred, wrong_paths


# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--test-csv",  default="test.csv")
    parser.add_argument("--val-csv", default="val.csv")
    parser.add_argument("--base-dir",  default=".")
    parser.add_argument("--out-dir",   default="outputs")
    parser.add_argument("--epochs",    type=int, default=10)
    parser.add_argument("--batch-size",type=int, default=16)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--img-size",  type=int, default=224)
    parser.add_argument("--workers",   type=int, default=0)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    logger, log_path, ts = setup_logger(args.out_dir)
    logger.info("=== Старт обучения ===")
    logger.info(f"Аргументы: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Устройство: {device}")

    # ---- transforms ----
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        JpegCompression(quality_range=(40, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- datasets ----
    train_ds = ImageCSVDataset(args.train_csv, args.base_dir, transform=train_tf, log=logger)

    val_ds = ImageCSVDataset(args.val_csv, args.base_dir, transform=eval_tf, log=logger)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers)

    test_ds = ImageCSVDataset(args.test_csv, args.base_dir, transform=eval_tf, log=logger)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)

    logger.info(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ---- model ----
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- training ----
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        tr_loss = running_loss / total
        tr_acc  = correct / total
        val_acc, val_loss, _, _, _ = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(f"Epoch {epoch:02d}/{args.epochs} | "
                    f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logger.info(f"  → сохранена лучшая модель (val_acc={val_acc:.4f})")

    curve_path = plot_history(history, args.out_dir)
    logger.info(f"График обучения: {curve_path}")

    # ---- testing ----
    logger.info("=== Тестирование на демо-данных (test.csv) ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, test_loss, y_true, y_pred, wrong = evaluate(model, test_loader, device, criterion)

    cm = confusion_matrix(y_true, y_pred, n=2)
    metrics = per_class_metrics(cm)

    logger.info(f"Test loss:     {test_loss:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Confusion matrix (rows=true, cols=pred):\n{cm}")
    for cls, m in metrics.items():
        logger.info(f"  {cls:10s} precision={m['precision']:.4f} "
                    f"recall={m['recall']:.4f} f1={m['f1']:.4f} support={m['support']}")
    if wrong:
        logger.info(f"Ошибочно классифицировано {len(wrong)} из {len(y_true)}:")
        for p, t, pr in wrong[:20]:
            logger.info(f"    {os.path.basename(p)}  true={t}  pred={pr}")
    else:
        logger.info("Ошибок нет — все изображения классифицированы верно.")

    # ---- separate text report ----
    report_path = os.path.join(args.out_dir, f"report_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Отчёт о классификации изображений: zebra vs bay_horse\n")
        f.write(f"Дата: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Устройство: {device}\n")
        f.write(f"train.csv: {args.train_csv}\n")
        f.write(f"val.csv  : {args.val_csv}\n")
        f.write(f"test.csv : {args.test_csv}\n")
        f.write(f"Эпох: {args.epochs}, batch={args.batch_size}, lr={args.lr}, img={args.img_size}x{args.img_size}\n")
        f.write(f"Обучающих: {len(train_ds)}, валидационных: {len(val_ds)}, тестовых: {len(test_ds)}\n\n")

        f.write("--- ТРАНСФОРМАЦИИ ---\n")
        f.write("  Train: Resize, RandomHorizontalFlip, RandomRotation(15),\n"
                "         ColorJitter(brightness/contrast/saturation=0.3, hue=0.05),\n"
                "         JpegCompression(quality=40..90), Normalize\n")
        f.write("  Eval : Resize, Normalize\n\n")

        f.write("--- РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ---\n")
        for i, (tl, ta, vl, va) in enumerate(zip(history["train_loss"], history["train_acc"],
                                                 history["val_loss"], history["val_acc"]), 1):
            f.write(f"  epoch {i:02d}: train_loss={tl:.4f} train_acc={ta:.4f} "
                    f"val_loss={vl:.4f} val_acc={va:.4f}\n")
        f.write(f"\nЛучший val_acc: {best_val_acc:.4f}\n\n")

        f.write("--- РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ---\n")
        f.write(f"Test loss     : {test_loss:.4f}\n")
        f.write(f"Test accuracy : {test_acc:.4f}\n\n")
        f.write("Confusion matrix (строки = истина, столбцы = предсказание):\n")
        f.write("            pred_bay_horse  pred_zebra\n")
        f.write(f"true_bay_horse  {cm[0,0]:>10d}  {cm[0,1]:>10d}\n")
        f.write(f"true_zebra      {cm[1,0]:>10d}  {cm[1,1]:>10d}\n\n")
        f.write("Метрики по классам:\n")
        for cls, m in metrics.items():
            f.write(f"  {cls:10s}: precision={m['precision']:.4f} "
                    f"recall={m['recall']:.4f} f1={m['f1']:.4f} support={m['support']}\n")
        f.write("\n")
        if wrong:
            f.write(f"Ошибочно классифицировано ({len(wrong)}):\n")
            for p, t, pr in wrong:
                f.write(f"  {p}  true={t}  pred={pr}\n")
        else:
            f.write("Все тестовые изображения классифицированы верно.\n")

    logger.info(f"Отчёт сохранён: {report_path}")
    logger.info(f"Лог сохранён: {log_path}")
    logger.info("=== Готово ===")


if __name__ == "__main__":
    main()