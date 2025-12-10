# train.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import random
import os
import json
from datetime import datetime
from dataset import MultiModalDataset, build_transforms
from model import MultiModalModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Pentru reproducibilitate completă
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """Custom collate function pentru DataLoader"""
    images = torch.stack([b["image"] for b in batch])
    numeric = torch.stack([b["numeric"] for b in batch])
    categorical = torch.stack([b["categorical"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return images, numeric, categorical, labels


def evaluate(model, dataloader, device):
    """Evaluează modelul pe un dataset"""
    model.eval()
    ys = []
    preds = []
    probs = []

    with torch.no_grad():
        for images, numeric, categorical, labels in dataloader:
            images = images.to(device)
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)

            logits = model(images, numeric, categorical)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).long()

            ys.extend(labels.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            probs.extend(prob.cpu().numpy().tolist())

    # Calculează metrici
    auc = None
    try:
        auc = roc_auc_score(ys, probs)
    except Exception as e:
        print(f"Warning: Nu s-a putut calcula AUC - {e}")
        auc = float("nan")

    acc = accuracy_score(ys, preds)
    prec, recall, f1, _ = precision_recall_fscore_support(ys, preds, average="binary", zero_division=0)

    return {
        "auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1
    }


def train_loop(train_csv, val_csv, image_root, out_dir, epochs=20, batch_size=8, lr=1e-4, device="cuda"):
    """
    Main training loop

    Args:
        train_csv: Calea către train.csv
        val_csv: Calea către val.csv
        image_root: Root pentru imagini (nu e folosit dacă căile sunt absolute)
        out_dir: Directory pentru salvare checkpoints
        epochs: Număr de epoci
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' sau 'cpu'
    """
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("CONFIGURARE TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("=" * 80 + "\n")

    # Încarcă datasets
    print("Încărcare dataset train...")
    train_ds = MultiModalDataset(train_csv, image_root=image_root, transforms=build_transforms(True))
    print("\nÎncărcare dataset validare...")
    val_ds = MultiModalDataset(val_csv, image_root=image_root, transforms=build_transforms(False))

    # Reduce num_workers dacă ai probleme cu memoria
    num_workers = 0 if device == 'cpu' else 2  # Windows poate avea probleme cu num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )

    print(f"\n✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches\n")

    # Construiește modelul
    cat_cardinalities = []
    for c in train_ds.cat_cols:
        if c in train_ds.cat_maps:
            cat_cardinalities.append(len(train_ds.cat_maps[c]))

    num_numeric = len(train_ds.numeric_cols)

    print("=" * 80)
    print("CONSTRUIRE MODEL")
    print("=" * 80)
    print(f"Numeric features: {num_numeric}")
    print(f"Categorical features: {len(cat_cardinalities)}")
    print(f"Categorical cardinalities: {cat_cardinalities}")
    print("=" * 80 + "\n")

    model = MultiModalModel(
        num_numeric=num_numeric,
        cat_cardinalities=cat_cardinalities
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Optimizer și loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Learning rate scheduler (opțional)
    # Removed 'verbose' argument for compatibility with older PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Tracking pentru best model
    best_val_auc = -1.0
    training_history = {
        'train_loss': [],
        'val_metrics': []
    }

    print("=" * 80)
    print("START TRAINING")
    print("=" * 80 + "\n")

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (images, numeric, categorical, labels) in enumerate(pbar):
            images = images.to(device)
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images, numeric, categorical)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (previne exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        epoch_loss = running_loss / len(train_loader)
        training_history['train_loss'].append(epoch_loss)

        # Validare
        print(f"\nEvaluare pe validation set...")
        val_metrics = evaluate(model, val_loader, device)
        training_history['val_metrics'].append(val_metrics)

        print(f"\nEpoch {epoch}/{epochs} Results:")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val AUC:    {val_metrics['auc']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
        print(f"  Val Prec:   {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1:     {val_metrics['f1']:.4f}")

        # Update learning rate
        if not np.isnan(val_metrics["auc"]):
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics["auc"])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        # Salvare best checkpoint
        if not np.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            checkpoint = {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "train_loss": epoch_loss,
                "cat_cardinalities": cat_cardinalities,
                "num_numeric": num_numeric,
                # IMPORTANT: Salvează mappings-urile pentru categorice
                "cat_maps": train_ds.cat_maps,
                "cat_cols": train_ds.cat_cols,
                "numeric_cols": train_ds.numeric_cols
            }
            checkpoint_path = os.path.join(out_dir, "best_checkpoint.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved best checkpoint (AUC: {best_val_auc:.4f})")

        print("-" * 80 + "\n")

    # Salvare training history
    history_path = os.path.join(out_dir, "training_history.json")
    with open(history_path, 'w') as f:
        # Convert np.float to float for JSON serialization
        history_serializable = {
            'train_loss': [float(x) for x in training_history['train_loss']],
            'val_metrics': [
                {k: float(v) if not np.isnan(v) else None for k, v in m.items()}
                for m in training_history['val_metrics']
            ]
        }
        json.dump(history_serializable, f, indent=2)
    print(f"✓ Training history salvat: {history_path}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    final_metrics = evaluate(model, val_loader, device)
    print(f"Final Val AUC:      {final_metrics['auc']:.4f}")
    print(f"Final Val Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final Val F1:       {final_metrics['f1']:.4f}")
    print(f"\nBest Val AUC:       {best_val_auc:.4f}")
    print("=" * 80 + "\n")

    return final_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multimodal breast cancer detection model")
    parser.add_argument("--train_csv", required=True, help="Path to train.csv")
    parser.add_argument("--val_csv", required=True, help="Path to val.csv")
    parser.add_argument("--image_root", default=None, help="Root directory for images (optional)")
    parser.add_argument("--out_dir", default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, default: auto-detect)")

    args = parser.parse_args()

    # Auto-detect device dacă nu e specificat
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loop(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_root=args.image_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )