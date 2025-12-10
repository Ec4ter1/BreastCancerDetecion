# test_and_report.py
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from dataset import MultiModalDataset, build_transforms
from model import MultiModalModel


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    numeric = torch.stack([b["numeric"] for b in batch])
    categorical = torch.stack([b["categorical"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    patient_ids = [b["patient_id"] for b in batch]
    return images, numeric, categorical, labels, patient_ids


def test_model(checkpoint_path, test_csv, image_root, output_dir, device="cuda"):
    """
    Testează modelul și generează raport complet
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ÎNCĂRCARE MODEL ȘI DATE")
    print("=" * 80)

    # STEP 1: Încarcă checkpoint-ul MAI ÎNTÂI
    print(f"Încărcare checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"✓ Checkpoint încărcat")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    # STEP 2: Extrage info din checkpoint
    cat_cardinalities = checkpoint.get('cat_cardinalities', [])
    num_numeric = checkpoint.get('num_numeric', 3)
    cat_maps = checkpoint.get('cat_maps', None)

    print(f"  Numeric features: {num_numeric}")
    print(f"  Categorical features: {len(cat_cardinalities)}")
    print(f"  Cat cardinalities: {cat_cardinalities}")

    if cat_maps is None:
        print("\n⚠ WARNING: cat_maps nu există în checkpoint!")
        print("  Modelul a fost antrenat cu versiune veche.")
        print("  Te rog re-antrenează cu: python train.py ...")
        return None, None

    # STEP 3: Încarcă dataset-ul de test cu mappings din checkpoint
    print(f"\nÎncărcare dataset test: {test_csv}")
    test_ds = MultiModalDataset(
        test_csv,
        image_root=image_root,
        transforms=build_transforms(False),
        cat_maps=cat_maps  # Folosește mappings din train
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"✓ Dataset test: {len(test_ds)} samples")

    # STEP 4: Construiește și încarcă modelul
    print(f"\nConstruire model...")
    model = MultiModalModel(
        num_numeric=num_numeric,
        cat_cardinalities=cat_cardinalities
    )
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model încărcat și gata de testare\n")

    # STEP 5: Predicții
    print("=" * 80)
    print("PREDICȚII PE TEST SET")
    print("=" * 80)

    all_labels = []
    all_probs = []
    all_preds = []
    all_patient_ids = []

    with torch.no_grad():
        for images, numeric, categorical, labels, patient_ids in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            numeric = numeric.to(device)
            categorical = categorical.to(device)

            logits = model(images, numeric, categorical)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_patient_ids.extend(patient_ids)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # STEP 6: Calculează metrici
    print("\n" + "=" * 80)
    print("METRICI DE PERFORMANȚĂ")
    print("=" * 80)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # Metrici per clasă
    prec_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    metrics = {
        'AUC-ROC': auc,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': recall,
        'F1-Score': f1,
        'Precision_BENIGN': prec_per_class[0],
        'Recall_BENIGN': recall_per_class[0],
        'F1_BENIGN': f1_per_class[0],
        'Precision_MALIGNANT': prec_per_class[1],
        'Recall_MALIGNANT': recall_per_class[1],
        'F1_MALIGNANT': f1_per_class[1],
        'Support_BENIGN': int(support[0]),
        'Support_MALIGNANT': int(support[1])
    }

    print(f"AUC-ROC:      {auc:.4f}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"\nPer Class Metrics:")
    print(
        f"  BENIGN     - Prec: {prec_per_class[0]:.4f}, Rec: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f} (n={support[0]})")
    print(
        f"  MALIGNANT  - Prec: {prec_per_class[1]:.4f}, Rec: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f} (n={support[1]})")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")

    # STEP 7: Salvează rezultatele detaliate
    results_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'true_label': all_labels,
        'predicted_label': all_preds,
        'probability_malignant': all_probs,
        'correct': all_labels == all_preds
    })

    results_csv = os.path.join(output_dir, 'test_results_detailed.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Rezultate detaliate salvate: {results_csv}")

    # Salvează metricile sumare
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = os.path.join(output_dir, 'test_metrics_summary.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"✓ Metrici sumare salvate: {metrics_csv}")

    # STEP 8: Generează grafice
    print("\n" + "=" * 80)
    print("GENERARE VIZUALIZĂRI")
    print("=" * 80)

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BENIGN', 'MALIGNANT'],
                yticklabels=['BENIGN', 'MALIGNANT'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix: {cm_path}")

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve: {roc_path}")

    # 3. Distribution of Predictions
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(all_probs[all_labels == 0], bins=30, alpha=0.7, label='BENIGN', color='blue')
    plt.hist(all_probs[all_labels == 1], bins=30, alpha=0.7, label='MALIGNANT', color='red')
    plt.xlabel('Predicted Probability (Malignant)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    error_benign = (all_labels == 0) & (all_preds == 1)  # False Positives
    error_malignant = (all_labels == 1) & (all_preds == 0)  # False Negatives
    plt.scatter(range(sum(error_benign)), all_probs[error_benign], c='orange', marker='x', s=50,
                label=f'False Positives (n={sum(error_benign)})')
    plt.scatter(range(sum(error_malignant)), all_probs[error_malignant], c='purple', marker='o', s=50,
                label=f'False Negatives (n={sum(error_malignant)})')
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.title('Misclassified Cases')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'prediction_distributions.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Distribuții predicții: {dist_path}")

    # 4. Classification Report
    report = classification_report(all_labels, all_preds, target_names=['BENIGN', 'MALIGNANT'], digits=4)
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("=" * 80 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:25s}: {v}\n")
    print(f"✓ Classification report: {report_path}")

    print("\n" + "=" * 80)
    print("TESTARE COMPLETĂ! 🎉")
    print("=" * 80)
    print(f"Toate rezultatele au fost salvate în: {output_dir}")

    return metrics, results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Calea către checkpoint-ul modelului")
    parser.add_argument("--test_csv", required=True, help="Calea către test.csv")
    parser.add_argument("--image_root", default=None, help="Root directory pentru imagini")
    parser.add_argument("--output_dir", default="./test_results", help="Director pentru rezultate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    test_model(args.checkpoint, args.test_csv, args.image_root, args.output_dir, args.device)