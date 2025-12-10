import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_dataset(merged_csv, output_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Împarte dataset-ul în train/val/test păstrând distribuția
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(merged_csv)
    print(f"Dataset total: {len(df)} înregistrări")
    if 'BENIGN_WITHOUT_CALLBACK' in df['pathology'].values:
        print("\n⚠ Găsit BENIGN_WITHOUT_CALLBACK - se unifică cu BENIGN")
        df['pathology'] = df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
    print("\nDistribuție pathology:")
    print(df['pathology'].value_counts())

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['pathology']
    )
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val['pathology']
    )
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "val.csv")
    test_csv = os.path.join(output_dir, "test.csv")

    train.to_csv(train_csv, index=False)
    val.to_csv(val_csv, index=False)
    test.to_csv(test_csv, index=False)

    print("\n" + "=" * 80)
    print("SPLIT FINALIZAT")
    print("=" * 80)
    print(f"Train: {len(train)} ({100 * len(train) / len(df):.1f}%)")
    print(f"  BENIGN: {(train['pathology'] == 'BENIGN').sum()}")
    print(f"  MALIGNANT: {(train['pathology'] == 'MALIGNANT').sum()}")
    print(f"\nVal: {len(val)} ({100 * len(val) / len(df):.1f}%)")
    print(f"  BENIGN: {(val['pathology'] == 'BENIGN').sum()}")
    print(f"  MALIGNANT: {(val['pathology'] == 'MALIGNANT').sum()}")
    print(f"\nTest: {len(test)} ({100 * len(test) / len(df):.1f}%)")
    print(f"  BENIGN: {(test['pathology'] == 'BENIGN').sum()}")
    print(f"  MALIGNANT: {(test['pathology'] == 'MALIGNANT').sum()}")
    return train_csv, val_csv, test_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Calea către merged_dataset.csv")
    parser.add_argument("--output_dir", default="./splits", help="Director pentru salvare")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proporție test set")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proporție validation set")
    args = parser.parse_args()

    split_dataset(args.input, args.output_dir, args.test_size, args.val_size)