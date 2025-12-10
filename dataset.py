# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
def build_transforms(is_training=True):
    """Transformări pentru imagini medicale"""
    if is_training:
        return T.Compose([
            T.Resize((512, 512)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class MultiModalDataset(Dataset):
    def __init__(self, csv_path, image_root=None, transforms=None, use_cropped=True, cat_maps=None):
        """
        Dataset multimodal pentru CBIS-DDSM

        Args:
            csv_path: Calea către CSV (train/val/test)
            image_root: Root directory pentru imagini (opțional, nu e folosit dacă căile sunt absolute)
            transforms: Transformări torchvision
            use_cropped: Dacă True, folosește cropped_image_path, altfel full_image_path
            cat_maps: Mappings pre-existente pentru categorice (pentru test/val consistency)
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transforms = transforms or build_transforms(False)
        self.use_cropped = use_cropped
        self.predefined_cat_maps = cat_maps

        self.numeric_cols = ['breast_density', 'assessment', 'subtlety']
        self.cat_cols = ['breast_side', 'image_view', 'abnormality_type']

        optional_cat = ['calc_type', 'calc_distribution', 'mass_shape', 'mass_margins']

        for col in optional_cat:
            if col in self.df.columns:
                self.cat_cols.append(col)

        self._prepare_data()

    def _prepare_data(self):
        self.df['label'] = (self.df['pathology'] == 'MALIGNANT').astype(int)

        for col in self.numeric_cols:
            if col in self.df.columns:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
            else:
                self.df[col] = 0

        self.cat_maps = {}

        if self.predefined_cat_maps is not None:
            self.cat_maps = self.predefined_cat_maps.copy()

            for col in self.cat_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('UNKNOWN')

                    def map_value(val):
                        if val in self.cat_maps[col]:
                            return val
                        else:
                            if 'UNKNOWN' in self.cat_maps[col]:
                                return 'UNKNOWN'
                            else:
                                return list(self.cat_maps[col].keys())[0]

                    self.df[col] = self.df[col].apply(map_value)
                else:
                    self.df[col] = 'UNKNOWN' if 'UNKNOWN' in self.cat_maps.get(col, {}) else \
                    list(self.cat_maps.get(col, {'UNKNOWN': 0}).keys())[0]
        else:
            for col in self.cat_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('UNKNOWN')
                    unique_vals = sorted(self.df[col].unique())
                    self.cat_maps[col] = {v: i for i, v in enumerate(unique_vals)}
                else:
                    self.cat_maps[col] = {'UNKNOWN': 0}
                    self.df[col] = 'UNKNOWN'

        print(f"Dataset pregătit: {len(self.df)} samples")
        print(f"  BENIGN: {(self.df['label'] == 0).sum()}")
        print(f"  MALIGNANT: {(self.df['label'] == 1).sum()}")
        print(f"\nColoane numerice: {self.numeric_cols}")
        print(f"Coloane categorice: {self.cat_cols}")
        for col, mapping in self.cat_maps.items():
            print(f"  {col}: {len(mapping)} categorii")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['cropped_image_path'] if self.use_cropped else row['full_image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transforms(image)
        except Exception as e:
            print(f"Eroare la încărcare imagine {img_path}: {e}")
            image = torch.zeros(3, 512, 512)
        numeric = torch.tensor([float(row[col]) for col in self.numeric_cols], dtype=torch.float32)
        categorical = torch.tensor([self.cat_maps[col][row[col]] for col in self.cat_cols], dtype=torch.long)

        label = float(row['label'])

        return {
            'image': image,
            'numeric': numeric,
            'categorical': categorical,
            'label': label,
            'patient_id': row['patient_id']
        }