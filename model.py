import torch
import torch.nn as nn
import torchvision.models as models

class TabularNet(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities=None, cat_embed_dim=8, hidden=128, dropout=0.3):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities or []
        self.embeddings = nn.ModuleList()
        for card in self.cat_cardinalities:
            emb_dim = min(cat_embed_dim, (card+1)//2) if card>1 else 1
            self.embeddings.append(nn.Embedding(card+1, emb_dim))
        emb_total = sum(e.embedding_dim for e in self.embeddings) if self.embeddings else 0
        input_dim = num_numeric + emb_total
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )

    def forward(self, numeric, categorical):
        emb = []
        for i, emb_layer in enumerate(self.embeddings):
            emb.append(emb_layer(categorical[:, i]))
        if emb:
            x = torch.cat([numeric] + emb, dim=1)
        else:
            x = numeric
        return self.net(x)

class MultiModalModel(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities=None, image_backbone="resnet50", pretrained=True, fusion_hidden=256):
        super().__init__()
        # image branch
        if image_backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_feats = self.backbone.fc.in_features
            # replace fc
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError
        self.tabular = TabularNet(num_numeric=num_numeric, cat_cardinalities=cat_cardinalities, hidden=128)
        tab_out = 64
        self.img_proj = nn.Linear(in_feats, fusion_hidden)
        self.tab_proj = nn.Linear(64, fusion_hidden)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2*fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_hidden, 1)
        )

    def forward(self, image, numeric, categorical):
        img_feats = self.backbone(image)
        img_p = self.img_proj(img_feats)
        tab_feats = self.tabular(numeric, categorical)
        tab_p = self.tab_proj(tab_feats)
        fused = torch.cat([img_p, tab_p], dim=1)
        out = self.classifier(fused)
        return out.squeeze(1)  # logits
