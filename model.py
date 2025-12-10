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
