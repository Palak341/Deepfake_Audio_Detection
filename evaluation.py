"""
model.py
--------
Feature Fusion + Embedding Model + Classifier.

FeatureFusion        : per-group LayerNorm + learned attention-gated fusion
                        of the raw hybrid feature groups (mfcc/cqcc/lfcc/...).
ProjectionHead        : projects fused features to an L2-normalized embedding.
GeneralEmbeddingModel : fusion -> projection -> embedding + logits (the
                        speaker-agnostic "general" detector).
PersonalizedRepresentation : merges the current embedding with a speaker
                        memory context (see src/memory.py) into a
                        personalized embedding.
DeepfakeClassifier     : predicts bona-fide/spoof from a (personalized)
                        embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import config


class FeatureFusion(nn.Module):
    def __init__(self, group_slices):
        super().__init__()
        self.group_slices = {name: (s, e) for name, (s, e) in group_slices.items() if e > s}
        self.group_names = list(self.group_slices.keys())
        self.norms = nn.ModuleDict({
            name: nn.LayerNorm(end - start) for name, (start, end) in self.group_slices.items()
        })
        total_dim = sum(end - start for start, end in self.group_slices.values())
        self.gate = nn.Sequential(
            nn.Linear(total_dim, len(self.group_names)),
        )

    def concatenateFeatures(self, x):
        return x

    def normalizeFeatures(self, x):
        parts = []
        for name in self.group_names:
            start, end = self.group_slices[name]
            parts.append(self.norms[name](x[:, start:end]))
        return torch.cat(parts, dim=1)

    def attentionFusion(self, x_norm):
        weights = torch.softmax(self.gate(x_norm), dim=1)
        parts = []
        for i, name in enumerate(self.group_names):
            start, end = self.group_slices[name]
            w = weights[:, i:i + 1]
            parts.append(x_norm[:, start:end] * w)
        return torch.cat(parts, dim=1), weights

    def forward(self, x):
        x = self.concatenateFeatures(x)
        x_norm = self.normalizeFeatures(x)
        fused, weights = self.attentionFusion(x_norm)
        return fused, weights


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or config.PROJ_HIDDEN
        out_dim = out_dim or config.EMBED_DIM
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def linearProjection(self, x):
        return self.linear1(x)

    def batchNormalize(self, x):
        return self.act(self.bn1(x))

    def generateEmbedding(self, x):
        emb = self.linear2(x)
        return F.normalize(emb, dim=1)

    def forward(self, x):
        x = self.linearProjection(x)
        x = self.batchNormalize(x)
        return self.generateEmbedding(x)


class GeneralEmbeddingModel(nn.Module):
    """Speaker-agnostic detector: fusion -> projection -> embedding, plus a
    lightweight classifier head trained jointly with a triplet loss on the
    embedding (see train.py)."""

    def __init__(self, fusion, projector):
        super().__init__()
        self.fusion = fusion
        self.projector = projector
        self.classifier = nn.Sequential(
            nn.Linear(config.EMBED_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        fused, attn_weights = self.fusion(x)
        embedding = self.projector(fused)
        logits = self.classifier(embedding)
        return embedding, logits, attn_weights


class PersonalizedRepresentation(nn.Module):
    """Merges the current sample's embedding with a speaker-memory context
    vector (produced by CrossAttentionModule, see src/memory.py) into a
    personalized embedding."""

    def __init__(self, dim=None, hidden=128):
        super().__init__()
        dim = dim or config.EMBED_DIM
        self.merge = nn.Linear(dim * 2, hidden)
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Linear(hidden, dim)

    def mergeContext(self, current_emb, speaker_context):
        return torch.cat([current_emb, speaker_context], dim=1)

    def generatePersonalizedEmbedding(self, merged):
        return self.out(self.act(self.merge(merged)))

    def forward(self, current_emb, speaker_context):
        merged = self.mergeContext(current_emb, speaker_context)
        return self.generatePersonalizedEmbedding(merged)


class DeepfakeClassifier(nn.Module):
    """predict / calculateProbability / classifyAudio."""

    def __init__(self, dim=None):
        super().__init__()
        dim = dim or config.EMBED_DIM
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def predict(self, personalized_emb):
        return self.mlp(personalized_emb)

    def calculateProbability(self, logits):
        return torch.softmax(logits, dim=1)

    def classifyAudio(self, logits):
        probs = self.calculateProbability(logits)
        pred = probs.argmax(dim=1)
        return pred, probs

    def forward(self, personalized_emb):
        logits = self.predict(personalized_emb)
        return self.classifyAudio(logits)


def build_general_model(group_slices, device=None):
    """Convenience constructor used by train.py / inference.py."""
    from src.utils import DEVICE
    device = device or DEVICE
    fusion = FeatureFusion(group_slices).to(device)
    projector = ProjectionHead(config.TOTAL_RAW_DIM).to(device)
    model = GeneralEmbeddingModel(fusion, projector).to(device)
    return model
