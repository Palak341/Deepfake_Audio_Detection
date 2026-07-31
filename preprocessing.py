"""
memory.py
---------
Memory Bank + Retrieval + Cross Attention.

MemoryBank              : persistent, (optionally FAISS-indexed) per-speaker
                           store of trusted embeddings, SQLite-backed.
CrossAttentionModule     : attends the current embedding over retrieved
                           speaker-memory references to build a context vector.
SpeakerContextBuilder    : convenience wrapper for building speaker profiles.
AdaptiveMemoryRetriever  : dynamic Top-K retrieval driven by an
                           acoustic-complexity estimate per speaker.
MemoryAgingManager       : decays older embeddings' influence over time.
PrototypeMemoryManager   : compresses a speaker's raw embeddings into a
                           bounded set of representative centroids.
MemoryQualityController  : gatekeeper deciding whether a new embedding is
                           trustworthy enough to enter the memory bank.
ConfidenceGate           : confidence/physics/similarity verification gate.
MemoryUpdateController   : orchestrates Insert -> Age-Refresh ->
                           Prototype-Compress -> Reindex as one update.
"""

import sqlite3
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, MiniBatchKMeans

from src.utils import HAS_FAISS, config

if HAS_FAISS:
    import faiss


# --------------------------------------------------------------------------- #
# Memory Bank
# --------------------------------------------------------------------------- #
class MemoryBank:
    """Persistent, FAISS-indexed, per-speaker store of trusted embeddings.
    Extended with `age_weight`, `physics_vector`, and `cluster_size` columns
    so it can support Memory Aging and Prototype Memory Compression on top
    of the base insert/retrieve behaviour."""

    def __init__(self, dim=None, physics_dim=None,
                 db_path=None, index_path=None, reset_db=None):
        self.dim = dim or config.EMBED_DIM
        self.physics_dim = physics_dim or config.DIM_PHYSICS
        self.db_path = db_path or config.MEMORY_DB_PATH
        self.index_path = index_path or config.FAISS_INDEX_PATH
        reset_db = config.RESET_MEMORY_DB if reset_db is None else reset_db
        self._init_db(reset=reset_db)
        self.buildIndex()

    def _init_db(self, reset=True):
        """`reset=True` wipes any pre-existing DB (fine for a fresh
        experimental run, but destroys all previously-learned speaker
        memory). In deployment, construct MemoryBank(reset_db=False) so
        memory persists across process restarts instead of resetting every
        run."""
        import os
        if reset and os.path.exists(self.db_path):
            print(f"[MemoryBank] reset_db=True -> wiping existing memory at {self.db_path}")
            os.remove(self.db_path)
        elif os.path.exists(self.db_path):
            print(f"[MemoryBank] reset_db=False -> reusing existing memory at {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id     TEXT NOT NULL,
                embedding      BLOB NOT NULL,
                timestamp      REAL NOT NULL,
                confidence     REAL NOT NULL,
                age_weight     REAL NOT NULL DEFAULT 1.0,
                physics_vector BLOB,
                cluster_size   INTEGER NOT NULL DEFAULT 1
            )
        """)
        self.conn.commit()

    def buildIndex(self):
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = None  # brute-force numpy fallback
        self._id_order = []
        self._refresh_from_db()

    def _refresh_from_db(self):
        rows = self.conn.execute("SELECT id, embedding FROM embeddings ORDER BY id").fetchall()
        self._id_order = [r[0] for r in rows]
        if len(rows) == 0:
            if HAS_FAISS:
                self.index = faiss.IndexFlatIP(self.dim)
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            return
        mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).astype(np.float32)
        self._matrix = mat
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(mat)

    def rebuildIndex(self):
        """Full-table reload of the global FAISS/_matrix index. Expensive
        (O(n) scan + full FAISS rebuild across ALL speakers) -- nothing in
        the retrieval path (getSpeakerRows/getSpeakerMatrix, used by every
        Top-K / physics-mean / similarity lookup) actually reads
        self._matrix or self.index, so this should be called sparingly
        (e.g. once at the end of a replay), not after every single write."""
        self._refresh_from_db()

    def addEmbedding(self, speaker_id, embedding, confidence, timestamp=None,
                      physics_vector=None, cluster_size=1, age_weight=1.0):
        """NOTE: does NOT rebuild the global FAISS/_matrix index (see
        rebuildIndex() docstring) -- every retrieval path
        (getSpeakerRows/getSpeakerMatrix) reads straight from SQLite by
        speaker_id, so reindexing on every single insert was pure O(n)
        wasted work per write (O(n^2) over a full replay) for an index
        nothing ever queries. Call rebuildIndex() explicitly if/when you
        actually need the global index refreshed."""
        timestamp = time.time() if timestamp is None else timestamp
        emb = np.asarray(embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        phys_blob = None
        if physics_vector is not None:
            phys_blob = np.asarray(physics_vector, dtype=np.float32).tobytes()
        self.conn.execute(
            "INSERT INTO embeddings (speaker_id, embedding, timestamp, confidence, "
            "age_weight, physics_vector, cluster_size) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (speaker_id, emb.tobytes(), timestamp, float(confidence),
             float(age_weight), phys_blob, int(cluster_size)),
        )
        self.conn.commit()

    def deleteEmbedding(self, entry_id):
        self.conn.execute("DELETE FROM embeddings WHERE id = ?", (entry_id,))
        self.conn.commit()

    def removeOutdatedEntries(self, speaker_id, max_entries=None):
        """Hard safety-net cap. Prototype Memory Compression normally keeps
        a speaker's raw-entry count well under this ceiling; this only
        fires if compression is skipped or unavailable."""
        max_entries = max_entries or config.MAX_ENTRIES_PER_SPEAKER
        rows = self.conn.execute(
            "SELECT id FROM embeddings WHERE speaker_id = ? ORDER BY timestamp DESC", (speaker_id,)
        ).fetchall()
        if len(rows) > max_entries:
            stale_ids = [r[0] for r in rows[max_entries:]]
            self.conn.executemany("DELETE FROM embeddings WHERE id = ?", [(i,) for i in stale_ids])
            self.conn.commit()

    def replaceSpeakerEntries(self, speaker_id, centroids, confidences, cluster_sizes,
                               physics_centroids=None, now=None):
        """Prototype Memory Compression write-back: atomically swaps a
        speaker's raw embeddings for ~PROTOTYPE_TARGET_COUNT representative,
        cluster-weighted centroids."""
        now = now if now is not None else time.time()
        self.conn.execute("DELETE FROM embeddings WHERE speaker_id = ?", (speaker_id,))
        for i in range(len(centroids)):
            emb = np.asarray(centroids[i], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            phys = physics_centroids[i] if physics_centroids is not None else None
            phys_blob = np.asarray(phys, dtype=np.float32).tobytes() if phys is not None else None
            self.conn.execute(
                "INSERT INTO embeddings (speaker_id, embedding, timestamp, confidence, "
                "age_weight, physics_vector, cluster_size) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (speaker_id, emb.tobytes(), now, float(confidences[i]), 1.0, phys_blob, int(cluster_sizes[i])),
            )
        self.conn.commit()

    def bulkUpdateAgeWeights(self, updates):
        """updates: list of (age_weight, id) tuples -- used by MemoryAgingManager."""
        if updates:
            self.conn.executemany("UPDATE embeddings SET age_weight = ? WHERE id = ?", updates)
            self.conn.commit()

    def updateSpeakerProfile(self, speaker_id, embedding, confidence, physics_vector=None):
        """Legacy convenience wrapper (plain insert + hard-cap safety net).
        The main deployment loop instead goes through
        MemoryUpdateController.processUpdate, which layers Aging +
        Prototype Compression on top of this."""
        self.addEmbedding(speaker_id, embedding, confidence, physics_vector=physics_vector)
        self.removeOutdatedEntries(speaker_id)

    def computeSimilarity(self, query_emb, candidate_mat):
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        c = candidate_mat / (np.linalg.norm(candidate_mat, axis=1, keepdims=True) + 1e-9)
        return c @ q

    def getSpeakerRows(self, speaker_id):
        rows = self.conn.execute(
            "SELECT id, embedding, timestamp, confidence, age_weight, physics_vector, cluster_size "
            "FROM embeddings WHERE speaker_id = ?", (speaker_id,)
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "embedding": np.frombuffer(r[1], dtype=np.float32),
                "timestamp": r[2],
                "confidence": r[3],
                "age_weight": r[4],
                "physics_vector": np.frombuffer(r[5], dtype=np.float32) if r[5] is not None else None,
                "cluster_size": r[6],
            })
        return out

    def getSpeakerMatrix(self, speaker_id):
        rows = self.getSpeakerRows(speaker_id)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32), np.array([])
        mat = np.stack([r["embedding"] for r in rows])
        weights = np.array([r["age_weight"] for r in rows], dtype=np.float32)
        return mat, weights

    def getSpeakerPhysicsMean(self, speaker_id):
        rows = [r for r in self.getSpeakerRows(speaker_id) if r["physics_vector"] is not None]
        if not rows:
            return None
        mat = np.stack([r["physics_vector"] for r in rows])
        weights = np.array([r["age_weight"] for r in rows], dtype=np.float32)
        return np.average(mat, axis=0, weights=weights)

    def retrieveWeightedReferences(self, speaker_id, query_emb, k):
        """Age-weighted Top-K genuine references so recent samples are
        preferred over stale ones (Memory Aging integration)."""
        rows = self.getSpeakerRows(speaker_id)
        if not rows:
            return (np.zeros((0, self.dim), dtype=np.float32), np.array([]), np.array([]))
        mat = np.stack([r["embedding"] for r in rows])
        age_w = np.array([r["age_weight"] for r in rows], dtype=np.float32)
        sims = self.computeSimilarity(np.asarray(query_emb, dtype=np.float32), mat)
        score = sims * age_w
        k = min(k, mat.shape[0])
        top_idx = np.argsort(-score)[:k]
        return mat[top_idx], sims[top_idx], age_w[top_idx]

    def retrieveReferences(self, speaker_id, query_emb, k=None):
        """Fixed Top-K convenience wrapper."""
        k = k or config.MEMORY_TOPK
        refs, sims, _ = self.retrieveWeightedReferences(speaker_id, query_emb, k)
        return refs, sims

    def hasSpeaker(self, speaker_id):
        cnt = self.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE speaker_id = ?", (speaker_id,)
        ).fetchone()[0]
        return cnt > 0

    def speakerCount(self, speaker_id):
        return self.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE speaker_id = ?", (speaker_id,)
        ).fetchone()[0]

    def allSpeakerIds(self):
        rows = self.conn.execute("SELECT DISTINCT speaker_id FROM embeddings").fetchall()
        return [r[0] for r in rows]


# --------------------------------------------------------------------------- #
# Cross Attention + Personalized context
# --------------------------------------------------------------------------- #
class CrossAttentionModule(nn.Module):
    def __init__(self, dim=None, n_heads=4):
        super().__init__()
        dim = dim or config.EMBED_DIM
        self.dim = dim
        self.n_heads = n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = (dim // n_heads) ** -0.5

    def generateQuery(self, current_emb):
        return self.q_proj(current_emb)

    def generateKeys(self, retrieved_embs):
        return self.k_proj(retrieved_embs)

    def generateValues(self, retrieved_embs):
        return self.v_proj(retrieved_embs)

    def computeAttention(self, q, k):
        scores = torch.einsum("bd,bkd->bk", q, k) * self.scale
        return torch.softmax(scores, dim=-1)

    def aggregateContext(self, attn_weights, v):
        context = torch.einsum("bk,bkd->bd", attn_weights, v)
        return self.out_proj(context)

    def forward(self, current_emb, retrieved_embs, retrieved_mask=None):
        """current_emb: (B, D); retrieved_embs: (B, K, D) zero-padded if K varies."""
        q = self.generateQuery(current_emb)
        k = self.generateKeys(retrieved_embs)
        v = self.generateValues(retrieved_embs)
        scores = torch.einsum("bd,bkd->bk", q, k) * self.scale
        if retrieved_mask is not None:
            scores = scores.masked_fill(~retrieved_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        context = self.aggregateContext(attn, v)
        return context, attn


class SpeakerContextBuilder:
    """extractSpeakerTraits / buildSpeakerProfile."""

    def __init__(self, memory_bank):
        self.memory_bank = memory_bank

    def extractSpeakerTraits(self, speaker_id):
        refs, sims = self.memory_bank.retrieveReferences(
            speaker_id, np.zeros(config.EMBED_DIM), k=config.MAX_ENTRIES_PER_SPEAKER)
        if refs.shape[0] == 0:
            return None
        return {"count": refs.shape[0], "mean_embedding": refs.mean(axis=0)}

    def buildSpeakerProfile(self, speaker_id):
        return self.extractSpeakerTraits(speaker_id)


# --------------------------------------------------------------------------- #
# Adaptive retrieval
# --------------------------------------------------------------------------- #
class AdaptiveMemoryRetriever:
    """Replaces a fixed Top-K lookup with a Dynamic Top-K: speakers whose
    stored memory already agrees tightly with the incoming sample ("easy")
    are searched shallowly, while speakers whose memory is spread out or
    disagrees with the incoming sample ("complex") get a deeper search
    (Easy Speaker -> Top-3, Complex Speaker -> Top-10 by default)."""

    def __init__(self, memory_bank, k_min=None, k_max=None):
        self.memory_bank = memory_bank
        self.k_min = k_min or config.ADAPTIVE_TOPK_MIN
        self.k_max = k_max or config.ADAPTIVE_TOPK_MAX

    def buildFAISS(self):
        self.memory_bank.rebuildIndex()

    def estimateComplexity(self, speaker_id, query_emb):
        """Acoustic-complexity proxy in [0, 1]: combines (a) how spread out
        the speaker's stored embeddings are relative to the query, and (b)
        how much, on average, they disagree with the query.

        NOTE: `0.5*spread + 0.5*disagreement` is an empirically-chosen
        heuristic, not a theoretically-derived optimum -- the 0.5/0.5
        weighting and the choice of spread+disagreement as the two signals
        were not tuned via ablation."""
        mat, _ = self.memory_bank.getSpeakerMatrix(speaker_id)
        if mat.shape[0] < 2:
            return 0.0
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = m @ q
        spread = float(sims.std())
        disagreement = float(max(1.0 - sims.mean(), 0.0))
        complexity = 0.5 * spread + 0.5 * disagreement
        return float(np.clip(complexity, 0.0, 1.0))

    def dynamicTopK(self, complexity):
        span = self.k_max - self.k_min
        k = self.k_min + int(round(complexity * span))
        return int(np.clip(k, self.k_min, self.k_max))

    def retrieveNearest(self, speaker_id, query_emb):
        """Returns (refs, sims, age_weights, k_used, complexity)."""
        complexity = self.estimateComplexity(speaker_id, query_emb)
        k = self.dynamicTopK(complexity)
        refs, sims, age_w = self.memory_bank.retrieveWeightedReferences(speaker_id, query_emb, k=k)
        return refs, sims, age_w, k, complexity


# --------------------------------------------------------------------------- #
# Memory aging
# --------------------------------------------------------------------------- #
class MemoryAgingManager:
    """Assigns higher weight to recent embeddings and decays older ones, so
    the system naturally adapts to a speaker's aging voice or new
    microphones instead of treating an old sample the same as a new one."""

    def __init__(self, decay_per_year=None, min_weight=None):
        self.decay_per_year = decay_per_year or config.AGE_DECAY_PER_YEAR
        self.min_weight = min_weight or config.MIN_AGE_WEIGHT

    def assignAgeWeight(self, timestamp, now=None):
        now = now if now is not None else time.time()
        years_old = max(now - timestamp, 0.0) / config.SECONDS_PER_YEAR
        weight = (1.0 - self.decay_per_year) ** years_old
        return float(max(weight, self.min_weight))

    def decayOldEmbeddings(self, memory_bank, speaker_id, now=None):
        rows = memory_bank.getSpeakerRows(speaker_id)
        updates = [(self.assignAgeWeight(r["timestamp"], now), r["id"]) for r in rows]
        memory_bank.bulkUpdateAgeWeights(updates)
        return len(updates)

    def refreshWeights(self, memory_bank, speaker_id=None):
        """Recompute age weights for one speaker, or, if speaker_id is None,
        for every enrolled speaker in the memory bank."""
        speakers = [speaker_id] if speaker_id is not None else memory_bank.allSpeakerIds()
        total = 0
        for sid in speakers:
            total += self.decayOldEmbeddings(memory_bank, sid)
        return total


# --------------------------------------------------------------------------- #
# Prototype compression
# --------------------------------------------------------------------------- #
class PrototypeMemoryManager:
    """Instead of storing every raw embedding forever, this compresses a
    speaker's memory into ~PROTOTYPE_TARGET_COUNT representative,
    age-weighted centroids once the raw count crosses
    PROTOTYPE_TRIGGER_RAW_COUNT -- keeping search fast and the memory
    footprint bounded."""

    def __init__(self, target_count=None, trigger_count=None):
        self.target_count = target_count or config.PROTOTYPE_TARGET_COUNT
        self.trigger_count = trigger_count or config.PROTOTYPE_TRIGGER_RAW_COUNT

    def removeOutliers(self, mat, weights, z_thresh=2.5):
        """Drops embeddings whose distance from the (weighted) speaker
        centroid is an outlier, so a single mislabeled/noisy sample can't
        distort a cluster."""
        center = np.average(mat, axis=0, weights=weights)
        dists = np.linalg.norm(mat - center, axis=1)
        if dists.std() < 1e-6:
            return np.ones(len(mat), dtype=bool)
        z = (dists - dists.mean()) / (dists.std() + 1e-9)
        return z < z_thresh

    def clusterEmbeddings(self, mat, weights, k):
        """Uses KMeans for small speaker pools; switches to MiniBatchKMeans
        once a speaker's raw embedding count reaches
        PROTOTYPE_MINIBATCH_THRESHOLD, since full KMeans becomes noticeably
        slower to re-fit on every compression pass once pools grow into the
        thousands."""
        k = max(1, min(k, mat.shape[0]))
        if mat.shape[0] >= config.PROTOTYPE_MINIBATCH_THRESHOLD:
            km = MiniBatchKMeans(
                n_clusters=k, n_init=4, random_state=config.RANDOM_SEED,
                batch_size=min(256, mat.shape[0]),
            )
        else:
            km = KMeans(n_clusters=k, n_init=4, random_state=config.RANDOM_SEED)
        labels = km.fit_predict(mat, sample_weight=weights)
        return labels, k

    def computeCentroids(self, mat, labels, confidences, weights, physics_mat, n_clusters):
        centroids, centroid_conf, centroid_size, centroid_phys = [], [], [], []
        for c in range(n_clusters):
            mask = labels == c
            if not mask.any():
                continue
            w = weights[mask]
            centroids.append(np.average(mat[mask], axis=0, weights=w))
            centroid_conf.append(float(np.average(confidences[mask], weights=w)))
            centroid_size.append(int(mask.sum()))
            if physics_mat is not None:
                pm = physics_mat[mask]
                valid = ~np.all(pm == 0, axis=1)
                if valid.any():
                    centroid_phys.append(np.average(pm[valid], axis=0, weights=w[valid]))
                else:
                    centroid_phys.append(np.zeros(pm.shape[1], dtype=np.float32))
        centroids = np.stack(centroids).astype(np.float32)
        centroid_conf = np.array(centroid_conf, dtype=np.float32)
        centroid_size = np.array(centroid_size, dtype=np.int64)
        centroid_phys = np.stack(centroid_phys).astype(np.float32) if physics_mat is not None else None
        return centroids, centroid_conf, centroid_size, centroid_phys

    def replacePrototype(self, memory_bank, speaker_id):
        """Runs the full compression pipeline for one speaker and writes
        the compressed prototypes back into the memory bank. Returns True
        if compression actually ran (i.e. the speaker was over the trigger
        count)."""
        rows = memory_bank.getSpeakerRows(speaker_id)
        if len(rows) < self.trigger_count:
            return False

        mat = np.stack([r["embedding"] for r in rows])
        weights = np.array([r["age_weight"] for r in rows], dtype=np.float32)
        confidences = np.array([r["confidence"] for r in rows], dtype=np.float32)
        has_physics = all(r["physics_vector"] is not None for r in rows)
        physics_mat = np.stack([r["physics_vector"] for r in rows]) if has_physics else None

        keep_mask = self.removeOutliers(mat, weights)
        mat, weights, confidences = mat[keep_mask], weights[keep_mask], confidences[keep_mask]
        physics_mat = physics_mat[keep_mask] if physics_mat is not None else None

        labels, n_clusters = self.clusterEmbeddings(mat, weights, self.target_count)
        centroids, centroid_conf, centroid_size, centroid_phys = self.computeCentroids(
            mat, labels, confidences, weights, physics_mat, n_clusters
        )
        memory_bank.replaceSpeakerEntries(
            speaker_id, centroids, centroid_conf, centroid_size, physics_centroids=centroid_phys
        )
        return True


# --------------------------------------------------------------------------- #
# Quality gating
# --------------------------------------------------------------------------- #
class MemoryQualityController:
    """The gatekeeper that decides whether a new embedding is trustworthy
    enough to enter the Speaker Memory Bank, including a Physics
    Consistency Score check."""

    def __init__(self, physics_thresh=None):
        self.physics_thresh = physics_thresh or config.PHYSICS_SIM_THRESH

    def computeConfidence(self, probs, pred_class):
        return float(probs[pred_class])

    def computePhysicsScore(self, memory_bank, speaker_id, physics_vec):
        """Cosine similarity (rescaled to [0, 1]) between the incoming
        sample's physics-guided features (jitter/shimmer/HNR/formant
        stability) and the speaker's historical, age-weighted physics
        profile. Returns 1.0 (pass) when there is no physics history yet
        to compare against."""
        mean_physics = memory_bank.getSpeakerPhysicsMean(speaker_id)
        if mean_physics is None or not np.any(mean_physics) or not np.any(physics_vec):
            return 1.0
        a = physics_vec / (np.linalg.norm(physics_vec) + 1e-9)
        b = mean_physics / (np.linalg.norm(mean_physics) + 1e-9)
        cos = float(np.dot(a, b))
        return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))

    def computeSimilarity(self, sims):
        return float(sims.max()) if len(sims) else 0.0

    def shouldStore(self, confidence, physics_score, similarity, conf_thresh, sim_thresh, is_first_for_speaker):
        if confidence < conf_thresh:
            return False
        if is_first_for_speaker:
            return True
        return (physics_score >= self.physics_thresh) and (similarity >= sim_thresh)


class ConfidenceGate:
    """Confidence Verification Gate. Update memory ONLY IF: Prediction
    Confidence > threshold AND Physics Score > threshold AND Similarity >
    threshold (bootstrap samples are exempt from the physics/similarity
    checks since there is nothing yet to compare against)."""

    def __init__(self, quality_controller, bootstrap_thresh=None,
                 update_thresh=None, sim_thresh=None):
        self.qc = quality_controller
        self.bootstrap_thresh = bootstrap_thresh or config.CONF_THRESH_BOOTSTRAP
        self.update_thresh = update_thresh or config.CONF_THRESH_UPDATE
        self.sim_thresh = sim_thresh or config.SIM_THRESH_UPDATE

    def validatePrediction(self, pred_class, confidence, is_first_for_speaker):
        if pred_class != 0:
            return False
        thresh = self.bootstrap_thresh if is_first_for_speaker else self.update_thresh
        return confidence >= thresh

    def shouldUpdateMemory(self, pred_class, confidence, similarity, physics_score, is_first_for_speaker):
        if not self.validatePrediction(pred_class, confidence, is_first_for_speaker):
            return False
        conf_thresh = self.bootstrap_thresh if is_first_for_speaker else self.update_thresh
        return self.qc.shouldStore(
            confidence, physics_score, similarity, conf_thresh, self.sim_thresh, is_first_for_speaker
        )


# --------------------------------------------------------------------------- #
# Update orchestration
# --------------------------------------------------------------------------- #
class MemoryUpdateController:
    """Manages state transitions within the memory bank. Instead of naive
    appending, it orchestrates Insert, Prototype/Merge (via
    PrototypeMemoryManager), Memory Aging (via MemoryAgingManager), and
    index rebuilding as one coordinated update."""

    def __init__(self, memory_bank, aging_manager, prototype_manager):
        self.memory_bank = memory_bank
        self.aging_manager = aging_manager
        self.prototype_manager = prototype_manager

    def validateMemory(self, quality_controller, confidence_gate, pred_class, confidence,
                        similarity, physics_score, is_first):
        return confidence_gate.shouldUpdateMemory(pred_class, confidence, similarity, physics_score, is_first)

    def insertEmbedding(self, speaker_id, embedding, physics_vec, confidence):
        self.memory_bank.addEmbedding(speaker_id, embedding, confidence, physics_vector=physics_vec)

    def recalculateWeights(self, speaker_id):
        return self.aging_manager.decayOldEmbeddings(self.memory_bank, speaker_id)

    def updatePrototype(self, speaker_id):
        return self.prototype_manager.replacePrototype(self.memory_bank, speaker_id)

    def rebuildIndex(self):
        self.memory_bank.rebuildIndex()

    def processUpdate(self, quality_controller, confidence_gate, speaker_id, embedding, physics_vec,
                       pred_class, confidence, similarity, physics_score, is_first):
        """Full pipeline for one candidate update. Returns
        (was_written, was_compressed).

        NOTE: intentionally does NOT call self.rebuildIndex() here -- that
        used to run on every accepted write (on top of
        MemoryBank.addEmbedding() already doing its own full rebuild), i.e.
        2-3 full-table FAISS rebuilds per write for a global index that no
        retrieval path here ever queries. Call update_controller
        .rebuildIndex() explicitly at the end of a replay if you need the
        global index refreshed for some other consumer."""
        if not self.validateMemory(quality_controller, confidence_gate, pred_class, confidence,
                                    similarity, physics_score, is_first):
            self.memory_bank.removeOutdatedEntries(speaker_id)  # hard safety-net cap still applies
            return False, False
        self.insertEmbedding(speaker_id, embedding, physics_vec, confidence)
        self.recalculateWeights(speaker_id)
        compressed = self.updatePrototype(speaker_id)
        return True, compressed


def build_memory_system(memory_bank=None):
    """Convenience constructor wiring up the full memory subsystem, used by
    train.py / inference.py."""
    memory_bank = memory_bank or MemoryBank()
    aging_manager = MemoryAgingManager()
    prototype_manager = PrototypeMemoryManager()
    adaptive_retriever = AdaptiveMemoryRetriever(memory_bank)
    quality_controller = MemoryQualityController()
    confidence_gate = ConfidenceGate(quality_controller)
    update_controller = MemoryUpdateController(memory_bank, aging_manager, prototype_manager)
    speaker_context_builder = SpeakerContextBuilder(memory_bank)
    return {
        "memory_bank": memory_bank,
        "aging_manager": aging_manager,
        "prototype_manager": prototype_manager,
        "adaptive_retriever": adaptive_retriever,
        "quality_controller": quality_controller,
        "confidence_gate": confidence_gate,
        "update_controller": update_controller,
        "speaker_context_builder": speaker_context_builder,
    }
