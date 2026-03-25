"""
Problem 1 - Task 2: Word2Vec Model Training (From Scratch)
===========================================================
Implements two Word2Vec variants using only NumPy:
  1. CBOW  - Continuous Bag of Words
  2. Skip-gram with Negative Sampling (SGNS)

Each model is trained with multiple hyperparameter configurations:
  - Embedding dimensions: [50, 100]
  - Context window sizes: [2, 4]
  - Negative samples: [5, 10]  (Skip-gram only)

Models are saved as numpy .npz files for downstream use.
"""

import numpy as np
import json
import os
import time
import collections
import random


# ════════════════════════════════════════════════════════════════
# SECTION 1: VOCABULARY & DATA UTILITIES
# ════════════════════════════════════════════════════════════════

def load_corpus(filepath):
    """
    Load the cleaned corpus.
    Each line is a sentence; tokens are space-separated.
    Returns list of sentences (each is a list of strings).
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def build_vocabulary(sentences, min_count=2):
    """
    Build vocabulary from corpus.
    - Only include words appearing >= min_count times.
    - Returns: word2idx, idx2word, word_counts dicts.
    """
    counter = collections.Counter()
    for sent in sentences:
        counter.update(sent)

    # Filter by minimum frequency
    vocab_words = [w for w, c in counter.items() if c >= min_count]
    vocab_words = sorted(vocab_words)  # Deterministic ordering

    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    word_counts = {w: counter[w] for w in vocab_words}

    return word2idx, idx2word, word_counts


def encode_corpus(sentences, word2idx):
    """
    Convert corpus sentences from strings to integer indices.
    Words not in vocabulary are skipped.
    """
    encoded = []
    for sent in sentences:
        ids = [word2idx[w] for w in sent if w in word2idx]
        if len(ids) >= 2:
            encoded.append(ids)
    return encoded


def build_negative_sampling_table(word_counts, word2idx, table_size=1_000_000):
    """
    Build a negative sampling frequency table.
    Words are sampled proportional to freq^(3/4) as per original Word2Vec paper.
    """
    vocab_size = len(word2idx)
    # Compute adjusted frequencies
    freqs = np.array([word_counts[w] ** 0.75 for w in sorted(word2idx, key=word2idx.get)])
    freqs /= freqs.sum()
    # Build table: each position maps to a word index
    table = np.zeros(table_size, dtype=np.int32)
    idx, cumulative = 0, 0.0
    for i in range(table_size):
        cumulative += freqs[idx] * table_size
        table[i] = idx
        if i > cumulative and idx < vocab_size - 1:
            idx += 1
    return table


def get_negative_samples(neg_table, target_idx, n_samples):
    """
    Sample n_samples negative word indices from the table,
    ensuring the target word itself is not sampled.
    """
    samples = []
    while len(samples) < n_samples:
        s = neg_table[np.random.randint(0, len(neg_table))]
        if s != target_idx:
            samples.append(s)
    return samples


# ════════════════════════════════════════════════════════════════
# SECTION 2: SIGMOID UTILITY
# ════════════════════════════════════════════════════════════════

def sigmoid(x):
    """Numerically stable sigmoid function."""
    # Clip to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# ════════════════════════════════════════════════════════════════
# SECTION 3: CBOW MODEL
# ════════════════════════════════════════════════════════════════

class CBOWModel:
    """
    Continuous Bag of Words (CBOW) Word2Vec Model.

    Architecture:
      - Input: context word one-hot vectors (averaged)
      - Hidden: embedding lookup (W_in matrix, shape: [vocab_size, embed_dim])
      - Output: softmax over vocabulary (W_out matrix, shape: [embed_dim, vocab_size])

    Training: Standard cross-entropy with hierarchical softmax approximated
              here by direct softmax over full vocabulary for simplicity.
              (Full softmax is exact but slower; suitable for smaller vocab.)

    For each training sample:
      - h = mean(W_in[context_words])  # hidden representation
      - scores = h @ W_out              # [vocab_size]
      - loss = cross_entropy(scores, target)
      - Gradient update via SGD
    """

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Xavier initialization for stable training
        scale = np.sqrt(2.0 / (vocab_size + embed_dim))
        self.W_in  = np.random.uniform(-scale, scale, (vocab_size, embed_dim))  # Input embeddings
        self.W_out = np.random.uniform(-scale, scale, (embed_dim, vocab_size))  # Output matrix

    def forward(self, context_ids):
        """
        Forward pass: average context embeddings -> output scores.
        Returns hidden vector h and raw output scores.
        """
        # h is the mean of context word embeddings
        h = self.W_in[context_ids].mean(axis=0)       # shape: [embed_dim]
        scores = h @ self.W_out                         # shape: [vocab_size]
        return h, scores

    def softmax(self, scores):
        """Numerically stable softmax."""
        scores = scores - scores.max()
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum()

    def train_step(self, context_ids, target_id, lr):
        """
        Single SGD training step.
        Returns cross-entropy loss for monitoring.
        """
        h, scores = self.forward(context_ids)
        probs = self.softmax(scores)

        # Cross-entropy loss: -log P(target | context)
        loss = -np.log(probs[target_id] + 1e-10)

        # Gradient w.r.t. output scores (softmax + CE gradient)
        d_scores = probs.copy()
        d_scores[target_id] -= 1.0                      # shape: [vocab_size]

        # Gradient w.r.t. W_out: h outer d_scores
        d_W_out = np.outer(h, d_scores)                 # shape: [embed_dim, vocab_size]
        # Gradient w.r.t. hidden h
        d_h = self.W_out @ d_scores                     # shape: [embed_dim]

        # Update W_out
        self.W_out -= lr * d_W_out

        # Update W_in for each context word (distribute gradient equally)
        d_W_in = d_h / len(context_ids)
        for cid in context_ids:
            self.W_in[cid] -= lr * d_W_in

        return loss

    def get_embeddings(self):
        """Return the learned input embedding matrix."""
        return self.W_in


# ════════════════════════════════════════════════════════════════
# SECTION 4: SKIP-GRAM WITH NEGATIVE SAMPLING
# ════════════════════════════════════════════════════════════════

class SkipGramNSModel:
    """
    Skip-Gram with Negative Sampling (SGNS) Word2Vec Model.

    Architecture:
      - For each (center, context) pair, maximize similarity
      - For each (center, negative) pair, minimize similarity
      - Uses two embedding matrices: W_in (center) and W_out (context)

    Objective (per training pair):
      L = log σ(v_c · v_o) + Σ_k E[log σ(-v_c · v_neg_k)]

    where:
      v_c  = embedding of center word (from W_in)
      v_o  = embedding of context word (from W_out)
      v_neg = embedding of negative samples (from W_out)

    Updates are via SGD on this objective.
    """

    def __init__(self, vocab_size, embed_dim, n_negative):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_negative = n_negative
        # Xavier initialization
        scale = np.sqrt(2.0 / (vocab_size + embed_dim))
        self.W_in  = np.random.uniform(-scale, scale, (vocab_size, embed_dim))  # Center words
        self.W_out = np.random.uniform(-scale, scale, (vocab_size, embed_dim))  # Context/output words

    def train_step(self, center_id, context_id, neg_ids, lr):
        """
        One SGNS update step for a single (center, context) pair.

        center_id  : int - index of center word
        context_id : int - index of positive context word
        neg_ids    : list of int - indices of negative sample words
        lr         : float - learning rate
        Returns: scalar loss value
        """
        # Center word embedding
        v_c = self.W_in[center_id]                      # shape: [embed_dim]

        # ── Positive pair ──────────────────────────────────────
        v_o = self.W_out[context_id]                    # shape: [embed_dim]
        dot_pos = np.dot(v_c, v_o)
        sig_pos = sigmoid(dot_pos)
        loss = -np.log(sig_pos + 1e-10)

        # Gradients for positive pair
        # d_L/d_v_c += (sig_pos - 1) * v_o
        # d_L/d_v_o += (sig_pos - 1) * v_c
        d_vc = (sig_pos - 1.0) * v_o                   # shape: [embed_dim]
        d_vo = (sig_pos - 1.0) * v_c                   # shape: [embed_dim]
        self.W_out[context_id] -= lr * d_vo

        # ── Negative pairs ─────────────────────────────────────
        for neg_id in neg_ids:
            v_neg = self.W_out[neg_id]                  # shape: [embed_dim]
            dot_neg = np.dot(v_c, v_neg)
            sig_neg = sigmoid(dot_neg)
            loss += -np.log(1.0 - sig_neg + 1e-10)

            # Gradients for negative pair
            # d_L/d_v_c += sig_neg * v_neg
            # d_L/d_v_neg += sig_neg * v_c
            d_vc += sig_neg * v_neg
            d_vneg = sig_neg * v_c
            self.W_out[neg_id] -= lr * d_vneg

        # Update center word embedding
        self.W_in[center_id] -= lr * d_vc

        return loss

    def get_embeddings(self):
        """Return the learned center word embedding matrix (W_in)."""
        return self.W_in


# ════════════════════════════════════════════════════════════════
# SECTION 5: TRAINING FUNCTIONS
# ════════════════════════════════════════════════════════════════

def train_cbow(sentences_encoded, vocab_size, embed_dim, window_size,
               n_epochs=5, lr=0.025, lr_min=0.0001):
    """
    Train the CBOW model on encoded corpus.
    Uses a linearly decaying learning rate schedule.

    Parameters:
      sentences_encoded : list of list of int
      vocab_size        : int
      embed_dim         : int
      window_size       : int (half context window)
      n_epochs          : int
      lr                : float (initial learning rate)
      lr_min            : float (minimum learning rate)
    """
    model = CBOWModel(vocab_size, embed_dim)
    # Total training pairs for LR decay calculation
    total_pairs = sum(
        max(0, len(s) - 2 * window_size) * 2 * window_size
        for s in sentences_encoded
    ) * n_epochs
    step = 0

    print(f"\n  Training CBOW | embed={embed_dim} | window={window_size}")
    print(f"  vocab_size={vocab_size} | epochs={n_epochs}")

    final_loss = None
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_pairs = 0
        # Shuffle sentences each epoch
        random.shuffle(sentences_encoded)

        for sent in sentences_encoded:
            # Slide context window over each target position
            for i in range(len(sent)):
                # Context: words within window (excluding target)
                start = max(0, i - window_size)
                end   = min(len(sent), i + window_size + 1)
                context_ids = [sent[j] for j in range(start, end) if j != i]

                if not context_ids:
                    continue

                target_id = sent[i]

                # Linearly decay learning rate
                cur_lr = max(lr_min, lr * (1.0 - step / (total_pairs + 1)))
                loss = model.train_step(context_ids, target_id, cur_lr)
                epoch_loss += loss
                n_pairs += 1
                step += 1

        avg_loss = epoch_loss / max(1, n_pairs)
        final_loss = avg_loss
        print(f"    Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.5f}")

    return model, final_loss


def train_skipgram(sentences_encoded, vocab_size, embed_dim, window_size, n_negative,
                   word_counts, word2idx, n_epochs=5, lr=0.025, lr_min=0.0001):
    """
    Train Skip-gram with Negative Sampling — vectorized for speed.

    Key optimization: instead of looping over each (center, context, negative)
    triple one by one, we:
      1. Collect ALL (center, context) pairs for the entire epoch into arrays
      2. Sample ALL negative indices at once using numpy (no Python loop)
      3. Compute gradients in batch using numpy matrix ops
    This avoids slow Python-level loops over individual word pairs.
    """
    model = SkipGramNSModel(vocab_size, embed_dim, n_negative)
    neg_table = build_negative_sampling_table(word_counts, word2idx)

    print(f"\n  Training Skip-gram | embed={embed_dim} | window={window_size} | neg={n_negative}")
    print(f"  vocab_size={vocab_size} | epochs={n_epochs}")

    final_loss = None
    for epoch in range(n_epochs):
        random.shuffle(sentences_encoded)

        # ── Step 1: Collect all (center, context) pairs for this epoch ──
        centers  = []
        contexts = []
        for sent in sentences_encoded:
            for i, cid in enumerate(sent):
                start = max(0, i - window_size)
                end   = min(len(sent), i + window_size + 1)
                for j in range(start, end):
                    if j != i:
                        centers.append(cid)
                        contexts.append(sent[j])

        centers  = np.array(centers,  dtype=np.int32)
        contexts = np.array(contexts, dtype=np.int32)
        n_pairs  = len(centers)

        # ── Step 2: Sample ALL negatives at once (vectorized) ───────────
        # Draw random indices into neg_table for all pairs × n_negative
        rand_idx  = np.random.randint(0, len(neg_table), size=(n_pairs, n_negative))
        neg_array = neg_table[rand_idx]                 # [n_pairs, n_negative]

        # ── Step 3: Shuffle pair order for SGD randomness ───────────────
        perm = np.random.permutation(n_pairs)
        centers  = centers[perm]
        contexts = contexts[perm]
        neg_array = neg_array[perm]

        epoch_loss = 0.0
        cur_lr = lr

        # ── Step 4: Mini-batch SGD (batch_size pairs per update) ─────────
        # Batching avoids Python loop overhead while staying memory-friendly
        BATCH = 256
        n_batches = (n_pairs + BATCH - 1) // BATCH

        for b in range(n_batches):
            # Linearly decay LR across the epoch
            cur_lr = max(lr_min, lr * (1.0 - (epoch * n_batches + b) /
                                        (n_epochs * n_batches + 1)))

            sl = slice(b * BATCH, (b + 1) * BATCH)
            c_ids  = centers[sl]                        # [B]
            o_ids  = contexts[sl]                       # [B]
            n_ids  = neg_array[sl]                      # [B, K]
            B = len(c_ids)

            # Retrieve embeddings
            vc = model.W_in[c_ids]                      # [B, E]
            vo = model.W_out[o_ids]                     # [B, E]
            vn = model.W_out[n_ids]                     # [B, K, E]

            # ── Positive pair gradient ───────────────────────────────
            dot_pos  = np.einsum('be,be->b', vc, vo)   # [B]
            sig_pos  = sigmoid(dot_pos)                 # [B]
            epoch_loss += -np.log(sig_pos + 1e-10).sum()

            err_pos  = sig_pos - 1.0                    # [B]
            d_vc_pos = err_pos[:, None] * vo            # [B, E]
            d_vo     = err_pos[:, None] * vc            # [B, E]

            # ── Negative pairs gradient ──────────────────────────────
            # vn: [B, K, E],  vc: [B, 1, E]
            dot_neg  = np.einsum('be,bke->bk', vc, vn) # [B, K]
            sig_neg  = sigmoid(dot_neg)                 # [B, K]
            epoch_loss += -np.log(1.0 - sig_neg + 1e-10).sum()

            # d_vc from negatives: sum over K  →  [B, E]
            d_vc_neg = np.einsum('bk,bke->be', sig_neg, vn)
            d_vc     = d_vc_pos + d_vc_neg              # [B, E]

            # d_vn per negative sample: [B, K, E]
            d_vn     = sig_neg[:, :, None] * vc[:, None, :]

            # ── SGD updates using np.add.at (handles repeated indices) ─
            np.add.at(model.W_in,  c_ids,  -cur_lr * d_vc)
            np.add.at(model.W_out, o_ids,  -cur_lr * d_vo)
            # Flatten [B,K] negative indices for scatter update
            np.add.at(model.W_out, n_ids.ravel(),
                      -cur_lr * d_vn.reshape(-1, embed_dim))

        avg_loss = epoch_loss / max(1, n_pairs)
        final_loss = avg_loss
        print(f"    Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.5f}")

    return model, final_loss


# ════════════════════════════════════════════════════════════════
# SECTION 6: SAVE / LOAD UTILITIES
# ════════════════════════════════════════════════════════════════

def save_model(embeddings, word2idx, idx2word, config, filepath):
    """Save model embeddings and vocabulary to a .npz file."""
    np.savez(filepath,
             embeddings=embeddings,
             word2idx=json.dumps(word2idx),
             idx2word=json.dumps(idx2word),
             config=json.dumps(config))
    print(f"  [SAVED] Model saved to {filepath}")


def load_model(filepath):
    """Load a saved Word2Vec model from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    embeddings = data["embeddings"]
    word2idx   = json.loads(str(data["word2idx"]))
    idx2word   = json.loads(str(data["idx2word"]))
    config     = json.loads(str(data["config"]))
    return embeddings, word2idx, idx2word, config


# ════════════════════════════════════════════════════════════════
# SECTION 7: MAIN - HYPERPARAMETER EXPERIMENTS
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Load corpus ────────────────────────────────────────────
    print("=" * 65)
    print("  TASK 2: WORD2VEC MODEL TRAINING")
    print("=" * 65)

    sentences = load_corpus("corpus_cleaned.txt")
    word2idx, idx2word, word_counts = build_vocabulary(sentences, min_count=1)
    sentences_encoded = encode_corpus(sentences, word2idx)
    vocab_size = len(word2idx)
    print(f"\n  Loaded {len(sentences)} sentences | Vocab size: {vocab_size}")

    # ── Hyperparameter grid ────────────────────────────────────
    embed_dims    = [50, 100]
    window_sizes  = [2, 4]
    neg_samples   = [5, 10]
    N_EPOCHS      = 5
    model_dir     = "models"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results_summary = []

    # ── Train CBOW models ──────────────────────────────────────
    print("\n" + "═" * 65)
    print("  CBOW MODELS")
    print("═" * 65)

    for embed_dim in embed_dims:
        for window_size in window_sizes:
            config = {
                "model": "CBOW",
                "embed_dim": embed_dim,
                "window_size": window_size,
                "vocab_size": vocab_size,
                "n_epochs": N_EPOCHS
            }
            t0 = time.time()
            model, final_loss = train_cbow(
                sentences_encoded, vocab_size,
                embed_dim, window_size,
                n_epochs=N_EPOCHS
            )
            elapsed = time.time() - t0
            fname = os.path.join(model_dir, f"cbow_d{embed_dim}_w{window_size}")
            save_model(model.get_embeddings(), word2idx, idx2word, config, fname)
            results_summary.append({
                **config,
                "final_loss": round(float(final_loss), 4),
                "train_time_sec": round(elapsed, 2)
            })
            print(f"  Training time: {elapsed:.1f}s")

    # ── Train Skip-gram models ─────────────────────────────────
    print("\n" + "═" * 65)
    print("  SKIP-GRAM WITH NEGATIVE SAMPLING MODELS")
    print("═" * 65)

    for embed_dim in embed_dims:
        for window_size in window_sizes:
            for n_neg in neg_samples:
                config = {
                    "model": "SkipGram-NS",
                    "embed_dim": embed_dim,
                    "window_size": window_size,
                    "n_negative": n_neg,
                    "vocab_size": vocab_size,
                    "n_epochs": N_EPOCHS
                }
                t0 = time.time()
                model, final_loss = train_skipgram(
                    sentences_encoded, vocab_size,
                    embed_dim, window_size, n_neg,
                    word_counts, word2idx,
                    n_epochs=N_EPOCHS
                )
                elapsed = time.time() - t0
                fname = os.path.join(model_dir, f"skipgram_d{embed_dim}_w{window_size}_neg{n_neg}")
                save_model(model.get_embeddings(), word2idx, idx2word, config, fname)
                results_summary.append({
                    **config,
                    "final_loss": round(float(final_loss), 4),
                    "train_time_sec": round(elapsed, 2)
                })
                print(f"  Training time: {elapsed:.1f}s")

    # ── Print results table ────────────────────────────────────
    print("\n" + "═" * 65)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("═" * 65)
    print(f"  {'Model':<18} {'EmbDim':>6} {'Window':>6} {'NegSamp':>7} {'Loss':>10} {'Time(s)':>8}")
    print("  " + "-" * 62)
    for r in results_summary:
        neg = r.get("n_negative", "-")
        print(
            f"  {r['model']:<18} {r['embed_dim']:>6} {r['window_size']:>6} "
            f"{str(neg):>7} {r['final_loss']:>10.4f} {r['train_time_sec']:>8}"
        )

    # Save summary to JSON
    with open("results/training_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[DONE] Task 2 completed. {len(results_summary)} models trained.")
    print("[DONE] Results saved to results/training_results.json\n")





# """
# Problem 1 - Task 2: Word2Vec Model Training (From Scratch)
# ===========================================================
# Implements two Word2Vec variants using only NumPy:
#   1. CBOW  - Continuous Bag of Words
#   2. Skip-gram with Negative Sampling (SGNS)

# Each model is trained with multiple hyperparameter configurations:
#   - Embedding dimensions: [50, 100]
#   - Context window sizes: [2, 4]
#   - Negative samples: [5, 10]  (Skip-gram only)

# Models are saved as numpy .npz files for downstream use.
# """

# import numpy as np
# import json
# import os
# import time
# import collections
# import random


# # ════════════════════════════════════════════════════════════════
# # SECTION 1: VOCABULARY & DATA UTILITIES
# # ════════════════════════════════════════════════════════════════

# def load_corpus(filepath):
#     """
#     Load the cleaned corpus.
#     Each line is a sentence; tokens are space-separated.
#     Returns list of sentences (each is a list of strings).
#     """
#     sentences = []
#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             tokens = line.strip().split()
#             if tokens:
#                 sentences.append(tokens)
#     return sentences


# def build_vocabulary(sentences, min_count=2):
#     """
#     Build vocabulary from corpus.
#     - Only include words appearing >= min_count times.
#     - Returns: word2idx, idx2word, word_counts dicts.
#     """
#     counter = collections.Counter()
#     for sent in sentences:
#         counter.update(sent)

#     # Filter by minimum frequency
#     vocab_words = [w for w, c in counter.items() if c >= min_count]
#     vocab_words = sorted(vocab_words)  # Deterministic ordering

#     word2idx = {w: i for i, w in enumerate(vocab_words)}
#     idx2word = {i: w for w, i in word2idx.items()}
#     word_counts = {w: counter[w] for w in vocab_words}

#     return word2idx, idx2word, word_counts


# def encode_corpus(sentences, word2idx):
#     """
#     Convert corpus sentences from strings to integer indices.
#     Words not in vocabulary are skipped.
#     """
#     encoded = []
#     for sent in sentences:
#         ids = [word2idx[w] for w in sent if w in word2idx]
#         if len(ids) >= 2:
#             encoded.append(ids)
#     return encoded


# def build_negative_sampling_table(word_counts, word2idx, table_size=1_000_000):
#     """
#     Build a negative sampling frequency table.
#     Words are sampled proportional to freq^(3/4) as per original Word2Vec paper.
#     """
#     vocab_size = len(word2idx)
#     # Compute adjusted frequencies
#     freqs = np.array([word_counts[w] ** 0.75 for w in sorted(word2idx, key=word2idx.get)])
#     freqs /= freqs.sum()
#     # Build table: each position maps to a word index
#     table = np.zeros(table_size, dtype=np.int32)
#     idx, cumulative = 0, 0.0
#     for i in range(table_size):
#         cumulative += freqs[idx] * table_size
#         table[i] = idx
#         if i > cumulative and idx < vocab_size - 1:
#             idx += 1
#     return table


# def get_negative_samples(neg_table, target_idx, n_samples):
#     """
#     Sample n_samples negative word indices from the table,
#     ensuring the target word itself is not sampled.
#     """
#     samples = []
#     while len(samples) < n_samples:
#         s = neg_table[np.random.randint(0, len(neg_table))]
#         if s != target_idx:
#             samples.append(s)
#     return samples


# # ════════════════════════════════════════════════════════════════
# # SECTION 2: SIGMOID UTILITY
# # ════════════════════════════════════════════════════════════════

# def sigmoid(x):
#     """Numerically stable sigmoid function."""
#     # Clip to prevent overflow in exp
#     x = np.clip(x, -500, 500)
#     return 1.0 / (1.0 + np.exp(-x))


# # ════════════════════════════════════════════════════════════════
# # SECTION 3: CBOW MODEL
# # ════════════════════════════════════════════════════════════════

# class CBOWModel:
#     """
#     Continuous Bag of Words (CBOW) Word2Vec Model.

#     Architecture:
#       - Input: context word one-hot vectors (averaged)
#       - Hidden: embedding lookup (W_in matrix, shape: [vocab_size, embed_dim])
#       - Output: softmax over vocabulary (W_out matrix, shape: [embed_dim, vocab_size])

#     Training: Standard cross-entropy with hierarchical softmax approximated
#               here by direct softmax over full vocabulary for simplicity.
#               (Full softmax is exact but slower; suitable for smaller vocab.)

#     For each training sample:
#       - h = mean(W_in[context_words])  # hidden representation
#       - scores = h @ W_out              # [vocab_size]
#       - loss = cross_entropy(scores, target)
#       - Gradient update via SGD
#     """

#     def __init__(self, vocab_size, embed_dim):
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         # Xavier initialization for stable training
#         scale = np.sqrt(2.0 / (vocab_size + embed_dim))
#         self.W_in  = np.random.uniform(-scale, scale, (vocab_size, embed_dim))  # Input embeddings
#         self.W_out = np.random.uniform(-scale, scale, (embed_dim, vocab_size))  # Output matrix

#     def forward(self, context_ids):
#         """
#         Forward pass: average context embeddings -> output scores.
#         Returns hidden vector h and raw output scores.
#         """
#         # h is the mean of context word embeddings
#         h = self.W_in[context_ids].mean(axis=0)       # shape: [embed_dim]
#         scores = h @ self.W_out                         # shape: [vocab_size]
#         return h, scores

#     def softmax(self, scores):
#         """Numerically stable softmax."""
#         scores = scores - scores.max()
#         exp_s = np.exp(scores)
#         return exp_s / exp_s.sum()

#     def train_step(self, context_ids, target_id, lr):
#         """
#         Single SGD training step.
#         Returns cross-entropy loss for monitoring.
#         """
#         h, scores = self.forward(context_ids)
#         probs = self.softmax(scores)

#         # Cross-entropy loss: -log P(target | context)
#         loss = -np.log(probs[target_id] + 1e-10)

#         # Gradient w.r.t. output scores (softmax + CE gradient)
#         d_scores = probs.copy()
#         d_scores[target_id] -= 1.0                      # shape: [vocab_size]

#         # Gradient w.r.t. W_out: h outer d_scores
#         d_W_out = np.outer(h, d_scores)                 # shape: [embed_dim, vocab_size]
#         # Gradient w.r.t. hidden h
#         d_h = self.W_out @ d_scores                     # shape: [embed_dim]

#         # Update W_out
#         self.W_out -= lr * d_W_out

#         # Update W_in for each context word (distribute gradient equally)
#         d_W_in = d_h / len(context_ids)
#         for cid in context_ids:
#             self.W_in[cid] -= lr * d_W_in

#         return loss

#     def get_embeddings(self):
#         """Return the learned input embedding matrix."""
#         return self.W_in


# # ════════════════════════════════════════════════════════════════
# # SECTION 4: SKIP-GRAM WITH NEGATIVE SAMPLING
# # ════════════════════════════════════════════════════════════════

# class SkipGramNSModel:
#     """
#     Skip-Gram with Negative Sampling (SGNS) Word2Vec Model.

#     Architecture:
#       - For each (center, context) pair, maximize similarity
#       - For each (center, negative) pair, minimize similarity
#       - Uses two embedding matrices: W_in (center) and W_out (context)

#     Objective (per training pair):
#       L = log σ(v_c · v_o) + Σ_k E[log σ(-v_c · v_neg_k)]

#     where:
#       v_c  = embedding of center word (from W_in)
#       v_o  = embedding of context word (from W_out)
#       v_neg = embedding of negative samples (from W_out)

#     Updates are via SGD on this objective.
#     """

#     def __init__(self, vocab_size, embed_dim, n_negative):
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.n_negative = n_negative
#         # Xavier initialization
#         scale = np.sqrt(2.0 / (vocab_size + embed_dim))
#         self.W_in  = np.random.uniform(-scale, scale, (vocab_size, embed_dim))  # Center words
#         self.W_out = np.random.uniform(-scale, scale, (vocab_size, embed_dim))  # Context/output words

#     def train_step(self, center_id, context_id, neg_ids, lr):
#         """
#         One SGNS update step for a single (center, context) pair.

#         center_id  : int - index of center word
#         context_id : int - index of positive context word
#         neg_ids    : list of int - indices of negative sample words
#         lr         : float - learning rate
#         Returns: scalar loss value
#         """
#         # Center word embedding
#         v_c = self.W_in[center_id]                      # shape: [embed_dim]

#         # ── Positive pair ──────────────────────────────────────
#         v_o = self.W_out[context_id]                    # shape: [embed_dim]
#         dot_pos = np.dot(v_c, v_o)
#         sig_pos = sigmoid(dot_pos)
#         loss = -np.log(sig_pos + 1e-10)

#         # Gradients for positive pair
#         # d_L/d_v_c += (sig_pos - 1) * v_o
#         # d_L/d_v_o += (sig_pos - 1) * v_c
#         d_vc = (sig_pos - 1.0) * v_o                   # shape: [embed_dim]
#         d_vo = (sig_pos - 1.0) * v_c                   # shape: [embed_dim]
#         self.W_out[context_id] -= lr * d_vo

#         # ── Negative pairs ─────────────────────────────────────
#         for neg_id in neg_ids:
#             v_neg = self.W_out[neg_id]                  # shape: [embed_dim]
#             dot_neg = np.dot(v_c, v_neg)
#             sig_neg = sigmoid(dot_neg)
#             loss += -np.log(1.0 - sig_neg + 1e-10)

#             # Gradients for negative pair
#             # d_L/d_v_c += sig_neg * v_neg
#             # d_L/d_v_neg += sig_neg * v_c
#             d_vc += sig_neg * v_neg
#             d_vneg = sig_neg * v_c
#             self.W_out[neg_id] -= lr * d_vneg

#         # Update center word embedding
#         self.W_in[center_id] -= lr * d_vc

#         return loss

#     def get_embeddings(self):
#         """Return the learned center word embedding matrix (W_in)."""
#         return self.W_in


# # ════════════════════════════════════════════════════════════════
# # SECTION 5: TRAINING FUNCTIONS
# # ════════════════════════════════════════════════════════════════

# def train_cbow(sentences_encoded, vocab_size, embed_dim, window_size,
#                n_epochs=5, lr=0.025, lr_min=0.0001):
#     """
#     Train the CBOW model on encoded corpus.
#     Uses a linearly decaying learning rate schedule.

#     Parameters:
#       sentences_encoded : list of list of int
#       vocab_size        : int
#       embed_dim         : int
#       window_size       : int (half context window)
#       n_epochs          : int
#       lr                : float (initial learning rate)
#       lr_min            : float (minimum learning rate)
#     """
#     model = CBOWModel(vocab_size, embed_dim)
#     # Total training pairs for LR decay calculation
#     total_pairs = sum(
#         max(0, len(s) - 2 * window_size) * 2 * window_size
#         for s in sentences_encoded
#     ) * n_epochs
#     step = 0

#     print(f"\n  Training CBOW | embed={embed_dim} | window={window_size}")
#     print(f"  vocab_size={vocab_size} | epochs={n_epochs}")

#     for epoch in range(n_epochs):
#         epoch_loss = 0.0
#         n_pairs = 0
#         # Shuffle sentences each epoch
#         random.shuffle(sentences_encoded)

#         for sent in sentences_encoded:
#             # Slide context window over each target position
#             for i in range(len(sent)):
#                 # Context: words within window (excluding target)
#                 start = max(0, i - window_size)
#                 end   = min(len(sent), i + window_size + 1)
#                 context_ids = [sent[j] for j in range(start, end) if j != i]

#                 if not context_ids:
#                     continue

#                 target_id = sent[i]

#                 # Linearly decay learning rate
#                 cur_lr = max(lr_min, lr * (1.0 - step / (total_pairs + 1)))
#                 loss = model.train_step(context_ids, target_id, cur_lr)
#                 epoch_loss += loss
#                 n_pairs += 1
#                 step += 1

#         avg_loss = epoch_loss / max(1, n_pairs)
#         print(f"    Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.5f}")

#     return model


# def train_skipgram(sentences_encoded, vocab_size, embed_dim, window_size, n_negative,
#                    word_counts, word2idx, n_epochs=5, lr=0.025, lr_min=0.0001):
#     """
#     Train the Skip-gram with Negative Sampling model.

#     Parameters:
#       n_negative  : int (number of negative samples per positive pair)
#       word_counts : dict (word -> count, for negative sampling table)
#       word2idx    : dict (for building negative sampling table)
#     """
#     model = SkipGramNSModel(vocab_size, embed_dim, n_negative)
#     neg_table = build_negative_sampling_table(word_counts, word2idx)

#     total_pairs = sum(
#         len(s) * 2 * window_size for s in sentences_encoded
#     ) * n_epochs
#     step = 0

#     print(f"\n  Training Skip-gram | embed={embed_dim} | window={window_size} | neg={n_negative}")
#     print(f"  vocab_size={vocab_size} | epochs={n_epochs}")

#     for epoch in range(n_epochs):
#         epoch_loss = 0.0
#         n_pairs = 0
#         random.shuffle(sentences_encoded)

#         for sent in sentences_encoded:
#             for i, center_id in enumerate(sent):
#                 # Context positions within window
#                 start = max(0, i - window_size)
#                 end   = min(len(sent), i + window_size + 1)
#                 context_ids = [sent[j] for j in range(start, end) if j != i]

#                 for context_id in context_ids:
#                     # Sample negative words for this (center, context) pair
#                     neg_ids = get_negative_samples(neg_table, center_id, n_negative)

#                     cur_lr = max(lr_min, lr * (1.0 - step / (total_pairs + 1)))
#                     loss = model.train_step(center_id, context_id, neg_ids, cur_lr)
#                     epoch_loss += loss
#                     n_pairs += 1
#                     step += 1

#         avg_loss = epoch_loss / max(1, n_pairs)
#         print(f"    Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.5f}")

#     return model


# # ════════════════════════════════════════════════════════════════
# # SECTION 6: SAVE / LOAD UTILITIES
# # ════════════════════════════════════════════════════════════════

# def save_model(embeddings, word2idx, idx2word, config, filepath):
#     """Save model embeddings and vocabulary to a .npz file."""
#     np.savez(filepath,
#              embeddings=embeddings,
#              word2idx=json.dumps(word2idx),
#              idx2word=json.dumps(idx2word),
#              config=json.dumps(config))
#     print(f"  [SAVED] Model saved to {filepath}")


# def load_model(filepath):
#     """Load a saved Word2Vec model from .npz file."""
#     data = np.load(filepath, allow_pickle=True)
#     embeddings = data["embeddings"]
#     word2idx   = json.loads(str(data["word2idx"]))
#     idx2word   = json.loads(str(data["idx2word"]))
#     config     = json.loads(str(data["config"]))
#     return embeddings, word2idx, idx2word, config


# # ════════════════════════════════════════════════════════════════
# # SECTION 7: MAIN - HYPERPARAMETER EXPERIMENTS
# # ════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     # ── Load corpus ────────────────────────────────────────────
#     print("=" * 65)
#     print("  TASK 2: WORD2VEC MODEL TRAINING")
#     print("=" * 65)

#     sentences = load_corpus("corpus_cleaned.txt")
#     word2idx, idx2word, word_counts = build_vocabulary(sentences, min_count=1)
#     sentences_encoded = encode_corpus(sentences, word2idx)
#     vocab_size = len(word2idx)
#     print(f"\n  Loaded {len(sentences)} sentences | Vocab size: {vocab_size}")

#     # ── Hyperparameter grid ────────────────────────────────────
#     embed_dims    = [50, 100]
#     window_sizes  = [2, 4]
#     neg_samples   = [5, 10]
#     N_EPOCHS      = 5

#     results_summary = []

#     # ── Train CBOW models ──────────────────────────────────────
#     print("\n" + "═" * 65)
#     print("  CBOW MODELS")
#     print("═" * 65)

#     for embed_dim in embed_dims:
#         for window_size in window_sizes:
#             config = {
#                 "model": "CBOW",
#                 "embed_dim": embed_dim,
#                 "window_size": window_size,
#                 "vocab_size": vocab_size,
#                 "n_epochs": N_EPOCHS
#             }
#             t0 = time.time()
#             model = train_cbow(
#                 sentences_encoded, vocab_size,
#                 embed_dim, window_size,
#                 n_epochs=N_EPOCHS
#             )
#             elapsed = time.time() - t0
#             fname = f"cbow_d{embed_dim}_w{window_size}"
#             save_model(model.get_embeddings(), word2idx, idx2word, config, fname)
#             results_summary.append({**config, "train_time_sec": round(elapsed, 2)})
#             print(f"  Training time: {elapsed:.1f}s")

#     # ── Train Skip-gram models ─────────────────────────────────
#     print("\n" + "═" * 65)
#     print("  SKIP-GRAM WITH NEGATIVE SAMPLING MODELS")
#     print("═" * 65)

#     for embed_dim in embed_dims:
#         for window_size in window_sizes:
#             for n_neg in neg_samples:
#                 config = {
#                     "model": "SkipGram-NS",
#                     "embed_dim": embed_dim,
#                     "window_size": window_size,
#                     "n_negative": n_neg,
#                     "vocab_size": vocab_size,
#                     "n_epochs": N_EPOCHS
#                 }
#                 t0 = time.time()
#                 model = train_skipgram(
#                     sentences_encoded, vocab_size,
#                     embed_dim, window_size, n_neg,
#                     word_counts, word2idx,
#                     n_epochs=N_EPOCHS
#                 )
#                 elapsed = time.time() - t0
#                 fname = f"skipgram_d{embed_dim}_w{window_size}_neg{n_neg}"
#                 save_model(model.get_embeddings(), word2idx, idx2word, config, fname)
#                 results_summary.append({**config, "train_time_sec": round(elapsed, 2)})
#                 print(f"  Training time: {elapsed:.1f}s")

#     # ── Print results table ────────────────────────────────────
#     print("\n" + "═" * 65)
#     print("  EXPERIMENT RESULTS SUMMARY")
#     print("═" * 65)
#     print(f"  {'Model':<18} {'EmbDim':>6} {'Window':>6} {'NegSamp':>7} {'Time(s)':>8}")
#     print("  " + "-" * 50)
#     for r in results_summary:
#         neg = r.get("n_negative", "-")
#         print(f"  {r['model']:<18} {r['embed_dim']:>6} {r['window_size']:>6} {str(neg):>7} {r['train_time_sec']:>8}")

#     # Save summary to JSON
#     with open("training_results.json", "w") as f:
#         json.dump(results_summary, f, indent=2)

#     print(f"\n[DONE] Task 2 completed. {len(results_summary)} models trained.\n")
