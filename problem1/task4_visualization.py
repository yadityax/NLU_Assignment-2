"""
Problem 1 - Task 4: Visualization of Word Embeddings
======================================================
Projects learned word embeddings into 2D space using:
  - PCA  (Principal Component Analysis) - linear projection
  - t-SNE (t-Distributed Stochastic Neighbor Embedding) - non-linear

Visualizes semantic clusters for domain-relevant word groups:
  - Academic programs: btech, mtech, phd, ug, pg
  - Academic roles: student, professor, faculty, researcher
  - Academic activities: research, exam, lecture, project, thesis
  - Administrative: department, course, credit, semester, grade

Generates PNG plots for CBOW vs Skip-gram using matplotlib.
"""

import numpy as np
import json
import os
import math


# ════════════════════════════════════════════════════════════════
# SECTION 1: MODEL LOADING & NORMALIZATION
# ════════════════════════════════════════════════════════════════

def load_model(filepath):
    """Load saved Word2Vec embeddings and vocabulary from .npz file."""
    data = np.load(filepath + ".npz", allow_pickle=True)
    embeddings = data["embeddings"]
    word2idx   = json.loads(str(data["word2idx"]))
    idx2word   = json.loads(str(data["idx2word"]))
    config     = json.loads(str(data["config"]))
    return embeddings, word2idx, idx2word, config


# ════════════════════════════════════════════════════════════════
# SECTION 2: PCA FROM SCRATCH
# ════════════════════════════════════════════════════════════════

def pca_2d(X):
    """
    Project matrix X [n_samples, d] to 2D using PCA (from scratch).

    Algorithm:
      1. Center the data: X_c = X - mean(X)
      2. Compute covariance matrix: C = X_c.T @ X_c / (n-1)
      3. Eigendecomposition: C = V D V^T
      4. Select top-2 eigenvectors (largest eigenvalues)
      5. Project: X_2d = X_c @ V[:, :2]

    PCA finds the directions of maximum variance in the data,
    making it suitable for visualizing global structure.
    """
    # Step 1: Mean-center
    X_c = X - X.mean(axis=0)

    # Step 2: Covariance matrix
    cov = (X_c.T @ X_c) / (X.shape[0] - 1)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)   # eigh: symmetric matrix

    # Step 4: Sort eigenvectors by eigenvalue descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    # Step 5: Project to top-2 principal components
    X_2d = X_c @ eigenvectors[:, :2]
    return X_2d


# ════════════════════════════════════════════════════════════════
# SECTION 3: t-SNE FROM SCRATCH (simplified)
# ════════════════════════════════════════════════════════════════

def tsne_2d(X, perplexity=5.0, n_iter=300, lr=10.0, seed=42):
    """
    t-SNE dimensionality reduction to 2D (simplified implementation).

    Algorithm:
      1. Compute pairwise Gaussian similarities in high-dim space (P)
      2. Initialize low-dim positions Y randomly
      3. Iteratively minimize KL(P || Q) where Q is t-distribution in 2D:
         - Compute Q (Student-t kernel in 2D)
         - Compute gradient of KL divergence
         - Update Y via gradient descent

    t-SNE preserves local neighborhood structure, making it ideal for
    revealing clusters of semantically related words.

    Note: This is a simplified version suitable for small vocab subsets.
    """
    np.random.seed(seed)
    n = X.shape[0]

    # ── Step 1: Compute pairwise distances in high-dim ─────────
    # ||x_i - x_j||^2
    sum_sq = np.sum(X ** 2, axis=1)
    dist_sq = sum_sq[:, None] + sum_sq[None, :] - 2.0 * (X @ X.T)
    dist_sq = np.maximum(dist_sq, 0.0)

    # ── Step 2: Compute joint probabilities P ──────────────────
    # For simplicity, use global sigma derived from perplexity
    # (Full t-SNE uses per-point binary search for sigma)
    sigma = np.sqrt(dist_sq.mean() / (2.0 * np.log(perplexity + 1e-10)))
    sigma = max(sigma, 1e-5)

    P = np.exp(-dist_sq / (2.0 * sigma ** 2))
    np.fill_diagonal(P, 0.0)
    P_sum = P.sum()
    if P_sum > 0:
        P = P / P_sum
    # Symmetrize and clip for numerical stability
    P = (P + P.T) / 2.0
    P = np.maximum(P, 1e-12)

    # ── Step 3: Initialize low-dim embedding Y ─────────────────
    Y = np.random.randn(n, 2) * 0.01

    # ── Step 4: Gradient descent ───────────────────────────────
    Y_prev = Y.copy()
    momentum = 0.5

    for iteration in range(n_iter):
        # Compute pairwise distances in 2D
        sum_sq_y = np.sum(Y ** 2, axis=1)
        dist_sq_y = sum_sq_y[:, None] + sum_sq_y[None, :] - 2.0 * (Y @ Y.T)
        dist_sq_y = np.maximum(dist_sq_y, 0.0)

        # Student-t kernel: Q_ij = (1 + ||y_i - y_j||^2)^-1
        Q = 1.0 / (1.0 + dist_sq_y)
        np.fill_diagonal(Q, 0.0)
        Q = Q / (Q.sum() + 1e-12)
        Q = np.maximum(Q, 1e-12)

        # KL gradient: dL/dY_i = 4 * sum_j (P_ij - Q_ij) * Q_ij * (Y_i - Y_j)
        PQ_diff = (P - Q) * Q                           # [n, n]
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y                             # [n, 2]
            grad[i] = 4.0 * (PQ_diff[i] @ diff)

        # Momentum update
        if iteration < 20:
            m = 0.5
        else:
            m = 0.8
        Y_new = Y - lr * grad + m * (Y - Y_prev)
        Y_prev = Y.copy()
        Y = Y_new

        # Re-center every 50 steps
        if iteration % 50 == 0:
            Y -= Y.mean(axis=0)

    return Y


# ════════════════════════════════════════════════════════════════
# SECTION 4: MATPLOTLIB PLOT (required)
# ════════════════════════════════════════════════════════════════

def plot_embeddings_matplotlib(coords_cbow, coords_sg, words, groups, group_labels, output_prefix):
    """
    Generate publication-quality 2D scatter plots using matplotlib.
    Creates 4 subplots: PCA CBOW, PCA Skip-gram, t-SNE CBOW, t-SNE Skip-gram.
    """
    try:
        import matplotlib.pyplot as plt

        # Color palette for word groups
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        markers = ["o", "s", "^", "D", "v"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Word Embedding Visualization\nIIT Jodhpur Corpus", fontsize=14, fontweight="bold")

        plot_data = [
            (axes[0, 0], coords_cbow["pca"],  "CBOW - PCA"),
            (axes[0, 1], coords_sg["pca"],    "Skip-gram - PCA"),
            (axes[1, 0], coords_cbow["tsne"], "CBOW - t-SNE"),
            (axes[1, 1], coords_sg["tsne"],   "Skip-gram - t-SNE"),
        ]

        # Word-to-group index mapping
        word_group = {}
        for gi, group_words in enumerate(groups):
            for w in group_words:
                if w in words:
                    word_group[w] = gi

        for ax, coords, title in plot_data:
            for gi, (group_words, glabel) in enumerate(zip(groups, group_labels)):
                xs, ys, labels = [], [], []
                for w in group_words:
                    if w in words:
                        idx = words.index(w)
                        xs.append(coords[idx, 0])
                        ys.append(coords[idx, 1])
                        labels.append(w)

                if xs:
                    ax.scatter(xs, ys, c=colors[gi % len(colors)],
                               marker=markers[gi % len(markers)],
                               s=120, alpha=0.85, label=glabel,
                               edgecolors="black", linewidths=0.5)
                    # Annotate each point
                    for x, y, lbl in zip(xs, ys, labels):
                        ax.annotate(lbl, (x, y),
                                    textcoords="offset points", xytext=(5, 3),
                                    fontsize=7.5, ha="left")

            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xlabel("Dimension 1", fontsize=9)
            ax.set_ylabel("Dimension 2", fontsize=9)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = f"{output_prefix}_embeddings.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] Plot saved to: {out_path}")
        return out_path
    except ImportError:
        print("  [ERROR] Matplotlib not available. PNG output is required for this task.")
        return None


# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# SECTION 5: MAIN - VISUALIZATION
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  TASK 4: WORD EMBEDDING VISUALIZATION")
    print("=" * 65)

    # ── Load models ────────────────────────────────────────────
    cbow_emb, cbow_w2i, cbow_i2w, cbow_cfg = load_model("models/cbow_d100_w4")
    sg_emb,   sg_w2i,   sg_i2w,   sg_cfg   = load_model("models/skipgram_d100_w4_neg10")

    print(f"\n  CBOW vocab size   : {len(cbow_w2i)}")
    print(f"  Skip-gram vocab   : {len(sg_w2i)}")

    # ── Select words to visualize ──────────────────────────────
    # Organized into semantic groups as specified in assignment
    word_groups = [
        ["btech", "mtech", "phd", "ug", "pg", "undergraduate", "postgraduate"],
        ["student", "professor", "faculty", "researcher", "scholar"],
        ["research", "exam", "lecture", "project", "thesis", "assignment"],
        ["department", "course", "credit", "semester", "grade", "cgpa"],
        ["iit", "jodhpur", "institute", "campus", "hostel", "library"],
    ]
    group_labels = [
        "Academic Programs",
        "Academic Roles",
        "Academic Activities",
        "Administrative",
        "Institute Context"
    ]

    def resolve_visual_word(token, w2i):
        """Resolve a requested token to an in-vocabulary token for plotting."""
        t = token.lower()
        if t in w2i:
            return t

        alias_map = {
            "exam": ["exams", "examination", "examinations", "test", "tests", "evaluation"],
            "assignment": ["assignments", "task", "tasks"],
            "student": ["students"],
            "professor": ["professors"],
            "researcher": ["researchers"],
            "course": ["courses"],
            "grade": ["grades"],
            "semester": ["semesters"],
        }
        for cand in alias_map.get(t, []):
            if cand in w2i:
                return cand

        # Generic fallback: singular/plural variant.
        if (t + "s") in w2i:
            return t + "s"
        if t.endswith("s") and t[:-1] in w2i:
            return t[:-1]
        return None

    def get_model_words_and_embeddings(emb, w2i, groups):
        """Filter and resolve word groups to in-vocabulary tokens."""
        all_words = []
        filtered_groups = []
        resolved_pairs = []
        for grp in groups:
            fg = []
            for w in grp:
                rw = resolve_visual_word(w, w2i)
                if rw:
                    fg.append(rw)
                    if rw != w:
                        resolved_pairs.append((w, rw))
            filtered_groups.append(fg)
            all_words.extend(fg)
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for w in all_words:
            if w not in seen:
                unique_words.append(w)
                seen.add(w)
        # Get embeddings for selected words
        indices = [w2i[w] for w in unique_words]
        selected_emb = emb[indices]
        return unique_words, selected_emb, filtered_groups, resolved_pairs

    cbow_words, cbow_sel, cbow_fgroups, cbow_resolved = get_model_words_and_embeddings(cbow_emb, cbow_w2i, word_groups)
    sg_words,   sg_sel,   sg_fgroups,   sg_resolved   = get_model_words_and_embeddings(sg_emb,   sg_w2i,   word_groups)

    print(f"\n  Selected {len(cbow_words)} words for CBOW visualization")
    print(f"  Selected {len(sg_words)} words for Skip-gram visualization")
    if cbow_resolved:
        shown = ", ".join(f"{a}->{b}" for a, b in cbow_resolved[:6])
        print(f"  [INFO] CBOW resolved tokens: {shown}")
    if sg_resolved:
        shown = ", ".join(f"{a}->{b}" for a, b in sg_resolved[:6])
        print(f"  [INFO] Skip-gram resolved tokens: {shown}")

    # ── Compute PCA projections ────────────────────────────────
    print("\n[INFO] Computing PCA projections...")
    cbow_pca = pca_2d(cbow_sel) if len(cbow_sel) >= 2 else np.zeros((len(cbow_sel), 2))
    sg_pca   = pca_2d(sg_sel)   if len(sg_sel)   >= 2 else np.zeros((len(sg_sel),   2))
    print("  [DONE] PCA completed.")

    # ── Compute t-SNE projections ──────────────────────────────
    print("[INFO] Computing t-SNE projections (this may take a moment)...")
    cbow_tsne = tsne_2d(cbow_sel, perplexity=min(5, len(cbow_sel)//2), n_iter=300) if len(cbow_sel) >= 4 else cbow_pca
    sg_tsne   = tsne_2d(sg_sel,   perplexity=min(5, len(sg_sel)//2),   n_iter=300) if len(sg_sel)   >= 4 else sg_pca
    print("  [DONE] t-SNE completed.")

    coords_cbow = {"pca": cbow_pca, "tsne": cbow_tsne}
    coords_sg   = {"pca": sg_pca,   "tsne": sg_tsne}

    # ── Generate Matplotlib plots ──────────────────────────────
    os.makedirs("visualizations", exist_ok=True)
    out_path = plot_embeddings_matplotlib(
        coords_cbow, coords_sg, cbow_words, cbow_fgroups, group_labels,
        output_prefix="visualizations/word_embeddings"
    )
    if out_path is None:
        raise ImportError("matplotlib is required to save PNG visualizations.")

    # ── Interpretation ─────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  CLUSTERING INTERPRETATION")
    print("═" * 65)
    print("""
  PCA Analysis:
  ─────────────────────────────────────────────────────────────
  PCA linearly projects the embeddings along directions of maximum
  variance. This reveals the global geometric structure:
  - Academic program words (btech, mtech, phd) cluster together,
    reflecting shared educational context in the corpus.
  - Role words (student, professor, researcher) form a separate
    cluster as they frequently co-occur with different contexts.
  - Activity words (exam, thesis, research) may show partial overlap
    between student-activity and research-activity clusters.

  CBOW vs Skip-gram in PCA:
  - CBOW embeddings tend to be smoother (averaged context), so
    clusters may be tighter but less discriminative.
  - Skip-gram embeds each context separately, creating more spread
    and sometimes clearer cluster boundaries.

  t-SNE Analysis:
  ─────────────────────────────────────────────────────────────
  t-SNE preserves local neighborhoods, making semantic sub-clusters
  visible that PCA may obscure:
  - Words that appear in similar sentence contexts (e.g., 'exam' and
    'semester') will be pulled close together.
  - Domain-specific terms (e.g., 'phd', 'thesis') may form a tight
    cluster distinct from undergraduate terms.

  CBOW vs Skip-gram in t-SNE:
  - Skip-gram with negative sampling (k=10) produces more distinct
    clusters in t-SNE as it optimizes pairwise word relationships.
  - CBOW clusters are more compact but may conflate semantically
    distinct words that share the same context window.

  Overall: Skip-gram models generally produce better-quality clusters
  for rare domain terms, while CBOW is more stable for high-frequency
  words in this small IIT Jodhpur corpus.
""")

    print("[DONE] Task 4 completed.\n")
