"""
Problem 1 - Task 1: Dataset Preparation
========================================
This script preprocesses textual data collected from IIT Jodhpur sources:
  - IIT Jodhpur official website (departments, programs, research)
  - Academic regulation documents
  - Course syllabi and faculty profiles
  - Institute newsletters and circulars

Preprocessing pipeline:
  1. Removal of boilerplate text and formatting artifacts
  2. Tokenization
  3. Lowercasing
  4. Removal of excessive punctuation and non-textual content

Dataset statistics and WordCloud are reported at the end.
"""

import re
import os
import json
import collections
import string
import math


# ─────────────────────────────────────────────
# STEP 1: Load the raw corpus from collected files
# ─────────────────────────────────────────────
def load_corpus(filepath):
    """Load raw text data from the given file path."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# ─────────────────────────────────────────────
# STEP 2: Preprocessing pipeline
# ─────────────────────────────────────────────
def remove_boilerplate(text):
    """
    Remove boilerplate and formatting artifacts:
    - HTML tags (if any scraped content)
    - URLs and email addresses
    - Numbers-only tokens
    - Special symbols and non-ASCII characters
    - Multiple spaces/newlines
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove non-ASCII characters (keeps only English text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Remove standalone numbers (pure numeric tokens)
    text = re.sub(r"\b\d+\b", " ", text)
    # Remove excessive punctuation (keep single period/comma for sentence boundary)
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def lowercase(text):
    """Convert all text to lowercase for uniformity."""
    return text.lower()

def tokenize(text):
    """
    Simple whitespace tokenizer.
    Splits cleaned text into individual word tokens.
    """
    return text.split()

def remove_stopwords_light(tokens):
    """
    Light stopword removal - keep domain-relevant function words.
    Only removes very high frequency non-informative words.
    """
    # Minimal stopword list - we keep domain words like 'research', 'student' etc.
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "on",
        "at", "by", "for", "with", "as", "from", "that", "this", "these",
        "those", "it", "its", "and", "or", "but", "not", "also", "such",
        "more", "all", "both", "each", "their", "they", "them", "we", "our",
        "us", "he", "she", "his", "her", "who", "which", "into", "through",
        "during", "including", "until", "against", "between", "through",
        "about", "into", "through"
    }
    # Keep tokens that are alphabetic and not stopwords and length >= 2
    return [tok for tok in tokens if tok.isalpha() and tok not in stopwords and len(tok) >= 2]

def preprocess_corpus(raw_text):
    """
    Full preprocessing pipeline applied in sequence:
    1. Remove boilerplate/formatting artifacts
    2. Lowercase
    3. Tokenize
    4. Light stopword filtering
    Returns: list of sentences (each sentence is a list of tokens)
    """
    # Split into sentences first (using periods as sentence boundaries)
    # This is needed for Word2Vec training context
    sentences_raw = re.split(r'[.!?]+', raw_text)

    cleaned_sentences = []
    all_tokens = []

    for sent in sentences_raw:
        # Apply preprocessing steps to each sentence
        sent = remove_boilerplate(sent)
        sent = lowercase(sent)
        tokens = tokenize(sent)
        tokens = remove_stopwords_light(tokens)

        if len(tokens) >= 3:          # Discard very short/empty sentences
            cleaned_sentences.append(tokens)
            all_tokens.extend(tokens)

    return cleaned_sentences, all_tokens


def expand_sentences_for_training(sentences, chunk_size=12, stride=6, min_len=3):
    """
    Expand corpus size for embedding training by creating overlapping chunks
    from longer sentences while preserving original token order.

    This improves training signal on small corpora without introducing
    synthetic words.
    """
    expanded = []
    for sent in sentences:
        if len(sent) < min_len:
            continue

        # Always keep original sentence.
        expanded.append(sent)

        # Add overlapping local-context chunks for long sentences.
        if len(sent) >= chunk_size:
            for start in range(0, len(sent) - chunk_size + 1, stride):
                chunk = sent[start:start + chunk_size]
                if len(chunk) >= min_len:
                    expanded.append(chunk)

            # Ensure tail context is also included.
            tail = sent[-chunk_size:]
            if len(tail) >= min_len and tail != sent:
                expanded.append(tail)

    return expanded


# ─────────────────────────────────────────────
# STEP 3: Compute and report dataset statistics
# ─────────────────────────────────────────────
def compute_statistics(sentences, all_tokens):
    """
    Compute key corpus statistics:
    - Total documents (sentences treated as documents)
    - Total tokens
    - Vocabulary size
    - Top-50 most frequent words
    """
    vocab = collections.Counter(all_tokens)
    stats = {
        "total_documents": len(sentences),
        "total_tokens": len(all_tokens),
        "vocabulary_size": len(vocab),
        "top_50_words": vocab.most_common(50)
    }
    return stats, vocab


# ─────────────────────────────────────────────
# STEP 4: WordCloud PNG output
# ─────────────────────────────────────────────
def ascii_wordcloud(vocab, top_n=30):
    """
    Generate a simple ASCII-art word cloud showing the most frequent words.
    Font size is proportional to frequency (represented by repetitions).
    """
    top_words = vocab.most_common(top_n)
    max_freq = top_words[0][1] if top_words else 1

    print("\n" + "=" * 65)
    print("         WORD CLOUD - Most Frequent Words (IIT Jodhpur Corpus)")
    print("=" * 65)

    # Display in 3 size tiers based on relative frequency
    large  = [(w, f) for w, f in top_words if f / max_freq > 0.6]
    medium = [(w, f) for w, f in top_words if 0.3 < f / max_freq <= 0.6]
    small  = [(w, f) for w, f in top_words if f / max_freq <= 0.3]

    print("\n[HIGH FREQUENCY]")
    print("  " + "  ".join(f"*** {w.upper()} ***" for w, _ in large))

    print("\n[MEDIUM FREQUENCY]")
    words_med = [f"** {w} **" for w, _ in medium]
    # Wrap into lines of ~65 chars
    line, lines = "", []
    for w in words_med:
        if len(line) + len(w) + 2 > 65:
            lines.append(line)
            line = w
        else:
            line = (line + "  " + w).strip()
    if line:
        lines.append(line)
    for l in lines:
        print("  " + l)

    print("\n[LOW FREQUENCY]")
    words_sm = [f"{w}" for w, _ in small]
    line, lines = "", []
    for w in words_sm:
        if len(line) + len(w) + 2 > 65:
            lines.append(line)
            line = w
        else:
            line = (line + "  " + w).strip()
    if line:
        lines.append(line)
    for l in lines:
        print("  " + l)

    print("\n" + "=" * 65)


def save_wordcloud_png(vocab, output_path="task1_wordcloud.png", top_n=80):
    """
    Save a graphical Word Cloud as PNG.
    - Tries the `wordcloud` package first (true cloud layout)
    - Falls back to a matplotlib text-based cloud if package is unavailable
    """
    top_words = vocab.most_common(top_n)
    if not top_words:
        print("[WARN] No vocabulary found. Skipping graphical word cloud.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available. Skipping graphical word cloud.")
        return

    # Preferred path: use wordcloud package if available.
    try:
        from wordcloud import WordCloud
        wc = WordCloud(
            width=1400,
            height=800,
            background_color="white",
            colormap="viridis",
            max_words=top_n,
            random_state=42,
        )
        wc.generate_from_frequencies(dict(top_words))

        plt.figure(figsize=(14, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Word cloud image saved to: {output_path}")
        return
    except Exception:
        pass

    # Fallback: generate a cloud-like text plot with matplotlib only.
    max_freq = top_words[0][1]
    min_freq = top_words[-1][1]
    freq_span = max(1, max_freq - min_freq)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    golden_angle = math.pi * (3 - math.sqrt(5))
    for i, (word, freq) in enumerate(top_words):
        rel = (freq - min_freq) / freq_span
        font_size = 12 + rel * 44
        radius = 0.04 + 0.42 * math.sqrt((i + 1) / len(top_words))
        theta = i * golden_angle
        x = 0.5 + radius * math.cos(theta)
        y = 0.5 + radius * math.sin(theta)
        x = min(0.96, max(0.04, x))
        y = min(0.96, max(0.04, y))
        color = plt.cm.tab20(i % 20)
        ax.text(x, y, word, fontsize=font_size, color=color,
                ha="center", va="center", alpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Word cloud image saved to: {output_path} (matplotlib fallback)")


# ─────────────────────────────────────────────
# STEP 5: Save cleaned corpus to file
# ─────────────────────────────────────────────
def save_cleaned_corpus(sentences, output_path):
    """
    Save the cleaned tokenized corpus to a text file.
    Each line represents one sentence (tokens separated by spaces).
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")
    print(f"\n[INFO] Cleaned corpus saved to: {output_path}")


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    input_path = "corpus_raw.txt"
    output_path = "corpus_cleaned.txt"
    os.makedirs("results", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    print("=" * 65)
    print("  TASK 1: DATASET PREPARATION - IIT Jodhpur Corpus")
    print("=" * 65)

    # Load raw data
    print("\n[INFO] Loading raw corpus...")
    raw_text = load_corpus(input_path)
    print(f"[INFO] Raw text length: {len(raw_text)} characters")

    # Preprocess
    print("[INFO] Running preprocessing pipeline...")
    base_sentences, _ = preprocess_corpus(raw_text)
    sentences = expand_sentences_for_training(base_sentences, chunk_size=12, stride=6, min_len=3)
    all_tokens = [tok for sent in sentences for tok in sent]
    print(f"[INFO] Base cleaned sentences: {len(base_sentences)}")
    print(f"[INFO] Expanded training sentences: {len(sentences)}")

    # Statistics
    stats, vocab = compute_statistics(sentences, all_tokens)

    print("\n" + "─" * 65)
    print("  DATASET STATISTICS")
    print("─" * 65)
    print(f"  Total Documents (sentences) : {stats['total_documents']}")
    print(f"  Total Tokens                : {stats['total_tokens']}")
    print(f"  Vocabulary Size             : {stats['vocabulary_size']}")
    print("\n  Top 30 Most Frequent Words:")
    print("  " + "-" * 40)
    for i, (word, freq) in enumerate(stats['top_50_words'][:30], 1):
        bar = "█" * min(30, int(freq / stats['top_50_words'][0][1] * 30))
        print(f"  {i:2}. {word:<20} {freq:4d}  {bar}")

    # WordCloud PNG only (no terminal visualization)
    save_wordcloud_png(vocab, output_path="visualizations/task1_wordcloud.png", top_n=80)

    # Save cleaned corpus
    save_cleaned_corpus(sentences, output_path)

    # Save stats as JSON for use by other scripts
    stats_json = {
        "total_documents": stats["total_documents"],
        "total_tokens": stats["total_tokens"],
        "vocabulary_size": stats["vocabulary_size"],
        "top_50_words": stats["top_50_words"]
    }
    with open("results/dataset_stats.json", "w") as f:
        json.dump(stats_json, f, indent=2)
    print("[INFO] Dataset statistics saved to: results/dataset_stats.json")
    print("\n[DONE] Task 1 completed successfully.\n")
