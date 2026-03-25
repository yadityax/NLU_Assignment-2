import numpy as np
import json
import random
import collections
import re
import os



# SECTION 1: RELOAD MODELS AND GENERATE NAMES

def load_names_set(filepath):
    """
    Load a conservative novelty reference set from training data.
    Includes:
      - full line-level entries (normalized)
      - individual alphabetic tokens from each line
    This prevents inflated 100% novelty when generated names match common
    first/last-name tokens seen in training lines.
    """
    names = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Normalize each line to alphabetic-only token string.
            tokens = re.findall(r"[A-Za-z]+", line.lower())
            if not tokens:
                continue
            entry = "".join(tokens)
            if len(entry) >= 2:
                names.add(entry)
            for tok in tokens:
                if len(tok) >= 2:
                    names.add(tok)
    return names


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def tanh_fn(x):
    return np.tanh(x)

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-10)

START_TOKEN = "<"
END_TOKEN   = ">"


def one_hot(idx, vocab_size):
    v = np.zeros(vocab_size)
    v[idx] = 1.0
    return v


#  Vanilla RNN generator
class VanillaRNNGenerator:
    def __init__(self, data):
        
        self.W_xh = data["W_xh"]
        self.W_hh = data["W_hh"]
        self.b_h  = data["b_h"]
        self.W_hy = data["W_hy"]
        self.b_y  = data["b_y"]
        self.vocab_size  = int(data["vocab_size"])
        self.hidden_size = int(data["hidden_size"])
        self.char2idx = json.loads(str(data["char2idx"]))
        self.idx2char = json.loads(str(data["idx2char"]))

    def generate(self, temperature=0.8, max_len=20):
        h = np.zeros(self.hidden_size)
        idx = self.char2idx[START_TOKEN]
        name = []
        for _ in range(max_len):
            x = one_hot(idx, self.vocab_size)
            h_raw = self.W_xh @ x + self.W_hh @ h + self.b_h
            h = tanh_fn(h_raw)
            y = softmax((self.W_hy @ h + self.b_y) / temperature)
            idx = np.random.choice(self.vocab_size, p=y)
            c = self.idx2char[str(idx)]
            if c == END_TOKEN: break
            if c != START_TOKEN: name.append(c)
        return "".join(name).capitalize()


class LSTMCellInference:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.hidden_size = W.shape[0] // 4

    def step(self, x, h, c):
        H = self.hidden_size
        concat = np.concatenate([h, x])
        gates = self.W @ concat + self.b
        f = sigmoid(gates[0*H:1*H])
        i = sigmoid(gates[1*H:2*H])
        g = tanh_fn(gates[2*H:3*H])
        o = sigmoid(gates[3*H:4*H])
        c_new = f * c + i * g
        h_new = o * tanh_fn(c_new)
        return h_new, c_new


class BLSTMGenerator:
    def __init__(self, data):
        self.fwd = LSTMCellInference(data["W_fwd"], data["b_fwd"])
        self.bwd = LSTMCellInference(data["W_bwd"], data["b_bwd"])
        self.W_out = data["W_out"]
        self.b_out = data["b_out"]
        self.char2idx = json.loads(str(data["char2idx"]))
        self.idx2char = json.loads(str(data["idx2char"]))
        self.vocab_size  = self.W_out.shape[0]
        self.hidden_size = self.fwd.hidden_size

    def generate(self, temperature=0.8, max_len=20):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        dummy = np.zeros(self.hidden_size)
        idx = self.char2idx[START_TOKEN]
        name = []
        for _ in range(max_len):
            x = one_hot(idx, self.vocab_size)
            h, c = self.fwd.step(x, h, c)
            combined = np.concatenate([h, dummy])
            y = softmax((self.W_out @ combined + self.b_out) / temperature)
            idx = np.random.choice(self.vocab_size, p=y)
            ch = self.idx2char[str(idx)]
            if ch == END_TOKEN: break
            if ch != START_TOKEN: name.append(ch)
        return "".join(name).capitalize()


class AttnRNNGenerator:
    def __init__(self, data):
        self.W_xh = data["W_xh"]
        self.W_hh = data["W_hh"]
        self.b_h  = data["b_h"]
        self.W_hy = data["W_hy"]
        self.b_y  = data["b_y"]
        self.W_a  = data["W_a"]
        self.b_a  = data["b_a"]
        self.v    = data["v"]
        self.char2idx = json.loads(str(data["char2idx"]))
        self.idx2char = json.loads(str(data["idx2char"]))
        self.vocab_size  = self.W_hy.shape[0]
        self.hidden_size = self.W_xh.shape[0]

    def attend(self, h, h_history):
        H_all = np.stack(h_history, axis=0)
        proj = np.tanh(H_all @ self.W_a.T + self.b_a)
        scores = proj @ self.v
        alpha = softmax(scores)
        ctx = alpha @ H_all
        return ctx

    def generate(self, temperature=0.8, max_len=20):
        h = np.zeros(self.hidden_size)
        idx = self.char2idx[START_TOKEN]
        name = []
        h_hist = [h.copy()]
        for _ in range(max_len):
            x = one_hot(idx, self.vocab_size)
            h_raw = self.W_xh @ x + self.W_hh @ h + self.b_h
            h = tanh_fn(h_raw)
            h_hist.append(h.copy())
            ctx = self.attend(h, h_hist)
            combined = np.concatenate([h, ctx])
            y = softmax((self.W_hy @ combined + self.b_y) / temperature)
            idx = np.random.choice(self.vocab_size, p=y)
            ch = self.idx2char[str(idx)]
            if ch == END_TOKEN: break
            if ch != START_TOKEN: name.append(ch)
        return "".join(name).capitalize()



# SECTION 2: EVALUATION METRICS

def generate_names_batch(generator, n=200, temperature=0.8):
    
    generated = []
    attempts = 0
    while len(generated) < n and attempts < n * 5:
        name = generator.generate(temperature=temperature)
        attempts += 1
        if name and len(name) >= 2 and name.replace("'", "").replace("-", "").isalpha():
            generated.append(name)
    return generated


def novelty_rate(generated_names, training_names_set):
    """
    Novelty Rate = (# generated names NOT in training set) / (total generated)

    A high novelty rate indicates the model is truly generating new names
    rather than memorizing and reproducing training examples.
    """
    novel = sum(1 for n in generated_names if n.lower() not in training_names_set)
    return novel / len(generated_names) if generated_names else 0.0


def diversity_score(generated_names):
    """
    Diversity = (# unique generated names) / (total generated names)

    Measures variety in generated output. Low diversity suggests the model
    is collapsing to a small set of repeated outputs (mode collapse).
    """
    unique = len(set(n.lower() for n in generated_names))
    return unique / len(generated_names) if generated_names else 0.0


def avg_length(names):
    if not names: return 0.0
    return sum(len(n) for n in names) / len(names)


def length_distribution(names):
    dist = collections.Counter(len(n) for n in names)
    return dict(sorted(dist.items()))


def char_frequency(names):
    
    first_chars = collections.Counter(n[0].lower() for n in names if n)
    return dict(first_chars.most_common(10))


def starts_with_vowel_ratio(names):
    
    vowels = set("aeiou")
    return sum(1 for n in names if n and n[0].lower() in vowels) / max(1, len(names))



# SECTION 3: QUALITATIVE ANALYSIS HELPERS

# Common Indian name substrings (for basic realism check)
INDIAN_PATTERNS = [
    r"an", r"ar", r"am", r"av", r"ay", r"ab",  # Common beginnings
    r"sh", r"ra", r"ri", r"ru", r"re",          # Common clusters
    r"th", r"kr", r"pr", r"tr",                  # Common clusters
    r"a$", r"i$", r"u$", r"e$", r"n$",           # Common endings
    r"ya", r"va", r"la", r"ma", r"na",           # Common patterns
]

def realism_score(name):
    
    if not name or len(name) < 2:
        return 0.0
    name_lower = name.lower()

    score = 0.0
    # Length check (4-12 is ideal)
    if 4 <= len(name) <= 12:
        score += 0.3
    elif 2 <= len(name) <= 15:
        score += 0.1

    # Pattern matching
    pattern_hits = sum(1 for p in INDIAN_PATTERNS if re.search(p, name_lower))
    score += min(0.4, pattern_hits * 0.05)

    # Vowel ratio (Indian names typically 40-60% vowels)
    vowels = set("aeiou")
    vowel_ratio = sum(1 for c in name_lower if c in vowels) / len(name_lower)
    if 0.35 <= vowel_ratio <= 0.65:
        score += 0.3

    return min(1.0, score)


def classify_failure_modes(generated_names):
    
    failures = collections.defaultdict(list)
    for name in generated_names:
        name_lower = name.lower()
        if len(name) < 3:
            failures["too_short"].append(name)
        elif len(name) > 15:
            failures["too_long"].append(name)
        elif any(name_lower.count(c*3) > 0 for c in set(name_lower)):
            failures["repetitive_chars"].append(name)
        elif not any(c in "aeiou" for c in name_lower):
            failures["no_vowels"].append(name)
        elif realism_score(name) < 0.3:
            failures["low_realism"].append(name)
    return dict(failures)


# SECTION 4: MAIN - EVALUATION

if __name__ == "__main__":
    print("=" * 65)
    print("  TASKS 2 & 3: EVALUATION AND QUALITATIVE ANALYSIS")
    print("=" * 65)

    # Load training names for novelty check 
    training_names = load_names_set("TrainingNames.txt")
    print(f"\n  Loaded {len(training_names)} training names.")

    #  Load saved models 
    print("\n Loading trained models...")

    models = {}
    model_dir = "models"

    rnn_path = os.path.join(model_dir, "rnn_model.npz")
    blstm_path = os.path.join(model_dir, "blstm_model.npz")
    attn_path = os.path.join(model_dir, "attn_model.npz")

    if os.path.exists(rnn_path):
        rnn_data = np.load(rnn_path, allow_pickle=True)
        models["Vanilla RNN"] = VanillaRNNGenerator(rnn_data)
        print("  Vanilla RNN loaded.")
    else:
        print(" models/rnn_model.npz not found. Run task1 first.")

    if os.path.exists(blstm_path):
        blstm_data = np.load(blstm_path, allow_pickle=True)
        models["Bidirectional LSTM"] = BLSTMGenerator(blstm_data)
        print(" BLSTM loaded.")
    else:
        print(" models/blstm_model.npz not found. Run task1 first.")

    if os.path.exists(attn_path):
        attn_data = np.load(attn_path, allow_pickle=True)
        models["RNN + Attention"] = AttnRNNGenerator(attn_data)
        print(" RNN + Attention loaded.")
    else:
        print(" models/attn_model.npz not found. Run task1 first.")

    if not models:
        print("\n No models found. Please run task1_model_implementation.py first.")
        exit(1)

    N_GENERATE = 200       # Number of names to generate per model
    TEMPERATURES = [0.7, 0.8, 0.9, 1.0]

    results_all = {}

    for model_name, generator in models.items():
        print(f"\n{'─'*65}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─'*65}")

        # Generate/evaluate at multiple temperatures and select by quality balance.
        temp_runs = []
        print(f"  Generating {N_GENERATE} names per temperature: {TEMPERATURES}")
        for temp in TEMPERATURES:
            generated_t = generate_names_batch(generator, n=N_GENERATE, temperature=temp)
            nov_t = novelty_rate(generated_t, training_names)
            div_t = diversity_score(generated_t)
            realism_t = float(np.mean([realism_score(n) for n in generated_t])) if generated_t else 0.0
            avg_len_t = avg_length(generated_t)
            failures_t = classify_failure_modes(generated_t)
            fail_ratio_t = (sum(len(v) for v in failures_t.values()) / max(1, len(generated_t)))
            # Prefer realistic names with reasonable length while still rewarding novelty/diversity.
            len_penalty = min(abs(avg_len_t - 6.5) / 6.5, 1.0)
            score_t = (
                0.25 * nov_t +
                0.20 * div_t +
                0.45 * realism_t +
                0.10 * (1.0 - len_penalty) -
                0.15 * fail_ratio_t
            )
            temp_runs.append((temp, generated_t, nov_t, div_t, realism_t, avg_len_t, fail_ratio_t, score_t))

        # Keep only quality-feasible runs first, then fallback to best overall score.
        feasible = [r for r in temp_runs if (4.0 <= r[5] <= 9.5 and r[4] >= 0.48 and r[6] <= 0.45)]
        if feasible:
            best_temp, generated, nov_rate, div_score, avg_realism, avg_len, fail_ratio, _ = max(feasible, key=lambda x: x[7])
            selection_note = "best quality-constrained setting"
        else:
            best_temp, generated, nov_rate, div_score, avg_realism, avg_len, fail_ratio, _ = max(temp_runs, key=lambda x: x[7])
            selection_note = "best available (constraints unmet)"
        print(f"  Selected temperature: {best_temp:.1f} (best overall quality balance)")
        print(f"  Selection mode: {selection_note}")
        print(f"  Successfully generated: {len(generated)} names")

        # Quantitative metrics on selected run
        len_dist = length_distribution(generated)
        first_char_dist = char_frequency(generated)
        vowel_ratio = starts_with_vowel_ratio(generated)

        print(f"\n  QUANTITATIVE METRICS:")
        print(f"  {'Metric':<35} {'Value':>10}")
        print(f"  {'─'*48}")
        for temp, _, nov_t, div_t, realism_t, avg_len_t, fail_ratio_t, _ in temp_runs:
            print(
                f"  {f'Temp {temp:.1f} (N/D/R/L/F)':<35} "
                f"{nov_t:>5.1%} / {div_t:>5.1%} / {realism_t:>5.3f} / {avg_len_t:>4.1f} / {fail_ratio_t:>5.1%}"
            )
        print(f"  {'Novelty Rate':<35} {nov_rate:>9.1%}")
        print(f"  {'Diversity Score':<35} {div_score:>9.1%}")
        print(f"  {'Average Name Length':<35} {avg_len:>9.2f}")
        print(f"  {'Vowel-Start Ratio':<35} {vowel_ratio:>9.1%}")
        print(f"  {'Avg Realism Score (heuristic)':<35} {avg_realism:>9.3f}")
        print(f"\n  Length Distribution: {len_dist}")
        print(f"  Top First-Char Distribution: {first_char_dist}")

        # Failure statistics kept for reporting/selection only.
        failures = classify_failure_modes(generated)

        # Save all generated names
        output_file = f"generated_{model_name.replace(' ', '_').replace('+', 'plus')}.txt"
        with open(output_file, "w") as f:
            for name in generated:
                f.write(name + "\n")
        print(f"\n Generated names → {output_file}")

        results_all[model_name] = {
            "selected_temperature": float(best_temp),
            "selection_mode": selection_note,
            "novelty_rate": round(nov_rate, 4),
            "diversity_score": round(div_score, 4),
            "avg_length": round(avg_len, 2),
            "vowel_start_ratio": round(vowel_ratio, 4),
            "avg_realism": round(float(avg_realism), 4),
            "failure_ratio": round(float(fail_ratio), 4),
            "n_generated": len(generated),
            "failure_counts": {k: len(v) for k, v in failures.items()}
        }

    # COMPARISON TABLE
    print("\n" + "═" * 65)
    print("  TASK 2: COMPARATIVE EVALUATION TABLE")
    print("═" * 65)
    print(f"  {'Model':<25} {'Temp':>6} {'Novelty':>8} {'Diversity':>10} {'AvgLen':>8} {'Realism':>8}")
    print("  " + "─" * 70)
    for model_name, r in results_all.items():
        print(f"  {model_name:<25} {r['selected_temperature']:>5.1f} {r['novelty_rate']:>7.1%} "
              f"{r['diversity_score']:>9.1%} {r['avg_length']:>7.1f}  "
              f"{r['avg_realism']:>7.3f}")

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(" evaluation_results.json")
    print("\n Tasks 2 & 3 completed.\n")
