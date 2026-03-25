import numpy as np
import json
import os
import random
import time
import re


# SECTION 1: DATA PREPARATION

# Special tokens
START_TOKEN = "<"      # Beginning of name
END_TOKEN   = ">"      # End of name
PAD_TOKEN   = " "      # Padding (not used in training but reserved)


def load_names(filepath):
    names = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Handle lines like "Myra Dubey" by extracting alphabetic tokens.
            for name in re.findall(r"[A-Za-z]+", line):
                if len(name) >= 2:
                    names.append(name.lower())
    return names


def build_char_vocab(names):
    
    # Collect all unique characters
    chars = set()
    for name in names:
        chars.update(set(name))
    chars.add(START_TOKEN)
    chars.add(END_TOKEN)

    vocab = sorted(chars)
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, vocab


def name_to_sequences(name, char2idx):
    
    input_seq  = [char2idx[START_TOKEN]] + [char2idx[c] for c in name]
    target_seq = [char2idx[c] for c in name] + [char2idx[END_TOKEN]]
    return input_seq, target_seq


def one_hot(idx, vocab_size):
    
    v = np.zeros(vocab_size)
    v[idx] = 1.0
    return v


# SECTION 2: UTILITY FUNCTIONS

def sigmoid(x):
    
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    
    return np.tanh(x)


def softmax(x):
    
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-10)


def cross_entropy_loss(probs, target_idx):
    
    return -np.log(probs[target_idx] + 1e-10)


def clip_gradients(grads_dict, clip_value=5.0):
    
    for key in grads_dict:
        np.clip(grads_dict[key], -clip_value, clip_value, out=grads_dict[key])
    return grads_dict


# SECTION 3: VANILLA RNN

class VanillaRNN:

    def __init__(self, vocab_size, hidden_size=64, lr=0.01):
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.n_layers    = 1
        self.lr          = lr

        # Xavier-like initialization: scale by 1/sqrt(fan_in)
        scale_xh = np.sqrt(1.0 / vocab_size)
        scale_hh = np.sqrt(1.0 / hidden_size)

        # Weight matrices
        self.W_xh = np.random.randn(hidden_size, vocab_size)  * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.b_h  = np.zeros(hidden_size)
        self.W_hy = np.random.randn(vocab_size, hidden_size)  * scale_hh
        self.b_y  = np.zeros(vocab_size)

    @property
    def n_params(self):
        """Total number of trainable parameters."""
        return (self.hidden_size * self.vocab_size +   # W_xh
                self.hidden_size * self.hidden_size +  # W_hh
                self.hidden_size +                     # b_h
                self.vocab_size  * self.hidden_size +  # W_hy
                self.vocab_size)                       # b_y

    def forward(self, input_seq):
        T = len(input_seq)
        h = np.zeros(self.hidden_size)   # Initial hidden state h_0 = 0

        h_states = [h.copy()]
        outputs  = []
        raw_outs = []

        for t in range(T):
            x_t = one_hot(input_seq[t], self.vocab_size)

            # Hidden state update: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
            h_raw = self.W_xh @ x_t + self.W_hh @ h + self.b_h
            h = tanh(h_raw)
            h_states.append(h.copy())

            # Output: y_t = softmax(W_hy h_t + b_y)
            y_raw = self.W_hy @ h + self.b_y
            y = softmax(y_raw)

            outputs.append(y)
            raw_outs.append(y_raw)

        return h_states, outputs, raw_outs

    def backward(self, input_seq, target_seq, h_states, outputs):
        
        T = len(input_seq)

        # Initialize gradient accumulators
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y  = np.zeros_like(self.b_y)

        dh_next = np.zeros(self.hidden_size)   # Gradient from future timestep
        total_loss = 0.0

        # BPTT: iterate backwards through time
        for t in reversed(range(T)):
            y      = outputs[t]
            target = target_seq[t]
            h_t    = h_states[t + 1]
            h_prev = h_states[t]
            x_t    = one_hot(input_seq[t], self.vocab_size)

            # Loss at timestep t
            total_loss += cross_entropy_loss(y, target)

            # Gradient of loss w.r.t. output logits (softmax + CE)
            dy = y.copy()
            dy[target] -= 1.0                          # dy: [vocab_size]

            # Gradient w.r.t. W_hy and b_y
            dW_hy += np.outer(dy, h_t)
            db_y  += dy

            # Gradient w.r.t. h_t (from output + future timestep)
            dh = self.W_hy.T @ dy + dh_next           # [hidden_size]

            # Gradient through tanh: dtanh = (1 - tanh^2) * upstream
            dh_raw = dh * (1.0 - h_t ** 2)            # [hidden_size]

            # Gradient w.r.t. weights and input
            dW_xh += np.outer(dh_raw, x_t)
            dW_hh += np.outer(dh_raw, h_prev)
            db_h  += dh_raw

            # Gradient to propagate to previous timestep
            dh_next = self.W_hh.T @ dh_raw

        # Gradient clipping
        grads = {"W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h,
                 "W_hy": dW_hy, "b_y": db_y}
        grads = clip_gradients(grads)

        # SGD parameter update
        self.W_xh -= self.lr * grads["W_xh"]
        self.W_hh -= self.lr * grads["W_hh"]
        self.b_h  -= self.lr * grads["b_h"]
        self.W_hy -= self.lr * grads["W_hy"]
        self.b_y  -= self.lr * grads["b_y"]

        return total_loss / T

    def train_step(self, input_seq, target_seq):
        h_states, outputs, _ = self.forward(input_seq)
        loss = self.backward(input_seq, target_seq, h_states, outputs)
        return loss

    def generate(self, char2idx, idx2char, max_len=20, temperature=0.8):
        
        h = np.zeros(self.hidden_size)
        input_idx = char2idx[START_TOKEN]
        name = []

        for _ in range(max_len):
            x_t = one_hot(input_idx, self.vocab_size)
            h_raw = self.W_xh @ x_t + self.W_hh @ h + self.b_h
            h = tanh(h_raw)
            y_raw = self.W_hy @ h + self.b_y

            # Apply temperature scaling before softmax
            y_scaled = softmax(y_raw / temperature)

            # Sample from distribution
            next_idx = np.random.choice(self.vocab_size, p=y_scaled)
            next_char = idx2char[next_idx]

            if next_char == END_TOKEN:
                break
            if next_char != START_TOKEN:
                name.append(next_char)
            input_idx = next_idx

        return "".join(name).capitalize() if name else ""


# SECTION 4: BIDIRECTIONAL LSTM (BLSTM)

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        # All 4 gate matrices stacked for efficiency: [4*H, input+hidden]
        scale = np.sqrt(2.0 / (concat_size + hidden_size))
        self.W = np.random.randn(4 * hidden_size, concat_size) * scale
        self.b = np.zeros(4 * hidden_size)

    @property
    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, x, h_prev, c_prev):
        
        # Concatenate input and previous hidden state
        concat = np.concatenate([h_prev, x])               # [hidden+input]

        # Compute all gates at once (one matrix multiply)
        gates_raw = self.W @ concat + self.b               # [4*hidden]
        H = self.hidden_size

        # Split into 4 gates
        f_raw = gates_raw[0*H : 1*H]
        i_raw = gates_raw[1*H : 2*H]
        g_raw = gates_raw[2*H : 3*H]
        o_raw = gates_raw[3*H : 4*H]

        f = sigmoid(f_raw)                                  # Forget gate
        i = sigmoid(i_raw)                                  # Input gate
        g = tanh(g_raw)                                     # Cell gate
        o = sigmoid(o_raw)                                  # Output gate

        c_t = f * c_prev + i * g                            # New cell state
        h_t = o * tanh(c_t)                                 # New hidden state

        cache = (concat, f, i, g, o, c_prev, c_t, h_t)
        return h_t, c_t, cache

    def backward(self, dh, dc, cache):
        concat, f, i, g, o, c_prev, c_t, h_t = cache
        H = self.hidden_size

        # Gradient through h_t = o * tanh(c_t)
        tanh_c = tanh(c_t)
        do = dh * tanh_c                                    # d_o
        dc_t = dh * o * (1.0 - tanh_c ** 2) + dc          # d_c_t (from h and future)

        # Gradients through cell update: c_t = f*c_prev + i*g
        df = dc_t * c_prev
        di = dc_t * g
        dg = dc_t * i
        dc_prev = dc_t * f

        # Gradients through gate activations (sigmoid/tanh)
        df_raw = df * f * (1.0 - f)
        di_raw = di * i * (1.0 - i)
        dg_raw = dg * (1.0 - g ** 2)
        do_raw = do * o * (1.0 - o)

        # Stack gate gradients
        dgates = np.concatenate([df_raw, di_raw, dg_raw, do_raw])  # [4*H]

        # Gradient w.r.t. weights and input
        dW = np.outer(dgates, concat)
        db = dgates
        dconcat = self.W.T @ dgates                         # [hidden+input]

        dh_prev = dconcat[:H]
        dx = dconcat[H:]

        return dx, dh_prev, dc_prev, dW, db


class BidirectionalLSTM:

    def __init__(self, vocab_size, hidden_size=48, lr=0.005):
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.n_layers    = 1
        self.n_directions = 2
        self.lr          = lr

        # Forward and backward LSTM cells
        self.fwd_lstm = LSTMCell(vocab_size, hidden_size)
        self.bwd_lstm = LSTMCell(vocab_size, hidden_size)

        # Output projection: combined [fwd, bwd] hidden → vocab
        out_size = 2 * hidden_size
        scale = np.sqrt(2.0 / (out_size + vocab_size))
        self.W_out = np.random.randn(vocab_size, out_size) * scale
        self.b_out = np.zeros(vocab_size)

    @property
    def n_params(self):
        """Total trainable parameters across all components."""
        return (self.fwd_lstm.n_params +
                self.bwd_lstm.n_params +
                self.W_out.size +
                self.b_out.size)

    def forward(self, input_seq):
        
        T  = len(input_seq)
        H  = self.hidden_size

        # Forward pass (left to right) 
        h_fwd = np.zeros(H)
        c_fwd = np.zeros(H)
        fwd_h_states = [h_fwd.copy()]
        fwd_caches   = []

        for t in range(T):
            x_t = one_hot(input_seq[t], self.vocab_size)
            h_fwd, c_fwd, cache = self.fwd_lstm.forward(x_t, h_fwd, c_fwd)
            fwd_h_states.append(h_fwd.copy())
            fwd_caches.append(cache)

        # Backward pass (right to left)
        h_bwd = np.zeros(H)
        c_bwd = np.zeros(H)
        bwd_h_states = [None] * T + [h_bwd.copy()]
        bwd_caches   = [None] * T

        for t in reversed(range(T)):
            x_t = one_hot(input_seq[t], self.vocab_size)
            h_bwd, c_bwd, cache = self.bwd_lstm.forward(x_t, h_bwd, c_bwd)
            bwd_h_states[t] = h_bwd.copy()
            bwd_caches[t]   = cache

        #  Compute outputs
        outputs = []
        for t in range(T):
            combined = np.concatenate([fwd_h_states[t+1], bwd_h_states[t]])
            y_raw = self.W_out @ combined + self.b_out
            outputs.append(softmax(y_raw))

        return fwd_h_states, bwd_h_states, fwd_caches, bwd_caches, outputs

    def train_step(self, input_seq, target_seq):
        """Forward + simplified backward pass with SGD update."""
        fwd_h, bwd_h, fwd_caches, bwd_caches, outputs = self.forward(input_seq)
        T = len(input_seq)

        # Accumulate gradients
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)
        dW_fwd = np.zeros_like(self.fwd_lstm.W)
        db_fwd = np.zeros_like(self.fwd_lstm.b)
        dW_bwd = np.zeros_like(self.bwd_lstm.W)
        db_bwd = np.zeros_like(self.bwd_lstm.b)

        dh_fwd_next = np.zeros(self.hidden_size)
        dc_fwd_next = np.zeros(self.hidden_size)
        dh_bwd_next = np.zeros(self.hidden_size)
        dc_bwd_next = np.zeros(self.hidden_size)

        total_loss = 0.0

        for t in reversed(range(T)):
            y      = outputs[t]
            target = target_seq[t]
            total_loss += cross_entropy_loss(y, target)

            # Output gradient
            dy = y.copy()
            dy[target] -= 1.0

            combined = np.concatenate([fwd_h[t+1], bwd_h[t]])
            dW_out += np.outer(dy, combined)
            db_out += dy

            # Split gradient to forward and backward LSTMs
            dcombined = self.W_out.T @ dy
            dh_fwd = dcombined[:self.hidden_size] + dh_fwd_next
            dh_bwd = dcombined[self.hidden_size:] + dh_bwd_next

            # Forward LSTM backward
            _, dh_fwd_next, dc_fwd_next, dW_f, db_f = self.fwd_lstm.backward(
                dh_fwd, dc_fwd_next, fwd_caches[t])
            dW_fwd += dW_f
            db_fwd += db_f

            # Backward LSTM backward
            _, dh_bwd_next, dc_bwd_next, dW_b, db_b = self.bwd_lstm.backward(
                dh_bwd, dc_bwd_next, bwd_caches[t])
            dW_bwd += dW_b
            db_bwd += db_b

        # Gradient clipping and updates
        for arr in [dW_out, db_out, dW_fwd, db_fwd, dW_bwd, db_bwd]:
            np.clip(arr, -5.0, 5.0, out=arr)

        self.W_out         -= self.lr * dW_out
        self.b_out         -= self.lr * db_out
        self.fwd_lstm.W    -= self.lr * dW_fwd
        self.fwd_lstm.b    -= self.lr * db_fwd
        self.bwd_lstm.W    -= self.lr * dW_bwd
        self.bwd_lstm.b    -= self.lr * db_bwd

        return total_loss / T

    def generate(self, char2idx, idx2char, max_len=20, temperature=0.8):
        
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        dummy_bwd = np.zeros(self.hidden_size)  # Approximation for inference

        input_idx = char2idx[START_TOKEN]
        name = []

        for _ in range(max_len):
            x_t = one_hot(input_idx, self.vocab_size)
            h, c, _ = self.fwd_lstm.forward(x_t, h, c)

            # At inference: use only forward hidden (concatenate with zeros for bwd)
            combined = np.concatenate([h, dummy_bwd])
            y_raw = self.W_out @ combined + self.b_out
            y = softmax(y_raw / temperature)

            next_idx = np.random.choice(self.vocab_size, p=y)
            next_char = idx2char[next_idx]

            if next_char == END_TOKEN:
                break
            if next_char != START_TOKEN:
                name.append(next_char)
            input_idx = next_idx

        return "".join(name).capitalize() if name else ""


# SECTION 5: RNN WITH ATTENTION MECHANISM

class RNNWithAttention:

    def __init__(self, vocab_size, hidden_size=64, attn_size=32, lr=0.01):
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.attn_size   = attn_size
        self.n_layers    = 1
        self.lr          = lr

        # Encoder RNN weights 
        scale_xh = np.sqrt(1.0 / vocab_size)
        scale_hh = np.sqrt(1.0 / hidden_size)
        self.W_xh = np.random.randn(hidden_size, vocab_size)  * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.b_h  = np.zeros(hidden_size)

        # Attention weights
        # W_a projects hidden state to attention space
        # v is the scoring vector
        scale_a = np.sqrt(1.0 / hidden_size)
        self.W_a = np.random.randn(attn_size, hidden_size) * scale_a
        self.b_a = np.zeros(attn_size)
        self.v   = np.random.randn(attn_size) * scale_a

        # Output layer (takes h_t + context = 2*hidden_size)
        combined_size = 2 * hidden_size
        scale_hy = np.sqrt(1.0 / combined_size)
        self.W_hy = np.random.randn(vocab_size, combined_size) * scale_hy
        self.b_y  = np.zeros(vocab_size)

    @property
    def n_params(self):
        """Total trainable parameters."""
        return (self.W_xh.size + self.W_hh.size + self.b_h.size +
                self.W_a.size  + self.b_a.size  + self.v.size   +
                self.W_hy.size + self.b_y.size)

    def encode(self, input_seq):
        
        T = len(input_seq)
        h = np.zeros(self.hidden_size)
        h_states = [h.copy()]

        for t in range(T):
            x_t = one_hot(input_seq[t], self.vocab_size)
            h_raw = self.W_xh @ x_t + self.W_hh @ h + self.b_h
            h = tanh(h_raw)
            h_states.append(h.copy())

        return h_states                                     # length T+1

    def attend(self, query_h, all_h_states):
        
        H_all = np.stack(all_h_states, axis=0)             # [T, hidden_size]
        T = H_all.shape[0]

        # Compute attention scores: e_t = v^T tanh(W_a h_t + b_a)
        # Note: in this formulation attention is over encoder states
        proj = np.tanh(H_all @ self.W_a.T + self.b_a)     # [T, attn_size]
        scores = proj @ self.v                              # [T]

        # Normalize to attention weights
        alpha = softmax(scores)                             # [T]

        # Context vector: weighted sum of encoder hidden states
        context = alpha @ H_all                            # [hidden_size]

        return context, alpha

    def forward(self, input_seq):
        
        T = len(input_seq)
        h_states = self.encode(input_seq)                  # length T+1

        outputs  = []
        contexts = []
        alphas   = []

        for t in range(T):
            h_t = h_states[t + 1]                         # Current hidden state

            # Attend over all encoder states (excluding padding h_0)
            ctx, alpha = self.attend(h_t, h_states[1:])   # Use h_1 to h_T

            # Combined representation
            combined = np.concatenate([h_t, ctx])          # [2*hidden_size]

            # Output projection
            y_raw = self.W_hy @ combined + self.b_y
            y = softmax(y_raw)

            outputs.append(y)
            contexts.append(ctx)
            alphas.append(alpha)

        return h_states, outputs, contexts, alphas

    def train_step(self, input_seq, target_seq):
        
        h_states, outputs, contexts, alphas = self.forward(input_seq)
        T = len(input_seq)

        # Initialize gradient buffers
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y  = np.zeros_like(self.b_y)
        dW_a  = np.zeros_like(self.W_a)
        db_a  = np.zeros_like(self.b_a)
        dv    = np.zeros_like(self.v)

        total_loss = 0.0
        dh_next = np.zeros(self.hidden_size)

        H_all = np.stack(h_states[1:], axis=0)            # [T, hidden_size]

        for t in reversed(range(T)):
            y      = outputs[t]
            target = target_seq[t]
            total_loss += cross_entropy_loss(y, target)

            h_t = h_states[t + 1]
            h_prev = h_states[t]
            ctx = contexts[t]
            alpha = alphas[t]
            x_t = one_hot(input_seq[t], self.vocab_size)

            # Output gradient
            dy = y.copy()
            dy[target] -= 1.0

            # Combined gradient
            combined = np.concatenate([h_t, ctx])
            dW_hy += np.outer(dy, combined)
            db_y  += dy
            dcombined = self.W_hy.T @ dy

            # Split combined gradient
            dh_from_out = dcombined[:self.hidden_size]
            dctx = dcombined[self.hidden_size:]

            # Context gradient: context = alpha @ H_all
            # d_alpha = H_all @ dctx  [T]
            # d_H_all += outer(alpha, dctx)
            dH_from_ctx = np.outer(alpha, dctx)            # [T, hidden]

            # Approximate attention weight gradient (simplified)
            # (Full gradient requires backprop through softmax over scores)
            proj = np.tanh(H_all @ self.W_a.T + self.b_a) # [T, attn_size]
            d_scores = alpha * (1.0 - alpha) * (H_all @ dctx)
            d_proj = np.outer(d_scores, self.v)            # [T, attn_size] approx
            d_proj_through_tanh = d_proj * (1.0 - proj**2)
            dW_a += d_proj_through_tanh.T @ H_all
            db_a += d_proj_through_tanh.sum(axis=0)
            dv   += (proj * np.outer(d_scores, np.ones(self.attn_size))).sum(axis=0)

            # Hidden state gradient
            dh = dh_from_out + dH_from_ctx[t] + dh_next
            dh_raw = dh * (1.0 - h_t ** 2)                # Through tanh

            dW_xh += np.outer(dh_raw, x_t)
            dW_hh += np.outer(dh_raw, h_prev)
            db_h  += dh_raw
            dh_next = self.W_hh.T @ dh_raw

        # Clip and update
        all_grads = [dW_xh, dW_hh, db_h, dW_hy, db_y, dW_a, db_a, dv]
        for g in all_grads:
            np.clip(g, -5.0, 5.0, out=g)

        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.b_h  -= self.lr * db_h
        self.W_hy -= self.lr * dW_hy
        self.b_y  -= self.lr * db_y
        self.W_a  -= self.lr * dW_a
        self.b_a  -= self.lr * db_a
        self.v    -= self.lr * dv

        return total_loss / T

    def generate(self, char2idx, idx2char, max_len=20, temperature=0.8):
        
        h = np.zeros(self.hidden_size)
        input_idx = char2idx[START_TOKEN]
        name = []
        h_history = [h.copy()]                             # Track for attention

        for _ in range(max_len):
            x_t = one_hot(input_idx, self.vocab_size)
            h_raw = self.W_xh @ x_t + self.W_hh @ h + self.b_h
            h = tanh(h_raw)
            h_history.append(h.copy())

            # Attend over history of hidden states
            ctx, _ = self.attend(h, h_history)

            combined = np.concatenate([h, ctx])
            y_raw = self.W_hy @ combined + self.b_y
            y = softmax(y_raw / temperature)

            next_idx = np.random.choice(self.vocab_size, p=y)
            next_char = idx2char[next_idx]

            if next_char == END_TOKEN:
                break
            if next_char != START_TOKEN:
                name.append(next_char)
            input_idx = next_idx

        return "".join(name).capitalize() if name else ""


# SECTION 6: TRAINING LOOP

def train_model(model, names, char2idx, idx2char, n_epochs=30, model_name="Model"):
    
    print(f"\n{'═'*60}")
    print(f"  Training: {model_name}")
    print(f"  Architecture: {type(model).__name__}")
    print(f"  Trainable parameters: {model.n_params:,}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Layers: {getattr(model, 'n_layers', 1)}", end="")
    if hasattr(model, "n_directions"):
        print(f" | Directions: {model.n_directions}")
    else:
        print()
    print(f"  Vocabulary size: {model.vocab_size}")
    print(f"  Learning rate: {model.lr}")
    print(f"  Epochs: {n_epochs} | Training names: {len(names)}")
    print(f"{'═'*60}")

    epoch_losses = []
    t0 = time.time()

    for epoch in range(n_epochs):
        random.shuffle(names)                              # Shuffle each epoch
        epoch_loss = 0.0
        n_steps    = 0

        for name in names:
            if len(name) < 2:
                continue
            input_seq, target_seq = name_to_sequences(name, char2idx)
            loss = model.train_step(input_seq, target_seq)
            epoch_loss += loss
            n_steps    += 1

        avg_loss = epoch_loss / max(1, n_steps)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            # Sample a name to show progress
            sample = model.generate(char2idx, idx2char, temperature=0.8)
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f} | "
                  f"Time: {elapsed:.1f}s | Sample: '{sample}'")

    total_time = time.time() - t0
    print(f"\n Training complete in {total_time:.1f}s | Final loss: {epoch_losses[-1]:.4f}")
    return epoch_losses


# SECTION 7: MAIN - TRAIN ALL MODELS

if __name__ == "__main__":
    print("=" * 60)
    print("  PROBLEM 2 - TASK 1: CHARACTER-LEVEL NAME GENERATION")
    print("=" * 60)

    # Load dataset
    names = load_names("TrainingNames.txt")
    if not names:
        raise ValueError(
            "No valid names found in TrainingNames.txt after preprocessing. "
            "Check file format and loader rules."
        )
    print(f"\n  Loaded {len(names)} training tokens from TrainingNames.txt")
    print(f"  Sample names: {names[:8]}")

    # Build vocabulary
    char2idx, idx2char, vocab = build_char_vocab(names)
    vocab_size = len(vocab)
    print(f"\n  Character vocabulary ({vocab_size} chars): {' '.join(vocab)}")

    # Hyperparameters 
    EPOCHS = 30   # Sufficient for convergence on this small dataset
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Model 1: Vanilla RNN
    rnn_model = VanillaRNN(vocab_size=vocab_size, hidden_size=128, lr=0.005)
    rnn_losses = train_model(rnn_model, names, char2idx, idx2char,
                              n_epochs=EPOCHS, model_name="Vanilla RNN")

    # Model 2: BLSTM
    blstm_model = BidirectionalLSTM(vocab_size=vocab_size, hidden_size=64, lr=0.003)
    blstm_losses = train_model(blstm_model, names, char2idx, idx2char,
                                n_epochs=EPOCHS, model_name="Bidirectional LSTM")

    #  Model 3: RNN + Attention
    attn_model = RNNWithAttention(vocab_size=vocab_size, hidden_size=96,
                                   attn_size=48, lr=0.004)
    attn_losses = train_model(attn_model, names, char2idx, idx2char,
                               n_epochs=EPOCHS, model_name="RNN with Attention")

    # Save models
    def save_rnn_model(model, fname, char2idx, idx2char, losses):
        np.savez(fname,
                 W_xh=model.W_xh, W_hh=model.W_hh, b_h=model.b_h,
                 W_hy=model.W_hy, b_y=model.b_y,
                 char2idx=json.dumps(char2idx),
                 idx2char=json.dumps(idx2char),
                 losses=losses,
                 vocab_size=model.vocab_size,
                 hidden_size=model.hidden_size)

    save_rnn_model(rnn_model, os.path.join(model_dir, "rnn_model"), char2idx, idx2char, rnn_losses)
    print("\n  [SAVED] models/rnn_model.npz")

    # Save vocab for use by other tasks
    with open("char_vocab.json", "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": idx2char,
                   "vocab_size": vocab_size, "vocab": vocab}, f, indent=2)
    print("  [SAVED] char_vocab.json")

    # Save all models via pickle-like approach using npz
    # (For BLSTM and Attention models, save key weights separately)
    np.savez(os.path.join(model_dir, "blstm_model"),
             W_fwd=blstm_model.fwd_lstm.W, b_fwd=blstm_model.fwd_lstm.b,
             W_bwd=blstm_model.bwd_lstm.W, b_bwd=blstm_model.bwd_lstm.b,
             W_out=blstm_model.W_out, b_out=blstm_model.b_out,
             losses=blstm_losses,
             char2idx=json.dumps(char2idx), idx2char=json.dumps(idx2char))
    print("  [SAVED] models/blstm_model.npz")

    np.savez(os.path.join(model_dir, "attn_model"),
             W_xh=attn_model.W_xh, W_hh=attn_model.W_hh, b_h=attn_model.b_h,
             W_hy=attn_model.W_hy, b_y=attn_model.b_y,
             W_a=attn_model.W_a, b_a=attn_model.b_a, v=attn_model.v,
             losses=attn_losses,
             char2idx=json.dumps(char2idx), idx2char=json.dumps(idx2char))
    print("  [SAVED] models/attn_model.npz")

    # Summary table
    print("\n" + "═" * 60)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("═" * 60)
    print(f"  {'Model':<24} {'Params':>10} {'Hidden':>8} {'Layers':>7} {'LR':>8} {'Final Loss':>12}")
    print("  " + "-" * 80)
    print(f"  {'Vanilla RNN':<24} {rnn_model.n_params:>10,} {rnn_model.hidden_size:>8} {rnn_model.n_layers:>7} {rnn_model.lr:>8.4f} {rnn_losses[-1]:>12.4f}")
    print(f"  {'Bidirectional LSTM':<24} {blstm_model.n_params:>10,} {blstm_model.hidden_size:>8} {blstm_model.n_layers:>7} {blstm_model.lr:>8.4f} {blstm_losses[-1]:>12.4f}")
    print(f"  {'RNN + Attention':<24} {attn_model.n_params:>10,} {attn_model.hidden_size:>8} {attn_model.n_layers:>7} {attn_model.lr:>8.4f} {attn_losses[-1]:>12.4f}")

    print("\n Task 1 completed - all 3 models trained and saved.\n")
