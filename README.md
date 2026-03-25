# NLU Assignment 2 

This repository contains two independent parts:
- `problem1`: IITJ corpus curation + Word2Vec training + semantic analysis + visualization
- `problem2`: Character-level name generation with Vanilla RNN, Bidirectional LSTM, and RNN+Attention

## 1) Prerequisites

- Linux/macOS terminal
- Python 3.10+ (project currently uses Python 3.12 venv)

Recommended Python packages:
- `numpy`
- `matplotlib`
- `requests`
- `beautifulsoup4`
- `wordcloud` (optional, for better word cloud rendering)

## 2) Environment Setup

From repository root:

```bash
cd /home/aditya/NLU_ASSIGNMENT-2
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib requests beautifulsoup4 wordcloud pypdf
```

If some optional packages fail, core scripts still run with fallbacks.

## 3) Run Problem 1 (Task 0 -> Task 4)

Go to `problem1`:

```bash
cd /home/aditya/NLU_ASSIGNMENT-2/problem1
```

### Task 0: Data collection (crawler)

```bash
python task0_data_collection.py
```


Expected outputs:
- `corpus_raw.txt`
- `results/source_manifest.json`

### Task 1: Dataset preparation

```bash
python task1_dataset_preparation.py
```

Expected outputs:
- `corpus_cleaned.txt`
- `results/dataset_stats.json`
- `visualizations/task1_wordcloud.png`

### Task 2: Word2Vec model training

```bash
python task2_model_training.py
```

Expected outputs (inside `models/`):
- `cbow_d50_w2.npz`, `cbow_d50_w4.npz`, `cbow_d100_w2.npz`, `cbow_d100_w4.npz`
- `skipgram_d50_w2_neg5.npz`, `skipgram_d50_w2_neg10.npz`, `skipgram_d50_w4_neg5.npz`, `skipgram_d50_w4_neg10.npz`
- `skipgram_d100_w2_neg5.npz`, `skipgram_d100_w2_neg10.npz`, `skipgram_d100_w4_neg5.npz`, `skipgram_d100_w4_neg10.npz`

Also saves:
- `results/training_results.json`

### Task 3: Semantic analysis

```bash
python task3_semantic_analysis.py
```

Expected outputs:
- Console nearest-neighbor + analogy results
- `results/semantic_analysis_results.json`

### Task 4: Visualization

```bash
python task4_visualization.py
```

Expected outputs:
- `visualizations/word_embeddings_embeddings.png`

## 4) Run Problem 2 (Task 1 -> Task 2/3)

Go to `problem2`:

```bash
cd /home/aditya/NLU_ASSIGNMENT-2/problem2
```

### Task 1: Train all sequence models

```bash
python task1_model_implementation.py
```

Expected outputs:
- `models/rnn_model.npz`
- `models/blstm_model.npz`
- `models/attn_model.npz`
- `char_vocab.json`

### Task 2/3: Evaluate generated names

```bash
python task2_3_evaluation.py
```

Expected outputs:
- `generated_Vanilla_RNN.txt`
- `generated_Bidirectional_LSTM.txt`
- `generated_RNN_plus_Attention.txt`
- `evaluation_results.json`

## 5) Typical Execution Order (Full Assignment)

From root:

```bash
cd problem1
python task0_data_collection.py
python task1_dataset_preparation.py
python task2_model_training.py
python task3_semantic_analysis.py
python task4_visualization.py

cd ../problem2
python task1_model_implementation.py
python task2_3_evaluation.py
```
