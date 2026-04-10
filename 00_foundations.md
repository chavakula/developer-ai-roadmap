# 00 — Foundations  
**Goal:** build the minimum technical base required for training, fine-tuning, RAG, and agents  
**Case study:** OrbitMart support tickets and product descriptions  
**Updated:** 2026-04-10

---

## Why this phase matters

Many people jump straight into “fine-tuning LLMs” and then get stuck because they do not yet understand:

- how tokens become tensors
- where loss is computed
- what a batch really is
- why train/validation/test splits matter
- how model behavior is evaluated
- the difference between **causal language modeling** and **masked language modeling**

If you master this file, later topics become much easier.

---

## What you should know by the end

You should be able to:

- explain how a tokenizer turns text into IDs
- write a small PyTorch training loop
- train a simple classifier
- explain embeddings, logits, loss, and backpropagation
- explain why next-token prediction is different from masked-token prediction
- read a modern training script without feeling lost

---

## Environment setup

Use a clean Python environment.

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets evaluate accelerate scikit-learn pandas matplotlib jupyter
```

### Optional but useful
```bash
pip install wandb rich tiktoken sentencepiece
```

### Suggested folder structure
```text
orbitmart-foundations/
├── data/
│   ├── tickets.csv
│   ├── product_descriptions.txt
│   └── splits/
├── notebooks/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── eval.py
└── experiments/
```

---

## Core mental model

A modern text model pipeline is:

```text
raw text
  -> tokenizer
  -> token IDs
  -> embeddings
  -> transformer layers
  -> logits
  -> loss
  -> gradients
  -> optimizer step
```

### Important words

- **tokenizer**: converts text into token IDs
- **embedding**: dense vector representation for each token
- **logits**: raw unnormalized scores before softmax
- **loss**: number that tells you how wrong the model is
- **backpropagation**: computes gradients for learning
- **optimizer**: updates parameters based on gradients

---

## AI terminology deep-dive for beginners

This section explains every technical term you will encounter in this tutorial set.  
If anything feels confusing later, come back here.

---

### Token

A **token** is the smallest unit a model works with.  
It is **not** always a full word. It can be a word, part of a word, a single character, or a punctuation mark.

```text
Sentence:    "The USB-C charger is not working"

Word tokens:    ["The", "USB-C", "charger", "is", "not", "working"]
Subword tokens: ["The", "USB", "-", "C", "char", "ger", "is", "not", "work", "ing"]
```

**Why subword?** Because real text has millions of possible words (misspellings, jargon, names).  
A subword tokenizer handles them all by breaking rare words into known pieces.

---

### Tokenizer

A **tokenizer** is the tool that converts raw text into tokens, then into integer IDs.

```text
┌──────────────────────┐
│   "The charger is"   │   ← raw text (string)
└──────────┬───────────┘
           │
    ┌──────▼──────┐
    │  Tokenizer  │
    └──────┬──────┘
           │
┌──────────▼────────────────┐
│ ["The", "char", "ger",    │   ← tokens (strings)
│  "is"]                    │
└──────────┬────────────────┘
           │
┌──────────▼────────────────┐
│ [464, 1149, 1362, 318]    │   ← token IDs (integers)
└───────────────────────────┘
```

The model never sees text — it only sees these integer IDs.

#### Quick example
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The USB-C charger is not working"
tokens = tokenizer.tokenize(text)
ids    = tokenizer.encode(text)

print("tokens:", tokens)
# ['The', ' USB', '-', 'C', ' charger', ' is', ' not', ' working']

print("ids:", ids)
# [464, 10448, 12, 34, 24530, 318, 407, 1762]
```

---

### Vocabulary (vocab)

The **vocabulary** is the complete list of tokens a tokenizer knows.  
Each token is assigned a unique integer ID.

Think of it as a lookup table:

```text
┌───────────┬────┐
│  Token    │ ID │
├───────────┼────┤
│ <pad>     │  0 │
│ <unk>     │  1 │
│ the       │  2 │
│ charger   │  3 │
│ is        │  4 │
│ ...       │... │
│ USB-C     │9842│
└───────────┴────┘
```

**Vocab size** = how many unique tokens the tokenizer knows.  
GPT-2 has ~50,257 tokens. Llama 3 has 128,256.

---

### Special tokens

Tokens with a reserved meaning that never come from user text:

| Token | Purpose | Example |
|---|---|---|
| `<pad>` | fills short sequences to same length | `[52, 18, 0, 0, 0]` |
| `<unk>` | replaces words not known to the vocab | `rare_word → <unk>` |
| `<bos>` | marks the **beginning** of a sequence | |
| `<eos>` | marks the **end** of a sequence | |
| `[MASK]` | placeholder for masked language models | |

---

### Tensor

A **tensor** is a multi-dimensional array of numbers — the data structure everything runs on in PyTorch.

```text
Scalar (0-D tensor):        42
Vector (1-D tensor):        [1, 2, 3]
Matrix (2-D tensor):        [[1, 2], [3, 4]]
3-D tensor (batch of data): [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

When you hear "tokens become tensors," it means the list of integer IDs gets wrapped into a PyTorch tensor so that GPU math can run on it efficiently.

```python
import torch
ids = [464, 10448, 12, 34]
tensor = torch.tensor(ids, dtype=torch.long)
# tensor([464, 10448, 12, 34])
```

---

### Embedding

An **embedding** turns each token ID into a dense vector of numbers (e.g., 128 or 768 numbers).  
This is where the model starts learning what tokens mean.

```text
Token ID: 464 ("The")
                    ┌─────────────────────────────────┐
   Embedding Layer  │  Look up row 464 in a big table │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
   Embedding Vector │ [0.12, -0.34, 0.56, ..., 0.08] │  ← 128 numbers
                    └─────────────────────────────────┘
```

Why not just use the integer ID?  
Because **ID 464** and **ID 465** are not inherently similar.  
Embeddings give the model a learnable, continuous space where meaning can emerge.

#### Example
```python
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=50257, embedding_dim=128)
# This creates a table with 50,257 rows and 128 columns.

token_ids = torch.tensor([464, 10448, 12])   # 3 tokens
vectors   = embedding(token_ids)              # shape: [3, 128]
print(vectors.shape)
# torch.Size([3, 128])
```

---

### Parameters (weights)

**Parameters** are the learnable numbers inside a model.  
Every embedding value, every connection weight — those are parameters.

```text
┌──────────────────────────────┐
│   Embedding table            │
│   50,257 tokens × 128 dims  │
│   = 6,432,896 parameters    │
├──────────────────────────────┤
│   Classifier layer           │
│   128 × 6 = 768 parameters  │
├──────────────────────────────┤
│   Total: ~6.4M parameters   │
└──────────────────────────────┘
```

When people say a model has "7 billion parameters," they mean 7 billion learnable numbers.

---

### Forward pass

The **forward pass** is when data flows through the model to produce a prediction.

```text
 Input IDs ──► Embedding ──► Model Layers ──► Output (logits)
                 ↓                ↓                ↓
           [464, 318]    [[0.1, -0.3, ...],   [2.1, -0.5, 0.3,
                          [0.5,  0.2, ...]]    1.8, -1.2, 0.9]
```

No learning happens during the forward pass — it only produces a prediction.

---

### Logits

**Logits** are the raw, unnormalized output numbers from the model's last layer.

```text
Model predicts ticket intent:

  logits = [2.1, -0.5, 0.3, 1.8, -1.2, 0.9]
             ↓     ↓    ↓    ↓     ↓    ↓
         order  return tech billing addr  other
         status request issue issue  chg
```

Higher logit = model thinks that class is more likely.  
But logits are **not probabilities** yet — they can be any number.

---

### Softmax

**Softmax** converts logits into probabilities that add up to 1.

```text
  logits:        [2.1,  -0.5,  0.3,  1.8,  -1.2,  0.9]
                   ↓ softmax ↓
  probabilities: [0.42,  0.03, 0.07, 0.31,  0.02,  0.13]
                   ↓
  prediction:    order_status (highest probability = 0.42)
```

```python
import torch
logits = torch.tensor([2.1, -0.5, 0.3, 1.8, -1.2, 0.9])
probs = torch.softmax(logits, dim=0)
print(probs)
# tensor([0.4243, 0.0316, 0.0703, 0.3146, 0.0157, 0.1280])
print(probs.sum())   # 1.0
```

---

### Loss function

The **loss** is a single number that measures how wrong the model's prediction is.  
**Lower loss = better model.**

```text
True label:     "order_status"   (index 0)
Model logits:   [2.1, -0.5, 0.3, 1.8, -1.2, 0.9]

                    ┌──────────────────┐
                    │ CrossEntropyLoss │
                    └────────┬─────────┘
                             │
                        loss = 0.86
```

#### Two loss functions you will use constantly

**1. Cross-entropy loss** — for classification tasks
```python
criterion = torch.nn.CrossEntropyLoss()
logits = torch.tensor([[2.1, -0.5, 0.3, 1.8, -1.2, 0.9]])   # [1, 6]
true_label = torch.tensor([0])   # correct class = order_status
loss = criterion(logits, true_label)
print(loss)   # 0.8572
```

**2. Per-token cross-entropy** — for language modeling
```python
# Model output: one distribution per position
logits = torch.randn(1, 5, 50257)   # [batch=1, seq_len=5, vocab_size=50257]
targets = torch.tensor([[101, 205, 33, 78, 910]])   # next tokens

loss = nn.CrossEntropyLoss()(
    logits.view(-1, 50257),   # flatten to [5, 50257]
    targets.view(-1)          # flatten to [5]
)
```

---

### Backpropagation (backward pass)

**Backpropagation** is how the model figures out which parameters caused the error.

```text
Forward pass (left to right):
  Input ──► Embedding ──► Layers ──► Logits ──► Loss = 0.86

Backward pass (right to left):
  Loss ◄── Logits ◄── Layers ◄── Embedding ◄── Input
    compute gradient for every parameter
```

After backpropagation, each parameter has a **gradient** — a number that says:
- "If this parameter were slightly larger, would the loss go up or down?"
- "And by how much?"

```python
loss.backward()   # computes all gradients
# Now model.embedding.weight.grad has gradient values
```

---

### Gradient

A **gradient** tells you the direction and magnitude of change needed to reduce the loss.

```text
Parameter value: 0.45
Gradient:       -0.03
Meaning:        "increasing this parameter slightly would decrease loss"
```

Gradients are calculated by backpropagation, then used by the optimizer.

---

### Optimizer

The **optimizer** updates parameters using the gradients to reduce loss.

```text
Before step:  parameter = 0.45, gradient = -0.03, learning_rate = 0.001
After step:   parameter = 0.45 - (0.001 × -0.03) = 0.45003
```

```text
┌──────────────┐    ┌────────────────┐    ┌──────────────────┐
│  Gradients   │ ──►│   Optimizer    │ ──►│ Updated weights  │
│  (from       │    │  (AdamW, SGD)  │    │ (slightly better)│
│  backward)   │    └────────────────┘    └──────────────────┘
└──────────────┘
```

**AdamW** is the default optimizer for transformer models.

---

### Learning rate

The **learning rate** controls how big each parameter update step is.

```text
Too high (lr=0.1):    ✗ model jumps erratically, loss explodes
Just right (lr=3e-4): ✓ model converges steadily
Too low (lr=1e-8):    ✗ model barely changes, training takes forever  
```

Typical learning rates for transformer fine-tuning: `1e-5` to `3e-4`.

---

### Epoch

One **epoch** = one complete pass through all training data.

```text
Dataset: 1000 examples
Batch size: 32

1 epoch = ceil(1000/32) = 32 batches (iterations)
5 epochs = 5 × 32 = 160 total iterations
```

---

### Batch and batch size

A **batch** is a group of examples processed together.

```text
Dataset: 1000 tickets

Batch size = 32 means:
  Batch 1: tickets 1–32    ──► forward pass ──► loss ──► backward ──► update
  Batch 2: tickets 33–64   ──► forward pass ──► loss ──► backward ──► update
  ...
  Batch 32: tickets 969–1000
```

Why batches, not one-at-a-time?
- **GPU parallelism**: GPUs process many examples at once efficiently
- **Stable gradients**: averaging over a batch reduces noise
- **Memory constraints**: the full dataset won't fit in GPU memory at once

---

### Overfitting

The model memorizes the training data instead of learning general patterns.

```text
                Training loss         Validation loss
Epoch 1         2.10                  2.15            (both improving)
Epoch 5         0.30                  0.45            (gap growing)
Epoch 20        0.01                  1.80            (overfitting!)
                 ↑                      ↑
          memorized training       failing on new data
```

Signs:
- training loss keeps dropping
- validation loss starts rising
- model gets test examples it saw during training right, but fails on new ones

Prevention:
- more diverse training data
- early stopping (stop when val loss stops improving)
- dropout (randomly ignoring some parameters during training)
- regularization (weight decay, etc.)

---

### Dropout

**Dropout** randomly sets some neuron outputs to zero during training.  
This prevents the model from depending too heavily on any single feature.

```text
Without dropout:  [0.5, 0.3, 0.8, 0.1, 0.6]   all active
With dropout=0.2: [0.5, 0.0, 0.8, 0.1, 0.0]   ~20% randomly zeroed

At inference (eval): dropout is turned OFF, all values are used.
```

That is why you call `model.train()` and `model.eval()` — they toggle dropout behavior.

---

### model.train() vs model.eval()

```text
model.train():
  - dropout is ON (randomly drops connections)
  - batch normalization uses batch statistics
  - used during: training loop

model.eval():
  - dropout is OFF (all connections active)
  - batch normalization uses running statistics
  - used during: validation and inference

torch.no_grad():
  - additionally tells PyTorch "don't track gradients"
  - saves memory during evaluation
```

---

### Self-attention

**Self-attention** is the core mechanism of transformers.  
For each token, it looks at **every other token** in the sequence and decides how much to "pay attention" to each one.

```text
Sentence: "The charger is not working"

When processing "working":
  ┌─────────────────────────────────────────────┐
  │  Attention weights (how much to look at):   │
  │                                             │
  │  "The"      →  0.05  (low — not very       │
  │  "charger"  →  0.35  (high — what's not    │
  │  "is"       →  0.10      working?)          │
  │  "not"      →  0.40  (highest — negation   │
  │  "working"  →  0.10      matters here)      │
  └─────────────────────────────────────────────┘
```

This lets the model learn:
- "not working" → the negation is important
- "charger" → what specifically is broken

---

### Causal mask

In **causal (autoregressive) models**, each token can only attend to itself and previous tokens, not future ones.

```text
Tokens:     The   charger   is   not   working

The         ✓      ✗        ✗    ✗     ✗
charger     ✓      ✓        ✗    ✗     ✗
is          ✓      ✓        ✓    ✗     ✗
not         ✓      ✓        ✓    ✓     ✗
working     ✓      ✓        ✓    ✓     ✓

✓ = can see     ✗ = masked (hidden)
```

This forces the model to predict each token using only past context — exactly what you want for text generation.

---

### Encoder vs decoder

```text
┌─────────────────────────────────────────────────────┐
│  ENCODER (BERT-style)                               │
│                                                     │
│  Reads entire sequence at once (bidirectional)      │
│  Good for: classification, NER, understanding       │
│  Uses: masked language modeling                     │
│  Example: BERT, RoBERTa, DeBERTa                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  DECODER (GPT-style)                                │
│                                                     │
│  Reads left-to-right only (causal mask)             │
│  Good for: text generation, chat, code generation   │
│  Uses: next-token prediction                        │
│  Example: GPT-2/3/4/5, Llama, Mistral              │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  ENCODER-DECODER (T5-style)                         │
│                                                     │
│  Encoder reads the input bidirectionally             │
│  Decoder generates output left-to-right             │
│  Good for: translation, summarization               │
│  Example: T5, BART, mBART                           │
└─────────────────────────────────────────────────────┘
```

Most modern LLMs (GPT-5, Llama, Mistral, etc.) are **decoder-only**.

---

### LLM (Large Language Model)

An **LLM** is a language model with billions of parameters, pretrained on a massive corpus of text.

- "Large" means: billions of parameters (7B, 70B, 400B+)
- "Language" means: trained on text (books, web, code, etc.)
- "Model" means: a function that takes tokens in and produces predictions

Examples: GPT-5.4, Llama 3, Mistral, Claude.

LLMs are trained on **next-token prediction** over internet-scale data, giving them broad language understanding.

---

### Pretrained model

A model that has already been trained on a large corpus.

```text
 ────────────────────────────────────────────────
 Step 1: Pretraining (done by large lab)
   Train on billions of tokens of text
   Cost: millions of dollars, thousands of GPUs
   Result: a general-purpose language model
 ────────────────────────────────────────────────
 Step 2: Fine-tuning (done by you)
   Train on YOUR specific data (hundreds to thousands of examples)
   Cost: hours on a single GPU
   Result: a model adapted to YOUR task
 ────────────────────────────────────────────────
```

You almost never train from scratch. You start from a pretrained model and adapt it.

---

### Fine-tuning

**Fine-tuning** = taking a pretrained model and continuing training on your specific data.

```text
Pretrained model (knows general language)
        │
        ▼  fine-tune on OrbitMart support data
        │
OrbitMart support model (knows your tone, policies, formats)
```

Covered in depth in [01_fine_tuning_models.md](./01_fine_tuning_models.md).

---

### RAG (Retrieval-Augmented Generation)

Instead of changing model weights, you **fetch relevant documents at runtime** and include them in the prompt.

```text
User question: "What is OrbitMart's return policy for earbuds?"

Without RAG:
  Model guesses from memory → might hallucinate

With RAG:
  1. Search knowledge base → find policy document
  2. Insert policy text into prompt → model reads it
  3. Model answers based on evidence → grounded answer
```

Covered in depth in [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md).

---

### Inference

**Inference** = using a trained model to make predictions (no learning happens).

```text
Training:   model learns from data     (forward + backward + optimizer)
Inference:  model produces predictions  (forward only)
```

---

### Perplexity

**Perplexity** measures how surprised a language model is by a text.

```text
Perplexity = 10   → model predicted the text well (low surprise)
Perplexity = 500  → model was very confused by the text (high surprise)
```

Lower perplexity = better language model performance.  
Mathematically: perplexity = $e^{\text{average cross-entropy loss per token}}$

---

### Accuracy, precision, recall, F1

These four metrics measure classification quality.

```text
Imagine 100 tickets classified as "billing_issue":

              Actually billing    Actually not billing
Predicted     ┌──────────────┬──────────────────────┐
billing       │  80 (TP)     │  10 (FP)             │
              ├──────────────┼──────────────────────┤
Predicted     │   5 (FN)     │   5 (TN)             │
not billing   └──────────────┴──────────────────────┘

TP = True Positive    FP = False Positive
FN = False Negative   TN = True Negative
```

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | (TP+TN) / total | overall correctness |
| **Precision** | TP / (TP+FP) | "of those I said are billing, how many really are?" |
| **Recall** | TP / (TP+FN) | "of all actual billing tickets, how many did I catch?" |
| **F1** | 2 × (P×R)/(P+R) | harmonic mean of precision and recall |

**Macro F1** averages F1 across all classes equally — important when some classes are rare.

---

### Confusion matrix

A table showing which classes get confused with each other.

```text
Predicted →    order   return  tech   billing  addr   other
Actual ↓
order_status    45       2      1       3       0       1
return_request   1      28      0       0       0       1
technical_issue  0       0     22       0       0       2
billing_issue    4       0      0      18       0       1
address_change   1       0      0       0      14       0
other            2       1      1       1       0      10
```

Reading: row = truth, column = prediction.  
Diagonal = correct. Off-diagonal = errors.

The most useful thing: spot which pairs are confused most (billing vs order_status above).

---

### Gradient clipping

Prevents exploding gradients by capping them to a maximum value.

```text
Without clipping:  gradient = 847.3    → huge update → training explodes
With clipping:     gradient = 1.0      → safe update → training stable
```

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### Weight decay

A regularization technique that slightly shrinks weights each step to prevent overfitting.

```text
Without weight decay: parameters can grow very large → overfitting
With weight decay:    parameters are gently pulled toward zero → simpler model
```

AdamW has weight decay built in: `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)`.

---

### GPU, CUDA, CPU

```text
CPU:  General-purpose processor. Good for data loading, logic.
      Slow for matrix math on millions of numbers.

GPU:  Specialized processor. Massively parallel.
      Fast at the matrix operations that dominate deep learning.

CUDA: NVIDIA's toolkit that lets PyTorch run computations on NVIDIA GPUs.
```

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)                     # move model to GPU
tensor = tensor.to(device)           # move data to GPU
```

If you have no GPU, everything still works on CPU — just slower.

---

### End-to-end diagram: the full training loop

```text
┌─────────────────────────────────────────────────────────────────┐
│                    ONE TRAINING STEP                             │
│                                                                 │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Batch   │──►│  Forward  │──►│   Loss   │──►│ Backward │  │
│  │  of data │   │   pass    │   │ function │   │   pass   │  │
│  └──────────┘   └───────────┘   └──────────┘   └────┬─────┘  │
│                                                      │         │
│                                              ┌───────▼───────┐ │
│                                              │   Optimizer   │ │
│                                              │   .step()     │ │
│                                              └───────────────┘ │
│                                                                 │
│  This repeats for every batch in the training set.              │
│  One full pass = one epoch.                                     │
│  Typical training: 3–20 epochs.                                 │
└─────────────────────────────────────────────────────────────────┘

After each epoch:
  ┌─────────────────────────────────────────────────────────┐
  │  Validation loop (no gradients, no optimizer)            │
  │  → compute val loss and metrics                          │
  │  → check if model improved or is overfitting             │
  └─────────────────────────────────────────────────────────┘
```

---

## The two language-modeling objectives you must know

These are the two fundamental ways language models learn from text.  
Understanding them is required before you can make sense of fine-tuning, training, or RAG.

### 1. Causal language modeling (next-token prediction)
The model predicts the **next token** using only the tokens on the left.

```text
Step-by-step:

Input text:       "The order arrived on"
After tokenizing: [The, order, arrived, on]

The model sees:   [The, order, arrived, on, ???]
It must predict:  "time"

       The ──► order ──► arrived ──► on ──► ???
                                             │
                                        prediction: "time"

Key idea: the model CANNOT see future tokens.
          It reads left to right, one token at a time.
```

Used for:
- chat assistants
- code generation
- text generation
- many decoder-only LLMs

### 2. Masked language modeling (fill-in-the-blank)
The model predicts **masked tokens** using both left and right context.

```text
Step-by-step:

Original text:    "The order arrived on time"
After masking:    "The order arrived [MASK] time"

The model sees:   [The, order, arrived, [MASK], time]
It must predict:  "on"

  The ──► order ──► arrived ──► [MASK] ◄── time
                                  │
                            prediction: "on"

Key idea: the model CAN see tokens on BOTH sides.
          It uses full context to fill in the blank.
```

Used for:
- contextual understanding
- classification/representation tasks
- many encoder-style models like BERT

### Side-by-side comparison

```text
┌────────────────────────────────────────────────────────────┐
│         CAUSAL LM                MASKED LM                 │
│                                                            │
│  Input:  "The order arrived"     "The order [MASK] on"     │
│  Target: "on"                    "arrived"                  │
│                                                            │
│  Context: left only ──►          ◄── both sides ──►        │
│  Output:  generates text         understands meaning        │
│  Models:  GPT, Llama, Mistral    BERT, RoBERTa, DeBERTa   │
└────────────────────────────────────────────────────────────┘
```

### Rule of thumb
- want the model to **generate**? think causal LM
- want the model to **understand/classify/extract**? masked LM style models often fit well

---

## Foundations project 1 — Support ticket intent classifier

### Real-world use case
OrbitMart receives customer emails like:

- “Where is my package?”
- “I want to return this speaker.”
- “The charger is not working.”
- “Can I change the shipping address?”
- “Why was my card charged twice?”

The business wants to classify each ticket into one label so it can be routed quickly.

### Example labels
- `order_status`
- `return_request`
- `technical_issue`
- `address_change`
- `billing_issue`
- `other`

### Example dataset
```csv
text,label
"Where is my delivery? It said shipped 3 days ago.",order_status
"I want to return my headphones.",return_request
"My laptop dock is not detected by Windows.",technical_issue
"Can you change the shipping address before dispatch?",address_change
"I was charged twice for order #88421.",billing_issue
```

---

## Step 1 — Prepare data splits

Always split data before training.

### Good baseline split
- **70%** training
- **15%** validation
- **15%** test

### Why this matters
- training set teaches the model
- validation set helps tune choices
- test set measures final performance

### Common beginner mistake
Looking at test performance repeatedly while tuning.  
That quietly turns the test set into part of training.

### Code: split your CSV into train / val / test

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/tickets.csv")   # columns: text, label

# First split: train vs temp (val + test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]   # keeps class proportions equal in each split
)

# Second split: val vs test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

train_df.to_csv("data/splits/train.csv", index=False)
val_df.to_csv("data/splits/val.csv", index=False)
test_df.to_csv("data/splits/test.csv", index=False)
```

> `stratify=df["label"]` ensures each label appears proportionally in every split.  
> This is important for rare classes like `billing_issue`.

---

## Step 2 — Build a minimal tokenizer

For foundations, do not overcomplicate.  
You can start with a tiny whitespace tokenizer and later replace it with a proper subword tokenizer.

### What a tokenizer does — step by step

```text
Input: "My laptop dock is not detected"

Step 1 — Split into tokens:
  ["my", "laptop", "dock", "is", "not", "detected"]

Step 2 — Look up each token in the vocabulary:
  my → 42,  laptop → 108,  dock → 215,  is → 4,  not → 17,  detected → 89

Step 3 — What if a word is unknown?
  "flibbergibbet" → not in vocab → replaced with <unk> → ID 1

Step 4 — What if the sentence is too short?
  Pad with <pad> tokens to reach max_len:
  [42, 108, 215, 4, 17, 89, 0, 0, 0, 0]   ← padded to length 10
```

### Why padding matters

```text
Batch of 3 sentences (different lengths):

"My dock"           → [42, 215]
"Not working"       → [17, 389]
"Where is my order" → [56, 4, 42, 108]

Problem: GPUs need rectangular tensors (same length per row).

After padding to max_len=4:
[42, 215,  0,   0 ]
[17, 389,  0,   0 ]
[56,   4, 42, 108]   ← now all rows have length 4
```

```python
from collections import Counter

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode(text, vocab, max_len=32):
    tokens = text.lower().split()
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids
```

### Complete working example
```python
# Build vocab from training text
texts = [
    "Where is my delivery",
    "I want to return my headphones",
    "My laptop dock is not detected",
]
vocab = build_vocab(texts)
print(vocab)
# {'<pad>': 0, '<unk>': 1, 'where': 2, 'is': 3, 'my': 4, 'delivery': 5,
#  'i': 6, 'want': 7, 'to': 8, 'return': 9, 'headphones': 10,
#  'laptop': 11, 'dock': 12, 'not': 13, 'detected': 14}

# Encode a sentence
ids = encode("Where is my delivery", vocab, max_len=8)
print(ids)
# [2, 3, 4, 5, 0, 0, 0, 0]   ← 4 real tokens + 4 padding

# Unknown word
ids = encode("my smartwatch broke", vocab, max_len=8)
print(ids)
# [4, 1, 1, 0, 0, 0, 0, 0]   ← "smartwatch" and "broke" → <unk> (ID 1)
```

### What this teaches you
- padding
- unknown tokens
- fixed-length batches
- the idea that text becomes integers before learning starts

---

## Step 3 — Create a PyTorch dataset

```python
import torch
from torch.utils.data import Dataset

class TicketDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id, max_len=32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.vocab, self.max_len)
        y = self.label2id[self.labels[idx]]
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.long)
        }
```

---

## Step 4 — Build a simple text classifier

This model is intentionally simple:

- embedding layer
- average pooling over tokens
- linear layer for class prediction

### Architecture diagram

```text
Input:  "Where is my delivery?"
         │
         ▼
┌──────────────────────────────┐
│  Token IDs: [2, 3, 4, 5]    │
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│  Embedding Layer              │
│  Each ID → 128-dim vector     │
│                               │
│  [2]→[0.12, -0.34, ..., 0.56]│
│  [3]→[0.45,  0.11, ..., 0.23]│
│  [4]→[0.88, -0.09, ..., 0.17]│
│  [5]→[0.33,  0.67, ..., 0.41]│
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│  Average Pooling              │
│  Average all 4 vectors into 1 │
│  → [0.445, 0.0875, ..., 0.34]│
│  Shape: [128]                 │
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│  Linear Classifier            │
│  128 → 6 (one score per class)│
│  → [2.1, -0.5, 0.3, 1.8,    │
│     -1.2, 0.9]               │
│  These are LOGITS             │
└──────────┬───────────────────┘
           │
           ▼
  Prediction: class 0 (order_status)
  (because 2.1 is the highest logit)
```

```python
import torch
import torch.nn as nn

class AvgEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)                   # [B, T, D]
        mask = (input_ids != 0).unsqueeze(-1)          # [B, T, 1]
        x = x * mask
        lengths = mask.sum(dim=1).clamp(min=1)         # [B, 1]
        pooled = x.sum(dim=1) / lengths                # [B, D]
        logits = self.classifier(pooled)               # [B, C]
        return logits
```

### Understanding the tensor shapes (B, T, D, C)

These letters appear everywhere in ML code. Here is what they mean:

```text
B = Batch size      (how many examples at once, e.g. 32)
T = Sequence length  (how many tokens per example, e.g. 32)
D = Embedding dim    (how many numbers per token vector, e.g. 128)
C = Number of classes (how many labels, e.g. 6)

Shape flow:
  input_ids:           [B, T]        = [32, 32]       (32 tickets, 32 tokens each)
  after embedding:     [B, T, D]     = [32, 32, 128]  (each token → 128-dim vector)
  after avg pooling:   [B, D]        = [32, 128]      (one vector per ticket)
  after classifier:    [B, C]        = [32, 6]         (one score per class per ticket)
```

### Why this is a good first model
It lets you understand:
- embeddings
- batching
- pooling
- logits
- classification loss

without hiding the learning process behind too much framework magic.

---

## Step 5 — Train it

### What happens during training — visual walkthrough

```text
Epoch 1, Batch 1:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. optimizer.zero_grad()                                │
  │    Clear leftover gradients from previous step          │
  │                                                         │
  │ 2. logits = model(input_ids)          FORWARD PASS      │
  │    Data flows through model → produces predictions      │
  │                                                         │
  │ 3. loss = criterion(logits, labels)   COMPUTE LOSS      │
  │    Compare predictions to truth → single number (0.86)  │
  │                                                         │
  │ 4. loss.backward()                    BACKWARD PASS     │
  │    Compute gradients for all parameters                 │
  │                                                         │
  │ 5. optimizer.step()                   UPDATE WEIGHTS    │
  │    Use gradients to nudge parameters toward better fit  │
  └─────────────────────────────────────────────────────────┘
  
  Then move to Batch 2, Batch 3, ... until all batches done = 1 epoch
```

```python
import torch
from torch.utils.data import DataLoader

model = AvgEmbeddingClassifier(vocab_size=len(vocab), embed_dim=128, num_labels=len(label2id))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(5):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch={epoch+1} train_loss={total_loss/len(train_loader):.4f}")
```

### What is happening in each step
- forward pass produces logits
- loss compares logits to true labels
- `loss.backward()` computes gradients
- `optimizer.step()` updates weights

### What good training looks like

```text
epoch=1 train_loss=1.7918    ← high loss (model is random)
epoch=2 train_loss=1.2003    ← learning
epoch=3 train_loss=0.7541    ← getting better
epoch=4 train_loss=0.4102    ← converging
epoch=5 train_loss=0.2389    ← low loss (model learned patterns)
```

If loss is not decreasing: check learning rate, data loading, and label correctness first.

---

## Step 6 — Evaluate it properly

Do not stop at accuracy.

### Track these
- accuracy
- macro F1
- confusion matrix
- per-class recall

### Why macro F1 matters — a concrete example

```text
Imagine 100 tickets:
  80 = order_status
  10 = return_request
   5 = billing_issue
   5 = technical_issue

A lazy model that predicts "order_status" for EVERYTHING gets:
  accuracy = 80/100 = 80%  ← looks decent!
  
But:
  billing_issue recall  = 0%   ← catches zero billing tickets  
  technical_issue recall = 0%  ← catches zero tech tickets
  macro F1 = 0.20             ← reveals the model is useless for rare classes
```

In support routing, some labels are rare.  
A model can get decent accuracy while still failing badly on minority classes.  
**Macro F1 treats all classes equally, so it exposes this problem.**

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

model.eval()
preds, gold = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids)
        batch_preds = logits.argmax(dim=-1)

        preds.extend(batch_preds.cpu().tolist())
        gold.extend(labels.cpu().tolist())

print("accuracy:", accuracy_score(gold, preds))
print("macro_f1:", f1_score(gold, preds, average="macro"))
print(classification_report(gold, preds))
```

### Reading the classification report

```text
              precision    recall  f1-score   support

order_status       0.82      0.90      0.86        50
return_request     0.88      0.78      0.82        18
technical_issue    0.91      0.83      0.87        12
billing_issue      0.75      0.60      0.67        10  ← weakest — needs more data
address_change     0.93      0.87      0.90        15
other              0.67      0.50      0.57         8  ← also weak

macro avg          0.83      0.75      0.78       113
```

- **support** = number of test examples per class
- low recall on "billing_issue" means the model misses many billing tickets
- action: collect more billing examples or improve label definitions

---

## Error analysis questions

After evaluation, read failed examples manually.

Ask:
- Did the label definition itself overlap with another label?
- Was the text too short or ambiguous?
- Was the tokenizer weak?
- Did the model confuse billing with order status because both mention order numbers?
- Are there too few examples for one class?

### Real-world lesson
Most first-model failures are **data definition failures**, not optimizer failures.

---

## Foundations project 2 — Tiny next-token predictor

### Real-world use case
OrbitMart wants draft help for product descriptions.  
You are not building a production generator yet.  
You are learning what next-token prediction means.

### Example corpus
```text
The OrbitMart AlphaDock supports dual 4K monitors over USB-C.
The OrbitMart NovaBuds offer active noise cancellation and 28-hour battery life.
The OrbitMart FlexCharge pad supports Qi2 fast charging for compatible phones.
```

---

## Minimal goal
Train a small model that sees:

```text
"The OrbitMart AlphaDock supports dual 4K"
```

and learns that likely next tokens include:

```text
"monitors"
```

---

## Data preparation idea

For a token sequence:

```text
[The, OrbitMart, AlphaDock, supports, dual, 4K, monitors]
```

Causal LM creates:

- inputs:  `[The, OrbitMart, AlphaDock, supports, dual, 4K]`
- targets: `[OrbitMart, AlphaDock, supports, dual, 4K, monitors]`

This one-token shift is the heart of next-token training.

---

## Tiny decoder-style example

This is intentionally small and educational.

```python
import torch
import torch.nn as nn

class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.rnn(x)
        logits = self.lm_head(x)
        return logits
```

This is not a transformer yet, but it teaches the objective clearly.

### Architecture diagram

```text
Input IDs: [101, 205, 33, 78]   ("The OrbitMart AlphaDock supports")
               │
    ┌──────────▼──────────┐
    │  Embedding Layer     │   Each ID → 128-dim vector
    │  [101]→[0.2, ...]   │
    │  [205]→[0.5, ...]   │   Shape: [B, T, 128]
    │  [33] →[0.1, ...]   │
    │  [78] →[0.8, ...]   │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  GRU (simple RNN)    │   Processes sequence left-to-right
    │  Keeps hidden state  │   Shape: [B, T, 128]
    │  as it reads tokens  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  LM Head (Linear)    │   128 → vocab_size scores
    │                      │   Shape: [B, T, vocab_size]
    │  At each position:   │
    │  "what token comes   │
    │   next?"             │
    └──────────┬──────────┘
               │
               ▼
  Position 0 → predicts token 205  (target: "OrbitMart")
  Position 1 → predicts token 33   (target: "AlphaDock")
  Position 2 → predicts token 78   (target: "supports")
  Position 3 → predicts token ???  (target: "dual")
```

### Loss computation
```python
logits = model(input_ids)                    # [B, T, V]
loss = nn.CrossEntropyLoss()(
    logits.view(-1, logits.size(-1)),
    labels.view(-1)
)
```

### Understanding the shape manipulation

```text
logits shape:  [B, T, V]  = [1, 4, 50257]   (1 example, 4 positions, 50257 vocab)
labels shape:  [B, T]     = [1, 4]           (1 example, 4 target token IDs)

CrossEntropyLoss needs: [N, C] and [N]
So we reshape:
  logits.view(-1, V)  → [4, 50257]   (flatten batch and time into one dimension)
  labels.view(-1)     → [4]          (flatten to match)

Now loss is computed for each position independently, then averaged.
```

### Why this matters
You now understand the shape of language-model training:
- model outputs one distribution per token position
- loss compares each position to the next true token

### Complete runnable next-token training example

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Tiny corpus ---
corpus = [
    "The OrbitMart AlphaDock supports dual 4K monitors",
    "The OrbitMart NovaBuds offer active noise cancellation",
    "The OrbitMart FlexCharge supports Qi2 fast charging",
]

# --- Simple whitespace tokenizer ---
all_words = set()
for line in corpus:
    all_words.update(line.lower().split())

word2id = {"<pad>": 0}
for w in sorted(all_words):
    word2id[w] = len(word2id)

id2word = {v: k for k, v in word2id.items()}
vocab_size = len(word2id)
print(f"Vocab size: {vocab_size}")

# --- Dataset: input = tokens[:-1], target = tokens[1:] ---
class CausalLMDataset(Dataset):
    def __init__(self, corpus, word2id, max_len=10):
        self.examples = []
        for line in corpus:
            ids = [word2id.get(w, 0) for w in line.lower().split()][:max_len]
            if len(ids) < 2:
                continue
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        return x, y

dataset = CausalLMDataset(corpus, word2id)

# --- Model ---
class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.rnn(x)
        return self.lm_head(x)

model = TinyCausalLM(vocab_size, d_model=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Train ---
for epoch in range(200):
    total_loss = 0
    for x, y in dataset:
        optimizer.zero_grad()
        logits = model(x.unsqueeze(0))           # [1, T, V]
        loss = criterion(logits.squeeze(0), y)   # [T, V] vs [T]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"epoch={epoch+1}  loss={total_loss/len(dataset):.4f}")

# --- Generate ---
def generate(model, prompt_words, max_new=5):
    model.eval()
    ids = [word2id.get(w, 0) for w in prompt_words.lower().split()]
    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        next_id = logits[0, -1].argmax().item()
        ids.append(next_id)
    return " ".join(id2word.get(i, "?") for i in ids)

print(generate(model, "the orbitmart"))
```

---

## From RNN to transformer mindset

Later, you will replace the RNN with a decoder-style transformer, but the training objective remains the same:

- tokenized sequence
- shifted targets
- per-token cross-entropy
- optimizer updates

### What changes, and what stays the same

```text
┌──────────────────────────────────────────────────────────┐
│                    STAYS THE SAME                         │
│                                                          │
│  • Input: token IDs                                      │
│  • Output: logits over vocabulary at each position       │
│  • Loss: cross-entropy between predictions and targets   │
│  • Training loop: forward → loss → backward → step       │
│  • Objective: predict the next token                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    WHAT CHANGES                           │
│                                                          │
│  RNN/GRU:                                                │
│    • Processes one token at a time, left to right        │
│    • Uses hidden state (memory cell) for context         │
│    • Struggles with very long sequences                  │
│    • Cannot be parallelized easily                       │
│                                                          │
│  Transformer:                                            │
│    • Processes ALL tokens at once (parallel)             │
│    • Uses self-attention for context                     │
│    • Handles long sequences much better                  │
│    • Runs efficiently on GPUs                            │
│    • Uses causal mask to prevent cheating                │
└──────────────────────────────────────────────────────────┘
```

---

## What a transformer changes

Compared with a simple RNN, a transformer:
- uses self-attention
- sees token relationships more directly
- scales much better
- is the basis of modern LLMs

But the learning objective can still be:
- next-token prediction for causal LMs
- masked-token prediction for encoder models

---

## Quick transformer intuition

A transformer block is the building unit of modern LLMs.  
Here is what happens inside one block, and how blocks stack.

### One transformer block

```text
┌─────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK                   │
│                                                  │
│  Input: token vectors  [B, T, D]                 │
│       │                                          │
│  ┌────▼─────────────────────────┐                │
│  │  Self-Attention               │                │
│  │  "Which other tokens should   │                │
│  │   I pay attention to?"        │                │
│  └────┬─────────────────────────┘                │
│       │ + residual connection                     │
│  ┌────▼─────────────────────────┐                │
│  │  Layer Normalization          │                │
│  │  Stabilizes values            │                │
│  └────┬─────────────────────────┘                │
│       │                                          │
│  ┌────▼─────────────────────────┐                │
│  │  Feed-Forward Network (FFN)   │                │
│  │  Two linear layers + ReLU     │                │
│  │  "Process what attention      │                │
│  │   gathered"                   │                │
│  └────┬─────────────────────────┘                │
│       │ + residual connection                     │
│  ┌────▼─────────────────────────┐                │
│  │  Layer Normalization          │                │
│  └────┬─────────────────────────┘                │
│       │                                          │
│  Output: transformed vectors [B, T, D]           │
└───────┼─────────────────────────────────────────┘
        │
        ▼
  (next block, or LM head)
```

### Stacking multiple blocks

```text
Token embeddings
     │
┌────▼────┐
│ Block 1  │  learns basic patterns (word relationships)
└────┬────┘
     │
┌────▼────┐
│ Block 2  │  learns intermediate patterns (phrases, syntax)
└────┬────┘
     │
┌────▼────┐
│ Block N  │  learns high-level patterns (intent, reasoning)
└────┬────┘
     │
┌────▼─────┐
│ LM Head   │  produces logits over vocabulary
└───────────┘
```

More blocks = deeper model = can learn more complex patterns (but costs more memory and compute).

### Embedding
Turns token IDs into vectors.

### Attention
Lets the model decide which previous tokens matter most for the current token.

### Causal mask
Prevents the model from seeing future tokens in generation training.

### Feed-forward block
Applies learned nonlinear transformations after attention.

### Layer stacking
Multiple layers let the model learn richer patterns.

### Residual connection
Adds the block's input back to its output.  
This helps gradients flow through deep networks and prevents training from getting stuck.

```text
Without residual:  output = block(input)
With residual:     output = block(input) + input    ← shortcut path
```

### Layer normalization
Rescales values to a stable range, preventing numbers from exploding or vanishing as they pass through many layers.

---

## A visual intuition for causal masking

This is one of the most important concepts for understanding GPT-style models.

User text:
```text
The charger is not working
```

### Attention matrix (what each token can see)

```text
             Attending to →
             The   charger   is   not   working
Querying ↓
The         ✓      ✗        ✗    ✗     ✗        sees: [The]
charger     ✓      ✓        ✗    ✗     ✗        sees: [The, charger]
is          ✓      ✓        ✓    ✗     ✗        sees: [The, charger, is]
not         ✓      ✓        ✓    ✓     ✗        sees: [The, charger, is, not]
working     ✓      ✓        ✓    ✓     ✓        sees: [The, charger, is, not, working]
```

When predicting `working`, the model can look at:
- The
- charger
- is
- not

It **cannot** look at future tokens to cheat.

That is why decoder-only generation models use causal masking.

### Masked LM comparison (BERT-style)

```text
             Attending to →
             The   charger   is   [MASK]   working
Querying ↓
The         ✓      ✓        ✓      ✓       ✓      sees: everything
charger     ✓      ✓        ✓      ✓       ✓      sees: everything
is          ✓      ✓        ✓      ✓       ✓      sees: everything
[MASK]      ✓      ✓        ✓      ✓       ✓      sees: everything
working     ✓      ✓        ✓      ✓       ✓      sees: everything
```

**Bidirectional models** can see the entire context to fill in blanks — that is why they are good at understanding but cannot generate left-to-right.

---

## Beginner checklist before moving on

You are ready for the next topic when you can answer these:

1. **What is the difference between training loss and validation loss?**
   *Training loss tells you how well the model fits the data it is learning from.  
   Validation loss tells you how well it generalizes to unseen data.  
   If training loss is low but validation loss is high → overfitting.*

2. **What is a vocabulary?**
   *The complete set of tokens the tokenizer knows, each mapped to a unique integer ID.*

3. **Why do we need a padding token?**
   *Sequences in a batch have different lengths. GPUs need rectangular tensors.  
   Padding fills shorter sequences to the same length, and the model ignores pad positions.*

4. **What are logits?**
   *Raw unnormalized scores from the model's output layer.  
   Apply softmax to get probabilities.*

5. **Why does classification use cross-entropy loss?**
   *Cross-entropy penalizes confident wrong predictions heavily.  
   It measures the gap between the model's probability distribution and the true label.*

6. **What is the difference between causal LM and masked LM?**
   *Causal LM sees only left context (for generation).  
   Masked LM sees full context (for understanding). See diagrams above.*

7. **Why is error analysis necessary even when metrics look good?**
   *Aggregate metrics can hide failures on specific classes or edge cases.  
   Only manual review reveals WHY the model fails.*

---

## Common mistakes in this phase

### Mistake 1: tuning before having a baseline
Always start with a simple baseline.

### Mistake 2: ignoring class imbalance
Rare classes need special attention.

### Mistake 3: evaluating only on training data
This tells you almost nothing useful.

### Mistake 4: making the model bigger before fixing the dataset
Usually the dataset is the first problem.

### Mistake 5: not reading failed examples
Metrics tell you **that** something is wrong.  
Manual review tells you **why**.

---

## Complete end-to-end example: ticket classifier in one script

This is a single, fully runnable script that ties together everything from Steps 1–6.  
Copy this into a Python file and run it. Every piece you learned above is combined here.

```python
"""
OrbitMart Support Ticket Classifier — Complete Foundations Example
Run with: python ticket_classifier.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter

# ──────────────────────────────────────────────────
# 1. Data
# ──────────────────────────────────────────────────
texts = [
    "Where is my delivery? It said shipped 3 days ago.",
    "When will my package arrive?",
    "Track my order 88421 please.",
    "My order hasn't arrived yet.",
    "Can you check shipping status?",
    "I want to return my headphones.",
    "How do I return a product?",
    "I need to send back the speaker.",
    "Can I return opened earbuds?",
    "Return request for order 1122.",
    "My laptop dock is not detected by Windows.",
    "The charger stopped working after two days.",
    "Bluetooth keeps disconnecting on my earbuds.",
    "Screen flickers when using the docking station.",
    "Camera does not turn on during video calls.",
    "Can you change the shipping address before dispatch?",
    "Update my delivery address please.",
    "I moved — can you redirect the package?",
    "Wrong address on my order.",
    "Change address to 123 Main St.",
    "I was charged twice for order 88421.",
    "Why is there an extra charge on my card?",
    "Refund for duplicate payment please.",
    "Billing shows wrong amount.",
    "Card was charged but order was cancelled.",
    "What colors do the NovaBuds come in?",
    "Do you sell USB-C to HDMI adapters?",
    "Is the FlexCharge pad compatible with iPhone?",
    "What are your store hours?",
    "Do you offer gift wrapping?",
]

labels = [
    "order_status", "order_status", "order_status", "order_status", "order_status",
    "return_request", "return_request", "return_request", "return_request", "return_request",
    "technical_issue", "technical_issue", "technical_issue", "technical_issue", "technical_issue",
    "address_change", "address_change", "address_change", "address_change", "address_change",
    "billing_issue", "billing_issue", "billing_issue", "billing_issue", "billing_issue",
    "other", "other", "other", "other", "other",
]

# Label mapping
label_names = sorted(set(labels))
label2id = {name: i for i, name in enumerate(label_names)}
id2label = {i: name for name, i in label2id.items()}
print("Labels:", label2id)

# ──────────────────────────────────────────────────
# 2. Train/val split (simple for this small dataset)
# ──────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

# ──────────────────────────────────────────────────
# 3. Tokenizer
# ──────────────────────────────────────────────────
def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode(text, vocab, max_len=20):
    tokens = text.lower().split()
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids

vocab = build_vocab(train_texts)
print(f"Vocabulary size: {len(vocab)}")

# ──────────────────────────────────────────────────
# 4. Dataset
# ──────────────────────────────────────────────────
class TicketDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id, max_len=20):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.vocab, self.max_len)
        y = self.label2id[self.labels[idx]]
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.long),
        }

train_dataset = TicketDataset(train_texts, train_labels, vocab, label2id)
val_dataset = TicketDataset(val_texts, val_labels, vocab, label2id)

# ──────────────────────────────────────────────────
# 5. Model
# ──────────────────────────────────────────────────
class AvgEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        mask = (input_ids != 0).unsqueeze(-1)
        x = x * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = x.sum(dim=1) / lengths
        logits = self.classifier(pooled)
        return logits

model = AvgEmbeddingClassifier(
    vocab_size=len(vocab), embed_dim=64, num_labels=len(label2id)
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ──────────────────────────────────────────────────
# 6. Train
# ──────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

for epoch in range(30):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"epoch={epoch+1}  train_loss={total_loss/len(train_loader):.4f}")

# ──────────────────────────────────────────────────
# 7. Evaluate
# ──────────────────────────────────────────────────
model.eval()
preds, gold = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch["input_ids"])
        preds.extend(logits.argmax(dim=-1).tolist())
        gold.extend(batch["label"].tolist())

print("\n--- Results ---")
print(f"Accuracy:  {accuracy_score(gold, preds):.2f}")
print(f"Macro F1:  {f1_score(gold, preds, average='macro'):.2f}")
print(classification_report(gold, preds, target_names=label_names))
```

> This is a self-contained file. It demonstrates every concept: tokenization, padding, embedding, model, training loop, loss, backpropagation, evaluation, and metrics — all in context.

---

## Exercises

### Exercise 1
Add a new label `cancel_order` and collect 20 examples.

### Exercise 2
Replace the whitespace tokenizer with a Hugging Face tokenizer.

```python
# Hint: here is how to use a pretrained tokenizer instead
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "My laptop dock is not detected by Windows."
encoded = tokenizer(text, padding="max_length", max_length=20, truncation=True)

print("tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
# ['[CLS]', 'my', 'laptop', 'dock', 'is', 'not', 'detected', 'by', 'windows', '.', '[SEP]', '[PAD]', ...]

print("input_ids:", encoded["input_ids"])
print("attention_mask:", encoded["attention_mask"])  # 1=real token, 0=padding
```

### Exercise 3
Compare:
- simple embedding classifier
- pretrained encoder classifier

### Exercise 4
Build a confusion matrix and identify the two most confused labels.

```python
# Hint: build and plot the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(gold, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("OrbitMart Ticket Classifier — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Find the two most confused pairs (highest off-diagonal values)
import numpy as np
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
flat_idx = cm_no_diag.argmax()
row, col = divmod(flat_idx, cm.shape[1])
print(f"Most confused: {label_names[row]} ↔ {label_names[col]}  ({cm_no_diag[row, col]} errors)")
```

### Exercise 5
Train the tiny next-token model on product descriptions and sample 10 completions.

---

## Mini project — End-to-end support triage baseline

Build this pipeline:

```text
incoming ticket
 -> simple classifier
 -> predicted intent
 -> confidence score
 -> queue routing
```

### Minimum deliverables
- train/val/test split
- reproducible script
- accuracy + macro F1
- top 20 failure examples
- notes on what to improve next

### Business interpretation
This is your first taste of ML engineering:
not just training a model, but deciding if the model is useful for the workflow.

---

## What you study next

Once you finish this foundations guide, move to:

- [01_fine_tuning_models.md](./01_fine_tuning_models.md) if you want to adapt strong pretrained models
- [02_training_models.md](./02_training_models.md) if you want to understand the full training stack deeply

---

## References

- PyTorch core docs: <https://docs.pytorch.org/docs/stable/torch.html>
- PyTorch optimizers: <https://docs.pytorch.org/docs/stable/optim.html>
- Hugging Face causal language modeling: <https://huggingface.co/docs/transformers/tasks/language_modeling>
- Hugging Face masked language modeling: <https://huggingface.co/docs/transformers/tasks/masked_language_modeling>