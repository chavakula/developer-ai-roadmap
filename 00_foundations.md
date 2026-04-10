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

## The two language-modeling objectives you must know

### 1. Causal language modeling
The model predicts the **next token** using only the tokens on the left.

Example:
```text
Input:  "The order arrived on"
Target: "time"
```

Used for:
- chat assistants
- code generation
- text generation
- many decoder-only LLMs

### 2. Masked language modeling
The model predicts **masked tokens** using both left and right context.

Example:
```text
Input:  "The order arrived [MASK] time"
Target: "on"
```

Used for:
- contextual understanding
- classification/representation tasks
- many encoder-style models like BERT

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

---

## Step 2 — Build a minimal tokenizer

For foundations, do not overcomplicate.  
You can start with a tiny whitespace tokenizer and later replace it with a proper subword tokenizer.

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

---

## Step 6 — Evaluate it properly

Do not stop at accuracy.

### Track these
- accuracy
- macro F1
- confusion matrix
- per-class recall

### Why macro F1 matters
In support routing, some labels are rare.  
A model can get decent accuracy while still failing badly on minority classes.

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

### Loss computation
```python
logits = model(input_ids)                    # [B, T, V]
loss = nn.CrossEntropyLoss()(
    logits.view(-1, logits.size(-1)),
    labels.view(-1)
)
```

### Why this matters
You now understand the shape of language-model training:
- model outputs one distribution per token position
- loss compares each position to the next true token

---

## From RNN to transformer mindset

Later, you will replace the RNN with a decoder-style transformer, but the training objective remains the same:

- tokenized sequence
- shifted targets
- per-token cross-entropy
- optimizer updates

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

---

## A visual intuition for causal masking

User text:
```text
The charger is not working
```

When predicting `working`, the model can look at:
- The
- charger
- is
- not

It **cannot** look at future tokens to cheat.

That is why decoder-only generation models use causal masking.

---

## Beginner checklist before moving on

You are ready for the next topic when you can answer these:

1. What is the difference between training loss and validation loss?
2. What is a vocabulary?
3. Why do we need a padding token?
4. What are logits?
5. Why does classification use cross-entropy loss?
6. What is the difference between causal LM and masked LM?
7. Why is error analysis necessary even when metrics look good?

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

## Exercises

### Exercise 1
Add a new label `cancel_order` and collect 20 examples.

### Exercise 2
Replace the whitespace tokenizer with a Hugging Face tokenizer.

### Exercise 3
Compare:
- simple embedding classifier
- pretrained encoder classifier

### Exercise 4
Build a confusion matrix and identify the two most confused labels.

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