# Day 0 — Prerequisites Knowledge Base
**Goal:** give you everything you need *before* starting [00_foundations.md](./00_foundations.md), so you never have to leave this repo to look something up.
**Audience:** complete beginner. No AI background assumed.
**Updated:** 2026-04-27

---

## How to use this file

Read it once, top to bottom. Do not try to memorize — just **recognize** the words. When a later chapter uses a term, your brain will say "I have seen this before" and you will be able to follow.

Every section ends with a "▶ Where this is used later" pointer so you know which chapter expands it.

---

## Table of contents

1. [What is AI, ML, DL, and an LLM?](#1-what-is-ai-ml-dl-and-an-llm)
2. [The 60-second mental model of how an AI model "learns"](#2-the-60-second-mental-model-of-how-an-ai-model-learns)
3. [Math you actually need (and the parts you don't)](#3-math-you-actually-need-and-the-parts-you-dont)
   - [3.1 Vectors](#31-vectors)
   - [3.2 Matrices](#32-matrices)
   - [3.3 Tensors](#33-tensors)
   - [3.4 Dot product and similarity](#34-dot-product-and-similarity)
   - [3.5 Probability basics](#35-probability-basics)
   - [3.6 Logarithms (just enough)](#36-logarithms-just-enough)
   - [3.7 Derivatives and gradients (intuition only)](#37-derivatives-and-gradients-intuition-only)
   - [3.8 Gradient descent in one picture](#38-gradient-descent-in-one-picture)
4. [Python you need](#4-python-you-need)
5. [PyTorch in 5 minutes](#5-pytorch-in-5-minutes)
6. [Tools and environment](#6-tools-and-environment)
7. [Vocabulary you'll see everywhere](#7-vocabulary-youll-see-everywhere)
8. [Common confusions cleared up](#8-common-confusions-cleared-up)
9. [Self-check before moving to chapter 00](#9-self-check-before-moving-to-chapter-00)

---

## 1. What is AI, ML, DL, and an LLM?

These four words get mixed up constantly. They are nested, not synonymous.

```text
┌────────────────────────────────────────────────────────────┐
│ AI — Artificial Intelligence                               │
│  (any system that mimics intelligent behavior)             │
│  e.g. a chess engine, a spam filter, a thermostat learning │
│  your schedule                                             │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ML — Machine Learning                                │  │
│  │  (systems that LEARN patterns from data instead of   │  │
│  │   being hand-coded with rules)                       │  │
│  │  e.g. spam filter trained on 100k emails             │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ DL — Deep Learning                             │  │  │
│  │  │  (ML using neural networks with many layers)   │  │  │
│  │  │  e.g. image classifier with 50 layers          │  │  │
│  │  │                                                │  │  │
│  │  │  ┌──────────────────────────────────────────┐  │  │  │
│  │  │  │ LLM — Large Language Model               │  │  │  │
│  │  │  │  (deep learning model with BILLIONS of   │  │  │  │
│  │  │  │   parameters trained on huge text)       │  │  │  │
│  │  │  │  e.g. GPT-5, Claude, Llama 3, Gemini     │  │  │  │
│  │  │  └──────────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

| Term | One-line meaning | Example |
|---|---|---|
| **AI** | Any system that acts "intelligently". | Chess engine, route planner. |
| **ML** | A subset of AI that learns from data. | Spam filter trained on examples. |
| **Neural network** | A specific kind of ML model made of layers of simple math units. | Image classifier. |
| **Deep learning** | Neural networks with many layers ("deep"). | Modern speech recognition. |
| **Model** | A file (weights) + the code that runs it. Takes input, produces output. | `gpt2.bin` + the code that loads it. |
| **LLM** | A deep-learning model trained on huge text, usually billions of parameters. | GPT-5, Claude, Llama 3. |
| **Generative AI** | Models that *produce* new content (text, images, audio, code). | ChatGPT, Midjourney. |
| **Foundation model** | A large pretrained model meant to be adapted to many tasks. | Llama 3, Claude, GPT-5. |

▶ Where this is used later: every chapter.

---

## 2. The 60-second mental model of how an AI model "learns"

Strip away the jargon and the loop is always the same:

```text
┌─────────────────────────────────────────────────────────────┐
│   1. Show the model an example.                             │
│         input  = "Where is my order?"                       │
│         truth  = "order_status"                             │
│                                                             │
│   2. Let the model GUESS an answer.                         │
│         guess  = "billing_issue"   (wrong)                  │
│                                                             │
│   3. Measure HOW WRONG the guess is. → call this LOSS.      │
│         loss   = 1.74                                       │
│                                                             │
│   4. Nudge the model's internal numbers a tiny bit so       │
│      that, next time, the guess will be a little closer     │
│      to the truth.                                          │
│                                                             │
│   5. Repeat steps 1–4 millions of times.                    │
└─────────────────────────────────────────────────────────────┘
```

That's it. Everything else (transformers, attention, embeddings, optimizers) is just **how** each step is implemented efficiently.

▶ Where this is used later: [00_foundations.md → "Step 5 — Train it"](./00_foundations.md).

---

## 3. Math you actually need (and the parts you don't)

You **do not** need calculus proofs, linear algebra theorems, or measure theory. You **do** need intuition for the following.

### 3.1 Vectors

A **vector** is just a list of numbers.

```text
v = [0.2, -0.7, 1.4]            ← a 3-dimensional vector
embedding_of_word = [0.12, -0.34, 0.56, ..., 0.08]   ← typically 128 / 768 / 4096 numbers
```

Why we care: **every word, every token, every image patch** in a model is represented as a vector inside the model. Words with similar meaning end up as vectors that point in similar directions.

### 3.2 Matrices

A **matrix** is a 2-D grid of numbers (a list of vectors).

```text
M = [[1, 2, 3],
     [4, 5, 6]]            ← a 2×3 matrix (2 rows, 3 columns)
```

Why we care: a neural-network "layer" is mostly a big matrix multiplication. Training a model is, mechanically, "find good numbers to put in these matrices."

### 3.3 Tensors

A **tensor** is the generalization: any N-dimensional grid of numbers.

```text
0-D tensor (scalar):   42
1-D tensor (vector):   [1, 2, 3]
2-D tensor (matrix):   [[1, 2], [3, 4]]
3-D tensor:            batch of matrices
4-D tensor:            batch of images   (batch, channels, height, width)
```

Why we care: PyTorch's main object is `torch.Tensor`. Everything you'll write moves tensors around.

### 3.4 Dot product and similarity

Take two vectors of the same length, multiply element-by-element, then sum.

```text
a = [1, 2, 3]
b = [4, 5, 6]
a · b = 1·4 + 2·5 + 3·6 = 32
```

Why we care: this single operation underlies **attention** ("how much should I look at this token?") and **semantic search** ("how similar are these two pieces of text?"). When two embeddings have a high dot product, they mean similar things.

### 3.5 Probability basics

You only need three ideas:

1. A probability is a number between 0 and 1.
2. Probabilities of mutually exclusive outcomes sum to 1.
3. **Softmax** turns any list of numbers into a probability distribution.

```text
raw scores (logits):  [2.1, -0.5, 0.3, 1.8]
after softmax:        [0.50, 0.04, 0.08, 0.38]   ← sums to 1.0
```

Why we care: every classifier and every next-token predictor outputs probabilities via softmax.

### 3.6 Logarithms (just enough)

`log(1) = 0`, `log` of a small probability is a large negative number, `log` of a probability close to 1 is near 0.

Why we care: **loss functions use logs** because multiplying many tiny probabilities underflows to zero, but adding their logs is numerically stable. When you see `log_softmax` or `cross-entropy`, this is why.

### 3.7 Derivatives and gradients (intuition only)

The **derivative** of a function at a point answers: *"if I nudge the input a tiny bit, how much does the output change, and in which direction?"*

```text
f(x) = x²

At x = 3:    f(3) = 9
At x = 3.01: f(3.01) ≈ 9.06
→ output changed by +0.06 for an input change of +0.01
→ derivative ≈ 6   (the slope at that point)
```

A **gradient** is just the derivative when the function has many inputs (which a neural network always does — millions of inputs called "parameters"). The gradient says: *"to make the loss smaller, change each parameter in this direction by this much."*

### 3.8 Gradient descent in one picture

```text
                        loss
                         │
       *                 │                  *
        \                │                /
         \               │               /
          \              │              /
           *             │            *
            \            │           /
             \           │          /
              *          │         *
               \         │        /
                \        │       /
                 *       │     *
                  \      │    /
                   *     │   *
                    \    │  /
                     \   │ /
                      ◯  │◯
                          ▼
                  bottom of the bowl  ← the model with the smallest loss

Each training step ≈ taking one small step downhill on this surface.
The size of each step is the LEARNING RATE.
```

| If learning rate is... | What happens |
|---|---|
| Way too big | You leap over the bowl and bounce around. Loss explodes. |
| Just right | You slide steadily into the bottom. Loss drops smoothly. |
| Way too small | You barely move. Training takes forever. |

▶ Where this is used later: [00_foundations.md → "Optimizer" / "Learning rate"](./00_foundations.md), every fine-tuning chapter.

---

## 4. Python you need

You need comfortable familiarity with:

- Variables, `if`/`for`/`while`
- Lists, dicts, tuples, sets
- Functions and `def`
- Classes and `self` (you only need to *read* classes, rarely write complex ones)
- Importing modules: `from x import y`
- f-strings: `f"value = {x}"`
- List comprehensions: `[x*2 for x in nums]`
- File paths and reading files

Two patterns you'll see constantly:

```python
# Pattern 1: looping through a dataset
for batch in dataloader:
    inputs = batch["input_ids"]
    labels = batch["label"]
    ...

# Pattern 2: a class that wraps a model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 10)

    def forward(self, x):
        return self.layer(x)
```

You do **not** need: decorators, metaclasses, async/await, threading. They essentially never appear in the tutorials.

▶ If you are rusty, write 10 small Python scripts (FizzBuzz, word counter, CSV reader, simple dictionary lookup) before chapter 00.

---

## 5. PyTorch in 5 minutes

PyTorch is the library used by all later chapters. Learn just these five things.

### 5.1 Create a tensor

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x.shape)   # torch.Size([3])
print(x.dtype)   # torch.float32
```

### 5.2 Move it to GPU (if you have one)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
```

### 5.3 Basic math

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
print(a + b)          # tensor([5., 7., 9.])
print((a * b).sum())  # tensor(32.)  ← dot product
```

### 5.4 A "layer" is just an `nn.Module`

```python
import torch.nn as nn

linear = nn.Linear(in_features=3, out_features=2)   # turns 3 numbers into 2
out = linear(torch.tensor([1., 2., 3.]))
print(out)   # tensor([..., ...])  (random until trained)
```

### 5.5 The four-line training step

```python
optimizer.zero_grad()       # clear old gradients
output = model(inputs)      # forward pass
loss   = loss_fn(output, targets)
loss.backward()             # compute gradients
optimizer.step()            # update weights
```

This four-line block appears in **every** chapter. Memorize it.

▶ Where this is used later: [00_foundations.md → "Step 5"](./00_foundations.md), [02_training_models.md](./02_training_models.md).

---

## 6. Tools and environment

| Tool | What it is | Why you need it |
|---|---|---|
| **Python 3.10+** | The language. | All code is Python. |
| **pip / uv** | Package manager. | Installs PyTorch, transformers, etc. |
| **venv** (`python -m venv .venv`) | Isolated project environment. | Stops different projects from breaking each other. |
| **VS Code** | Editor. | Syntax highlighting, debugger, Jupyter support. |
| **Jupyter Notebook** | Interactive Python in cells. | Quick experiments and plots. |
| **Git** | Version control. | Save and undo changes. |
| **A GPU (optional)** | Faster math chip. | Training without one is slow but possible. |
| **Hugging Face account** | Free model/data hub. | Required to download some models. |
| **OpenAI / Anthropic account** | Hosted-LLM access. | Required for chapters that call hosted APIs. |

For account / API key setup, see [00_foundations.md → "API keys" section](./00_foundations.md).

---

## 7. Vocabulary you'll see everywhere

This is the absolute minimum word list. Each term has a one-line definition; the chapter that *fully* explains it is in the right column.

| Word | One-line meaning | Fully explained in |
|---|---|---|
| Token | A small piece of text (word, subword, or character). | [00_foundations.md](./00_foundations.md) |
| Tokenizer | Tool that turns text into token IDs. | [00_foundations.md](./00_foundations.md) |
| Vocabulary | Full set of tokens a tokenizer knows. | [00_foundations.md](./00_foundations.md) |
| Embedding | Vector (list of numbers) that represents a token's meaning. | [00_foundations.md](./00_foundations.md) |
| Parameter / weight | One of the millions/billions of learnable numbers in a model. | [00_foundations.md](./00_foundations.md) |
| Forward pass | Running input through the model to get a prediction. | [00_foundations.md](./00_foundations.md) |
| Logits | Raw, unnormalized scores from the final layer. | [00_foundations.md](./00_foundations.md) |
| Softmax | Function that turns logits into probabilities. | [00_foundations.md](./00_foundations.md) |
| Loss | A single number measuring how wrong the model is. | [00_foundations.md](./00_foundations.md) |
| Backpropagation | Algorithm that computes the gradient of the loss. | [00_foundations.md](./00_foundations.md) |
| Gradient | Direction and amount to change a parameter to reduce loss. | [00_foundations.md](./00_foundations.md) |
| Optimizer | Code that uses gradients to update parameters (e.g. AdamW). | [00_foundations.md](./00_foundations.md) |
| Learning rate | How big each weight update is. | [00_foundations.md](./00_foundations.md) |
| Batch | A group of examples processed together. | [00_foundations.md](./00_foundations.md) |
| Epoch | One full pass over the training data. | [00_foundations.md](./00_foundations.md) |
| Overfitting | Model memorized training data but fails on new data. | [00_foundations.md](./00_foundations.md) |
| Train / val / test split | The three slices of your dataset. | [00_foundations.md](./00_foundations.md) |
| Self-attention | Mechanism that lets each token "look at" other tokens. | [00_foundations.md](./00_foundations.md) |
| Causal mask | Trick that hides future tokens during training. | [00_foundations.md](./00_foundations.md) |
| Encoder / Decoder | Two main transformer shapes (BERT vs GPT). | [00_foundations.md](./00_foundations.md) |
| Pretrained model | A model someone else already trained on huge data. | [00_foundations.md](./00_foundations.md), [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| Fine-tuning | Continuing training of a pretrained model on your data. | [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| LoRA / QLoRA | Cheap fine-tuning by training tiny "adapter" weights. | [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| SFT / DPO / RFT | Three flavors of fine-tuning (instructions / preferences / reward). | [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| DDP / FSDP / DeepSpeed | Ways to train across multiple GPUs. | [02_training_models.md](./02_training_models.md) |
| Mixed precision (bf16 / fp16) | Using smaller number formats to train faster. | [02_training_models.md](./02_training_models.md) |
| Tokenizer training | Building a custom vocabulary from your own text. | [03_creation_of_models.md](./03_creation_of_models.md) |
| Model card | A README describing a model's purpose, data, and limits. | [03_creation_of_models.md](./03_creation_of_models.md) |
| RAG | Retrieval-Augmented Generation — give the model an "open book". | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Chunk | A small piece of a document used as a retrieval unit. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Vector store | A database that searches by embedding similarity. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Reranker | Second-stage model that re-orders retrieved chunks. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Hallucination | Model inventing facts that aren't in the source. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Grounding / Citation | Tying an answer back to a source document. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Ingestion | The pipeline that pulls docs into a RAG system. | [05_building_rag_systems.md](./05_building_rag_systems.md) |
| ACL (Access Control List) | Rules for who is allowed to see which document. | [05_building_rag_systems.md](./05_building_rag_systems.md) |
| Trace / Eval | Logged record / scored test cases for a system. | [05_building_rag_systems.md](./05_building_rag_systems.md), [06_creating_agents.md](./06_creating_agents.md) |
| Agent | Model + tools + a loop that takes multi-step actions. | [06_creating_agents.md](./06_creating_agents.md) |
| Tool / Function-calling | Model deciding to call a typed function. | [06_creating_agents.md](./06_creating_agents.md) |
| Handoff | Passing the conversation to a specialist agent. | [06_creating_agents.md](./06_creating_agents.md) |
| Guardrail | A safety check that blocks unsafe inputs/outputs. | [06_creating_agents.md](./06_creating_agents.md) |
| MCP | Model Context Protocol — standard way to expose tools/data. | [06_creating_agents.md](./06_creating_agents.md) |

If you can read this table and feel "I have at least heard of every word," you are ready.

---

## 8. Common confusions cleared up

### "Is GPT a model, an algorithm, or a product?"
GPT is a **family of models** (GPT-2, GPT-3, GPT-4o, GPT-5...) built using the **transformer algorithm** and shipped as a **product** (ChatGPT, the API). All three names get used loosely.

### "Are AI and ML the same?"
No. AI is the umbrella; ML is a subset of AI; deep learning is a subset of ML; LLMs are a subset of deep learning. See the diagram in [section 1](#1-what-is-ai-ml-dl-and-an-llm).

### "Do I need to know calculus?"
No. You need *intuition* for derivatives (section 3.7). PyTorch computes the actual derivatives for you with `loss.backward()`.

### "Do I need a GPU?"
Recommended but not required. Chapters 00 and 04 run fine on CPU. Chapters 01, 02, 03 are much faster on a GPU; you can also use free Colab / Kaggle notebooks.

### "What is the difference between training, fine-tuning, and inference?"
- **Training from scratch:** start with random weights, learn everything. Takes huge compute. Done by big labs.
- **Fine-tuning:** start with a pretrained model, keep training on your data. Hours to days on a single GPU.
- **Inference:** *use* a trained model to get predictions. No learning happens.

### "Tokens vs words vs characters?"
Modern LLMs use **subword tokens** — usually larger than characters, often smaller than full words. "OrbitMart" might be `["Orbit", "Mart"]` (two tokens). One English word is ~1.3 tokens on average.

### "Embedding vs encoding vs encoder?"
- **Encoding** a string usually just means turning it into bytes/IDs (tokenizer's job).
- **Embedding** means turning an ID into a *learned vector of numbers*.
- **Encoder** is a kind of transformer model (BERT-style) that reads the whole sequence.

### "Parameters vs hyperparameters?"
- **Parameters** = numbers the model learns (weights).
- **Hyperparameters** = numbers *you* choose before training (learning rate, batch size, number of epochs, model size).

### "Pretrained vs fine-tuned vs instruction-tuned vs aligned?"
- **Pretrained**: trained on a giant text corpus to predict the next token. Knows language but doesn't follow instructions well.
- **Fine-tuned**: a pretrained model continued on a smaller, specific dataset.
- **Instruction-tuned**: fine-tuned specifically on (instruction, response) pairs so it follows requests.
- **Aligned (RLHF / DPO / RFT)**: further trained on human preferences so its answers are helpful, harmless, and honest. (Covered in [01_fine_tuning_models.md](./01_fine_tuning_models.md).)

---

## 9. Self-check before moving to chapter 00

You are ready for [00_foundations.md](./00_foundations.md) when you can answer **yes** to each of these:

- [ ] I can explain in one sentence the difference between AI, ML, DL, and LLM.
- [ ] I know what a vector, matrix, and tensor are.
- [ ] I understand intuitively that "training" means "nudge weights to make the loss smaller."
- [ ] I can read a simple `for` loop and a simple `class` in Python.
- [ ] I have Python 3.10+ and `pip` installed.
- [ ] I know what a token is and roughly why models use them.
- [ ] I have *seen* the words `loss`, `gradient`, `optimizer`, `epoch`, `batch`, `embedding` and have a vague feel for each.
- [ ] I have a plan for getting an OpenAI / Anthropic / Hugging Face key (see chapter 00).

If you ticked them all — open [00_foundations.md](./00_foundations.md) and start.

If you missed one or two — re-read just that section above. Don't worry about the rest; later chapters will reinforce these ideas naturally.

---

## Where to go next

| You want to... | Open |
|---|---|
| Build the core mental model with a real classifier | [00_foundations.md](./00_foundations.md) |
| Adapt big pretrained models to your data | [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| Train across multiple GPUs / understand the training stack | [02_training_models.md](./02_training_models.md) |
| Build your own tokenizer / model package | [03_creation_of_models.md](./03_creation_of_models.md) |
| Make models answer from your private docs | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Turn a RAG notebook into a real product | [05_building_rag_systems.md](./05_building_rag_systems.md) |
| Build agents that use tools and take actions | [06_creating_agents.md](./06_creating_agents.md) |
| See the whole curriculum | [MASTER_INDEX.md](./MASTER_INDEX.md) |
