# Day 0 — Prerequisites (Plain English Edition)
**Goal:** explain everything you need before chapter 00 in the simplest possible words, with everyday examples.
**Audience:** total beginner. No coding or maths background assumed.
**Updated:** 2026-04-27

---

## How to read this file

- Read it slowly, like a story.
- Every idea is explained with a real-life example (cooking, school, sport, shopping).
- You do **not** need to memorize anything.
- If one section feels heavy, skip it and come back later.
- At the end there is a short checklist. If most boxes are ticked, you are ready for chapter 00.

---

## Table of contents

1. [What are AI, ML, DL, and LLM? (with one big picture)](#1-what-are-ai-ml-dl-and-llm)
2. [How does a model "learn"? (the cooking analogy)](#2-how-does-a-model-learn)
3. [The maths you actually need](#3-the-maths-you-actually-need)
4. [The Python you actually need](#4-the-python-you-actually-need)
5. [PyTorch in 5 minutes](#5-pytorch-in-5-minutes)
6. [Tools you will install](#6-tools-you-will-install)
7. [Words you will see everywhere (with one-line meanings)](#7-words-you-will-see-everywhere)
8. [Things people get confused about](#8-things-people-get-confused-about)
9. [Are you ready? (self-check)](#9-are-you-ready)

---

## 1. What are AI, ML, DL, and LLM?

These four words are not the same thing. They are nested, like Russian dolls.

```text
┌──────────────────────────────────────────────────────────────┐
│ AI  =  any computer program that acts "smart".               │
│        Example: the auto-correct on your phone.              │
│                                                              │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ ML  =  AI that LEARNS from examples instead of being   │ │
│   │        hand-written rules.                             │ │
│   │        Example: Gmail's spam filter.                   │ │
│   │                                                        │ │
│   │   ┌──────────────────────────────────────────────────┐ │ │
│   │   │ DL  =  ML that uses big "neural networks"         │ │ │
│   │   │        (lots of layers).                         │ │ │
│   │   │        Example: face unlock on your phone.       │ │ │
│   │   │                                                  │ │ │
│   │   │   ┌────────────────────────────────────────────┐ │ │ │
│   │   │   │ LLM  =  a very large DL model trained on   │ │ │ │
│   │   │   │         huge amounts of text.              │ │ │ │
│   │   │   │         Example: ChatGPT, Claude.          │ │ │ │
│   │   │   └────────────────────────────────────────────┘ │ │ │
│   │   └──────────────────────────────────────────────────┘ │ │
│   └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Plain language

- **AI** is the **whole world** of "smart" computer behaviour.
- **ML** is the part of that world that **learns from data** (not from hand-written rules).
- **DL** is the part of ML that uses **deep neural networks** (think: many layers of small calculations).
- **LLM** is one **kind** of deep-learning model that is trained on **lots of text** so it can read and write language.

### A real example for each

| Word | Real example |
|---|---|
| AI | A chess engine that decides the next move. |
| ML | A spam filter that learned from 1 million emails what spam looks like. |
| DL | An app that recognises your face from a photo. |
| LLM | ChatGPT writing an email for you. |

> Whenever you see "model", just think: **a file plus the code that runs it**. You give it some input, it gives you some output.

---

## 2. How does a model "learn"?

This is the most important idea. Once you get this, the rest is easy.

### The cooking analogy

Imagine you are learning to bake a cake without a recipe. Here is what you do:

1. **Try a recipe.** You guess the amounts of flour, sugar, butter.
2. **Bake it and taste.** You compare it to a "good cake".
3. **Notice what was wrong.** Maybe it was too dry, or not sweet enough.
4. **Adjust.** Next time, add a little more butter, a little more sugar.
5. **Repeat** many times until your cake matches the "good cake".

A model learns in **exactly the same way**:

| Cooking | Model training |
|---|---|
| Your guessed recipe | The model's current numbers (called **weights**) |
| Bake the cake | **Forward pass** — run the input through the model |
| Taste vs the "good cake" | **Loss** — how wrong was the guess? |
| Notice what to change | **Backpropagation** — figure out which numbers were responsible |
| Adjust the recipe | **Optimizer step** — change the numbers a tiny bit |
| Repeat 100 times | **Epochs / steps** — repeat with many examples |

That is the whole game. Everything fancy you will read later (transformers, attention, embeddings) is just a clever way to do these same five steps faster and on a bigger scale.

### One picture

```text
┌──────────────────────────────────────────────────────────────┐
│  Step 1.  Show the model an example.                         │
│           input  = "Where is my order?"                      │
│           truth  = "this is an ORDER question"               │
│                                                              │
│  Step 2.  The model GUESSES.                                 │
│           guess  = "this is a BILLING question"   (wrong!)   │
│                                                              │
│  Step 3.  Measure how wrong = LOSS.                          │
│           loss   = 1.74   (a number — bigger means worse)    │
│                                                              │
│  Step 4.  Nudge the model's numbers a tiny bit so the next   │
│           guess will be closer to the truth.                 │
│                                                              │
│  Step 5.  Go back to step 1 with the next example.           │
│           Do this thousands or millions of times.            │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. The maths you actually need

Good news: you do **not** need calculus, proofs, or fancy formulas. You only need the **feel** of a few simple ideas.

### 3.1 A vector is just a list of numbers

```text
A vector:  [3, 7, 2]
```

That's it. Three numbers in a row.

**Real example.** Think of a person described by 3 numbers: `[height, weight, age]`. So `[170, 65, 30]` is one person, `[180, 80, 25]` is another. Each person is a vector.

In AI, every word is also turned into a vector — usually a very long one, like `[0.12, -0.4, 0.9, ..., 0.07]` with hundreds of numbers. We call this an **embedding** (more on that later).

### 3.2 A matrix is a table of numbers

```text
A matrix (2 rows, 3 columns):

   [ 3, 7, 2 ]
   [ 1, 4, 5 ]
```

**Real example.** A school timetable is a matrix: rows = days, columns = periods, each cell = a subject.

A neural-network "layer" is mostly a big matrix multiplication. You don't have to know how to multiply matrices by hand — the computer does it. You just need to know **a layer = a matrix of learnable numbers**.

### 3.3 A tensor is just "any number of dimensions"

```text
A single number:        42                        ← 0-D tensor
A list:                 [1, 2, 3]                  ← 1-D tensor (vector)
A table:                [[1, 2], [3, 4]]           ← 2-D tensor (matrix)
A stack of tables:      [[[1,2],[3,4]], [[5,6],[7,8]]]  ← 3-D tensor
```

**Real example.** One photo is a 3-D tensor: `[height, width, colour-channels]`. A folder of 100 photos is a 4-D tensor: `[100, height, width, channels]`.

PyTorch's main object is `torch.Tensor`. Everything in this course moves tensors around.

### 3.4 Dot product = "how similar are these two lists?"

You take two lists of the same length, multiply them position-by-position, then add up.

```text
a = [1, 2, 3]
b = [4, 5, 6]

a · b  =  (1×4) + (2×5) + (3×6)
        =   4   +  10   +  18
        =  32
```

**Why this matters.** When two word embeddings have a **high** dot product, the words mean **similar** things.

**Real example.** Imagine each word is rated on `[is_food, is_animal, is_emotion]`:
- `pizza` → `[1.0, 0.0, 0.0]`
- `burger` → `[1.0, 0.0, 0.0]`
- `tiger` → `[0.0, 1.0, 0.0]`

Dot product of `pizza` and `burger` = `1×1 + 0×0 + 0×0 = 1` → **similar (both food)**.
Dot product of `pizza` and `tiger`  = `1×0 + 0×1 + 0×0 = 0` → **not similar**.

This single trick is how Google-style search and chatbot "memory" work.

### 3.5 Probability — only three rules

1. A probability is a number between **0** (impossible) and **1** (certain).
2. The probabilities of all possible outcomes add up to **1**.
3. **Softmax** is a button you press to turn any list of numbers into proper probabilities.

```text
Raw scores from the model:    [2.1, -0.5, 0.3, 1.8]
                                         ↓  press softmax button
Proper probabilities:         [0.50, 0.04, 0.08, 0.38]   ← adds up to 1.00
```

**Real example.** A weather model spits out scores `[sunny=2.1, rainy=-0.5, cloudy=0.3, snowy=1.8]`. Softmax turns those into `[50% sunny, 4% rainy, 8% cloudy, 38% snowy]`.

### 3.6 Logarithms — only the feel

You will see the word `log` a lot. You only need this feel:

- `log(1) = 0`  → "perfect, no error"
- `log(0.5)` is small negative → "a little wrong"
- `log(0.001)` is a big negative → "very wrong"

That's why **loss** functions use logs: tiny mistakes become small numbers, big mistakes become big numbers, and the model knows which way to fix things.

### 3.7 What is a "gradient"?

The gradient is just an **arrow** that tells the model:
- **which direction** to nudge each number, and
- **by how much**.

**Real example.** You are hiking on a foggy hill and you want to walk **down**. You can't see the bottom, but you can feel which way the ground slopes under your feet. That slope, in every direction at once, is the **gradient**.

```text
                       loss
                        │
    *                   │                    *
     \                  │                   /
      \                 │                  /
       *                │                 *
        \               │                /
         *              │               *
          \             │              /
           *            │             *
            \           │            /
             ◯          │           ◯
                         ▼
                  bottom of the hill  ← we want to be here
                  (the model that is least wrong)
```

Each training step = take **one small step downhill**.
The size of that step is called the **learning rate**.

| Learning rate | What happens (hiking version) |
|---|---|
| Too big | You leap and fly off the hill. |
| Just right | You walk steadily to the bottom. |
| Too small | You shuffle. It takes forever. |

That is honestly the whole maths story. The computer does all the actual calculations.

---

## 4. The Python you actually need

You are comfortable enough if you can read code like this and understand roughly what it does:

```python
# Variables and lists
fruits = ["apple", "banana", "cherry"]

# A loop
for fruit in fruits:
    print("I like", fruit)

# An if statement
if "apple" in fruits:
    print("Yes, apple is here")

# A dictionary (key → value)
prices = {"apple": 50, "banana": 20}
print(prices["apple"])   # prints 50

# A function
def add(a, b):
    return a + b

print(add(3, 4))   # prints 7

# An f-string (insert a value into text)
name = "Sam"
print(f"Hello, {name}")
```

You will also see **classes**. You don't need to write complex ones — just **read** them. A class is a recipe for making objects:

```python
class Dog:
    def __init__(self, name):
        self.name = name          # stored on every Dog object

    def bark(self):
        print(f"{self.name} says woof!")

d = Dog("Rex")
d.bark()    # prints "Rex says woof!"
```

You do **not** need: decorators, async/await, threading. They almost never show up in the tutorials.

If you feel rusty, write 5 tiny scripts before chapter 00:
1. Print numbers 1 to 10.
2. Read a CSV file using `pandas`.
3. Count how many times each word appears in a sentence.
4. A function that returns the average of a list.
5. A class `Cat` with a name and a `meow()` method.

---

## 5. PyTorch in 5 minutes

PyTorch is the library every later chapter uses. Learn just these five things now.

### 5.1 Make a tensor

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x)          # tensor([1., 2., 3.])
print(x.shape)    # torch.Size([3])
```

Think of `torch.tensor` as PyTorch's version of a Python list, but built for fast maths.

### 5.2 Move it to a GPU (if you have one)

A GPU is a special chip that does big maths super fast. It is optional.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
```

Reading: "if there is a GPU, use it; otherwise stay on CPU." Everything still works on CPU — just slower.

### 5.3 Do basic maths

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

print(a + b)           # tensor([5., 7., 9.])
print((a * b).sum())   # tensor(32.)   ← this is the dot product from section 3.4
```

### 5.4 A "layer" is just a tiny `nn.Module`

```python
import torch.nn as nn

# A layer that turns 3 numbers into 2 numbers.
linear = nn.Linear(in_features=3, out_features=2)

# Run it
out = linear(torch.tensor([1., 2., 3.]))
print(out)   # tensor([..., ...])  (random until trained)
```

Real-world translation: "I have an input vector of length 3 and want an output vector of length 2." A `Linear` layer is one matrix multiplication.

### 5.5 The four-line training step (memorise this)

This block appears in **every** later chapter:

```python
optimizer.zero_grad()              # 1. forget last step's gradients
prediction = model(inputs)         # 2. forward pass — make a guess
loss       = loss_fn(prediction, targets)   # 3. how wrong was the guess?
loss.backward()                    # 4. work out gradients
optimizer.step()                   # 5. nudge the numbers
```

Compare back to the cake analogy:
1. Forget last attempt's notes.
2. Bake.
3. Taste vs the perfect cake.
4. Note what was wrong.
5. Adjust the recipe.

That's it. That is what every model in this course is doing under the hood.

---

## 6. Tools you will install

| Tool | What it is, in plain words | Why you need it |
|---|---|---|
| **Python 3.10+** | The language you will write code in. | All code in this course is Python. |
| **pip** | A "tool installer" that comes with Python. | Lets you install libraries like PyTorch with one command. |
| **venv** (`python -m venv .venv`) | A separate "box" for each project's libraries. | Stops project A from breaking project B. |
| **VS Code** | A code editor (like Word, but for code). | Highlights mistakes, runs code, debugs. |
| **Jupyter Notebook** | An editor where code and notes live together in cells. | Great for experimenting and seeing results step by step. |
| **Git** | A "save with history" tool for code. | Lets you undo changes and share code on GitHub. |
| **A GPU (optional)** | A chip that does big maths fast. | Training is faster with one. Without one, things still work — just slower. |
| **Hugging Face account** | Free website where pretrained models are hosted. | You'll download models from here. |
| **OpenAI / Anthropic account** | The companies behind ChatGPT and Claude. | Some chapters call their APIs. |

For how to set the API keys (so the code can talk to OpenAI / Claude / Hugging Face), see the **API keys** section at the top of [00_foundations.md](./00_foundations.md).

---

## 7. Words you will see everywhere

Think of this as a small phrasebook. Read it once. When a chapter uses a word, your brain will recognise it.

| Word | One-line meaning (in plain English) | Where it is fully explained |
|---|---|---|
| Token | A small chunk of text — could be a word, half a word, or a single letter. | [00_foundations.md](./00_foundations.md) |
| Tokenizer | The tool that chops text into tokens and gives each one a number. | [00_foundations.md](./00_foundations.md) |
| Vocabulary | The full list of tokens the tokenizer knows. Like a dictionary. | [00_foundations.md](./00_foundations.md) |
| Embedding | A long list of numbers that represents a word's "meaning". | [00_foundations.md](./00_foundations.md) |
| Parameter / Weight | One of the millions or billions of numbers inside a model. The model "learns" by changing them. | [00_foundations.md](./00_foundations.md) |
| Forward pass | Letting the model make a guess. (No learning yet.) | [00_foundations.md](./00_foundations.md) |
| Logits | The model's raw scores before they become proper probabilities. | [00_foundations.md](./00_foundations.md) |
| Softmax | A button that turns raw scores into proper probabilities. | [00_foundations.md](./00_foundations.md) |
| Loss | A single number that says "how wrong" the guess was. Smaller = better. | [00_foundations.md](./00_foundations.md) |
| Backpropagation | The maths trick that finds out which numbers caused the wrongness. | [00_foundations.md](./00_foundations.md) |
| Gradient | The arrow that says "nudge this number this much, in this direction". | [00_foundations.md](./00_foundations.md) |
| Optimizer | The little helper that uses the gradient to actually nudge the numbers (e.g. AdamW). | [00_foundations.md](./00_foundations.md) |
| Learning rate | How big each nudge is. | [00_foundations.md](./00_foundations.md) |
| Batch | A group of examples shown to the model at the same time. | [00_foundations.md](./00_foundations.md) |
| Epoch | One full lap through all your training data. | [00_foundations.md](./00_foundations.md) |
| Overfitting | The model memorised the training examples but fails on new ones. (Like a student who memorised the answers, not the topic.) | [00_foundations.md](./00_foundations.md) |
| Train / Val / Test | Three pieces of your data: one to learn from, one to tune with, one to grade with at the end. | [00_foundations.md](./00_foundations.md) |
| Self-attention | A trick that lets each word "look at" the other words to understand context. | [00_foundations.md](./00_foundations.md) |
| Causal mask | A blindfold that hides future words during training so the model can't cheat. | [00_foundations.md](./00_foundations.md) |
| Encoder / Decoder | Two main shapes of transformer. Encoder = good at understanding (BERT). Decoder = good at writing (GPT). | [00_foundations.md](./00_foundations.md) |
| Pretrained model | A model that someone else already trained on huge data. You start from this. | [00_foundations.md](./00_foundations.md), [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| Fine-tuning | Continuing the training of a pretrained model on **your** data so it learns your style. | [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| LoRA / QLoRA | Cheap fine-tuning. Instead of changing the whole model, you train a tiny "add-on". | [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| SFT / DPO / RFT | Three flavours of fine-tuning: showing examples / showing preferences / using a reward. | [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| DDP / FSDP / DeepSpeed | Ways to share training across many GPUs. | [02_training_models.md](./02_training_models.md) |
| Mixed precision (bf16 / fp16) | Using smaller "shoe size" numbers to train faster. | [02_training_models.md](./02_training_models.md) |
| Tokenizer training | Building your own vocabulary from your own text. | [03_creation_of_models.md](./03_creation_of_models.md) |
| Model card | A README that says what a model is, how it was trained, and where not to use it. | [03_creation_of_models.md](./03_creation_of_models.md) |
| RAG | "Open-book exam" for the model: fetch some documents, then answer using them. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Chunk | A small piece of a document (like a paragraph) that we store in a search index. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Vector store | A database that searches by **meaning**, not by exact words. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Reranker | A second helper that re-orders search results so the best ones go first. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Hallucination | When the model **makes up** something that is not in the source. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Grounding / Citation | Tying the answer back to a real document so the user can check. | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Ingestion | The pipeline that pulls documents from places like Notion or S3 into your system. | [05_building_rag_systems.md](./05_building_rag_systems.md) |
| ACL | "Who is allowed to see what" — like file permissions. | [05_building_rag_systems.md](./05_building_rag_systems.md) |
| Trace / Eval | A logged record of one run / a scored test for the system. | [05_building_rag_systems.md](./05_building_rag_systems.md), [06_creating_agents.md](./06_creating_agents.md) |
| Agent | A model + a list of tools + a loop. It can take actions, not just talk. | [06_creating_agents.md](./06_creating_agents.md) |
| Tool / Function-calling | The model deciding to call a real function (e.g. `get_order_status`). | [06_creating_agents.md](./06_creating_agents.md) |
| Handoff | One agent passing the conversation to another, more specialised agent. | [06_creating_agents.md](./06_creating_agents.md) |
| Guardrail | A safety check that blocks bad inputs or bad answers. | [06_creating_agents.md](./06_creating_agents.md) |
| MCP | "Model Context Protocol" — a standard plug for connecting tools and data to any agent. | [06_creating_agents.md](./06_creating_agents.md) |

If you can read this table and feel "I have at least **heard of** every word", you are ready.

---

## 8. Things people get confused about

### "Is GPT a model, an algorithm, or a product?"
GPT is a **family of models** (GPT-2, GPT-3, GPT-4o, GPT-5...). The recipe used to build them is the **transformer**. The product you talk to is **ChatGPT** (or the API). All three names get used loosely.

### "Are AI and ML the same thing?"
No. AI is the **whole umbrella**. ML is one slice of AI. DL is one slice of ML. LLMs are one slice of DL. See the Russian-doll picture in section 1.

### "Do I need to know calculus?"
**No.** You only need the **feeling** of "gradient = which way to nudge things". The computer does all the actual maths via `loss.backward()`.

### "Do I need a GPU?"
**Not for chapters 00 and 04.** They run fine on a normal laptop. Chapters 01, 02, 03 are much faster on a GPU. If you don't have one, free **Google Colab** or **Kaggle** notebooks give you a GPU for a few hours per day.

### "What is the difference between training, fine-tuning, and inference?"
- **Training from scratch** — start with random numbers, learn everything. Costs millions. Done by big labs.
- **Fine-tuning** — start from a pretrained model, keep training on your data. Hours on one GPU.
- **Inference** — just **using** a trained model to get answers. No learning happens.

Cooking analogy: training = inventing a new recipe; fine-tuning = adjusting someone's recipe to your taste; inference = cooking the dish for dinner.

### "Tokens vs words vs characters?"
Modern LLMs use **subword tokens**: usually bigger than a letter, often smaller than a whole word.
- "OrbitMart" might become two tokens: `["Orbit", "Mart"]`.
- One average English word ≈ 1.3 tokens.

### "Embedding vs encoding vs encoder?"
- **Encoding** text = turning it into numbers (the tokenizer's job).
- **Embedding** = turning a token ID into a **learned vector** (a list of numbers).
- **Encoder** = a kind of model (BERT-style) that reads the whole sentence at once.

Don't worry if these blur together at first — the difference becomes obvious in chapter 00.

### "Parameters vs hyperparameters?"
- **Parameters** = numbers the model **learns** (weights).
- **Hyperparameters** = numbers **you choose** before training (learning rate, batch size, number of epochs, model size).

### "Pretrained vs fine-tuned vs instruction-tuned vs aligned?"
- **Pretrained** — knows language but doesn't follow instructions well. Like a person who has read a million books but never been asked a question.
- **Fine-tuned** — same model, but extra trained on a smaller specific dataset.
- **Instruction-tuned** — fine-tuned specifically on (instruction, answer) pairs so it follows requests.
- **Aligned (RLHF / DPO / RFT)** — extra trained on human preferences so it tries to be helpful, honest, and not harmful.

Chapter 01 explains each in detail.

---

## 9. Are you ready?

Tick each box mentally. You don't need every box — most is enough.

- [ ] I can say in one sentence what AI, ML, DL and LLM each mean.
- [ ] I get the cooking analogy: a model "learns" by guessing, measuring how wrong it was, and nudging itself.
- [ ] I know what a vector, matrix and tensor are (lists, tables, and stacks of tables).
- [ ] I get that a "gradient" is just an arrow that says which way to nudge things.
- [ ] I can read a Python `for` loop, an `if`, and a small `class` without panicking.
- [ ] I have Python 3.10+ installed (or I know how to install it).
- [ ] I have heard the words **token**, **embedding**, **loss**, **gradient**, **optimizer**, **epoch**, **batch** before.
- [ ] I know roughly how to get an OpenAI / Anthropic / Hugging Face key (chapter 00 walks me through it).

If yes — open [00_foundations.md](./00_foundations.md).
If you are unsure on one or two — re-read just those sections. Don't worry, the later chapters explain everything again with more examples.

---

## Where to go next

| You want to... | Open this file |
|---|---|
| Build the core mental model with a real tiny classifier | [00_foundations.md](./00_foundations.md) |
| Adapt big pretrained models to **your** data | [01_fine_tuning_models.md](./01_fine_tuning_models.md) |
| Train across multiple GPUs / understand the training stack | [02_training_models.md](./02_training_models.md) |
| Build your own tokenizer / model package | [03_creation_of_models.md](./03_creation_of_models.md) |
| Make models answer from your private docs | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) |
| Turn a RAG notebook into a real product | [05_building_rag_systems.md](./05_building_rag_systems.md) |
| Build agents that use tools and take actions | [06_creating_agents.md](./06_creating_agents.md) |
| See the whole curriculum on one page | [MASTER_INDEX.md](./MASTER_INDEX.md) |
