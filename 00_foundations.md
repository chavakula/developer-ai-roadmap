# 00 — Foundations (in plain English)

**Goal:** understand, in everyday language, how an AI model actually learns — without any prior background.
**Story we'll use:** OrbitMart, a small online electronics shop drowning in customer emails.
**Updated:** 2026-04-28

> **Brand new to AI?** Start with [day_zero_prerequisites.md](./day_zero_prerequisites.md) first. It explains AI vs ML vs DL vs LLM, the tiny bit of math you need, and Python/PyTorch in 5 minutes — all in plain English.

---

## The whole chapter in one paragraph

OrbitMart gets hundreds of customer emails every day ("Where's my package?", "I want a refund", "My charger is broken"). Today a human reads each email and forwards it to the right team. That is slow and expensive. **Our job:** teach a small AI model to read each email and pick the right team automatically. While we build that tiny thing, every important idea in modern AI shows up — tokens, embeddings, training, loss, evaluation. Once you understand them here, the rest of the curriculum will feel obvious.

---

## The mental model: AI is just "guess, check, adjust"

Forget the buzzwords for a moment. Every AI model — even ChatGPT — does the same three things over and over:

1. **Guess.** Look at the input, output a guess.
2. **Check.** Compare the guess to the right answer. How wrong was it?
3. **Adjust.** Nudge its internal numbers a tiny bit so next time it's slightly less wrong.

Do that millions of times on millions of examples and you get something that looks like magic.

> **Real-life analogy.** A toddler learning to throw a basketball. Throw → too short → throw harder. Throw → too far → throw softer. After a hundred tries the throws land in the basket. The toddler's brain is doing "guess, check, adjust." That's all training is.

---

## The pipeline you're about to build (one picture)

```text
"Where is my package?"          ← the customer's email (just text)
        │
        ▼ tokenizer
[Where, is, my, package, ?]     ← chop the sentence into pieces
        │
        ▼ vocabulary lookup
[42, 18, 7, 233, 9]             ← turn each piece into a number
        │
        ▼ embedding layer
[[0.1, -0.4, ...], ...]         ← turn each number into a list of numbers
                                  (that's what "meaning" looks like to a computer)
        │
        ▼ model layers
[[...thinking...], ...]         ← the model "thinks" about the words
        │
        ▼ classifier
[2.1, -0.5, 0.3, 1.8, -1.2, 0.9]  ← one score per team (these are "logits")
        │
        ▼ pick the highest
"order_status"                  ← the predicted team
```

That's it. Every section below zooms into one box of this picture.

---

## The vocabulary you actually need (plain English)

Don't memorize. Just skim. You'll come back here when a word confuses you.

| Jargon | What it really means | Real-life version |
|---|---|---|
| **Token** | A small chunk of text the model works with — usually part of a word | Cutting a recipe into individual ingredients |
| **Tokenizer** | The tool that does the chopping and gives each chunk a number | A barcode scanner at a grocery store |
| **Vocabulary** | The list of all chunks the tokenizer knows | The full menu at a restaurant |
| **Embedding** | A list of numbers that represents the "meaning" of one chunk | A fingerprint — a unique pattern of numbers per word |
| **Tensor** | A box of numbers (1D = list, 2D = grid, 3D = stack of grids) | A spreadsheet (or a stack of spreadsheets) |
| **Parameters / weights** | The internal numbers the model is trying to learn | The knobs on a sound mixer — millions of them |
| **Forward pass** | The model takes the input and produces a guess | A student reads a question and writes an answer |
| **Logits** | The raw scores the model spits out (one per option) | Confidence scores before any tidying up |
| **Softmax** | A small step that turns those scores into clean percentages that add to 100% | Curving a class's grades into percentages |
| **Loss** | A single number that says "how wrong was the guess" | The score on a graded test (lower = better) |
| **Backpropagation** | Math that figures out which knobs to turn, and which way | A coach reviewing the play and telling each player what to fix |
| **Gradient** | One number per knob saying "turn this knob this much, this direction" | The arrow on a thermostat showing "warmer / cooler" |
| **Optimizer** | The thing that actually turns the knobs based on the gradients | The hand on the thermostat |
| **Learning rate** | How big each knob-turn is | How aggressively you're allowed to adjust the thermostat per try |
| **Epoch** | One full pass through all your training data | Reading the whole textbook once |
| **Batch** | A small group of examples processed together | Cooking 12 burgers at once instead of one at a time |
| **Overfitting** | The model memorizes instead of learning | A student who memorizes practice questions but flunks the real test |
| **Inference** | Just using the trained model (no learning happening) | Driving the car after the driving lessons are over |

Don't worry if half of this still feels fuzzy. We're about to use them in context.

---

## Step 1 — Tokens: how a computer reads text

Computers don't understand "Where is my package?" They only understand numbers. So before anything else, we chop text into pieces and assign each piece a number.

```text
"The USB-C charger is not working"
       │
       ▼ (tokenizer chops it)
["The", " USB", "-", "C", " charger", " is", " not", " working"]
       │
       ▼ (look each piece up in a giant dictionary)
[464,  10448,  12,  34,  24530,  318,  407,  1762]
```

> **Why subwords?** If we used full words, the dictionary would have to contain every word ever invented, including typos and brand names. By breaking words into smaller chunks, the model can handle any new word ("orbitmart", "smartwatchify") by combining pieces it already knows.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# A real OrbitMart support email:
email = "My USB-C charger is not working since yesterday"

tokens = tokenizer.tokenize(email)
ids    = tokenizer.encode(email)

print("tokens:", tokens)
# ['My', 'ĠUSB', '-', 'C', 'Ġcharger', 'Ġis', 'Ġnot', 'Ġworking',
#  'Ġsince', 'Ġyesterday']

print("ids   :", ids)
# [3666, 11403, 12, 34, 24530, 318, 407, 1762, 1201, 7415]
```

Notice how `USB-C` got split into `USB`, `-`, `C` — the tokenizer didn't know `USB-C` as one word, but it could still handle it by combining smaller pieces.

The model never sees text. It only sees those integers.

#### Try it on a few OrbitMart emails

```python
for email in [
    "Where is my package?",
    "I want to return my NovaBuds.",
    "Charged twice for order 88421",
]:
    print(f"{len(tokenizer.encode(email)):>3} tokens  |  {email}")
# 5 tokens  |  Where is my package?
# 8 tokens  |  I want to return my NovaBuds.
# 7 tokens  |  Charged twice for order 88421
```

This is also how you estimate **API costs** — OpenAI and Anthropic charge per token, so this number directly maps to dollars.

---

## Step 2 — Embeddings: turning numbers into "meaning"

The integer `464` ("The") and `465` (something else) aren't related. To a computer, they're just two numbers. So we do one more step: each integer becomes a **list of numbers** (e.g. 128 numbers long). That list is called an **embedding**.

```text
Token "The"  →  ID 464  →  [0.12, -0.34, 0.56, ..., 0.08]   (128 numbers)
Token "king" →  ID 778  →  [0.41,  0.29, 0.07, ..., 0.55]
```

These number-lists start out random. As the model trains, they slowly arrange themselves so that words with similar meaning end up with similar lists. After training, "king" and "queen" land near each other in this number-space; "king" and "carrot" land far apart.

> **Real-life analogy.** Imagine plotting every word on a giant 3D map. After training, "happy", "joyful", "cheerful" cluster on one hill. "Sad", "miserable", "blue" cluster on another. The model didn't "understand" the words — it just learned which ones tend to show up in similar contexts. That clustering is the embedding.

#### Code: turn an OrbitMart email into embeddings

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size      # 50257
embed_dim  = 16                        # tiny on purpose for printing

embedding = nn.Embedding(vocab_size, embed_dim)

email = "refund my charger"
ids   = torch.tensor(tokenizer.encode(email))     # shape: [3]
vecs  = embedding(ids)                            # shape: [3, 16]

print("shape :", vecs.shape)
print("refund:", vecs[0])
# tensor([-0.41, 0.07, ..., 0.32], grad_fn=<...>)
```

Three words → three rows of 16 numbers each. Those 48 numbers are the **only** thing the model sees — it has no idea "refund" is a word. It just learns that this particular row of numbers tends to show up when something needs to be paid back.

#### Why embeddings matter for OrbitMart

```python
import torch.nn.functional as F

# Pretend after training, these are the (mean) embeddings of three OrbitMart emails:
refund_email = torch.tensor([0.8, 0.1, -0.3, 0.5])
charge_email = torch.tensor([0.7, 0.2, -0.4, 0.6])    # billing-ish
delivery_email = torch.tensor([-0.2, 0.9, 0.4, 0.1])  # totally different topic

print("refund vs billing :", F.cosine_similarity(refund_email, charge_email, 0).item())
print("refund vs delivery:", F.cosine_similarity(refund_email, delivery_email, 0).item())
# refund vs billing : 0.99   ← close in meaning
# refund vs delivery: 0.18   ← far apart
```

This is the same trick that powers **RAG** in [chapter 04](./04_empowering_models_with_rag.md): "find the policy passage whose embedding is closest to the customer's question."

---

## Step 3 — The model: stacks of "thinking"

Once we have a list of embedding vectors (one per token), the **model** does its thinking. For now, picture it as a black box that takes "vectors going in" and produces "vectors coming out, but smarter."

The most important kind of model today is the **transformer**, which uses something called **self-attention**:

> **Self-attention in plain English.** For each word in the sentence, the model asks: "Which of the other words should I pay the most attention to right now?" When processing "working" in *"The charger is not working"*, it pays a lot of attention to "not" (negation matters!) and "charger" (what's broken?), and very little to "the".

> **Real-life analogy.** You're reading a long email and trying to figure out the action item. Your eyes don't weight every word equally — they jump to the deadline, the request, the names. Self-attention is the model doing the same thing, automatically.

You don't need to understand transformer math right now. Just remember: the model takes vectors in, mixes them up using attention, and produces new vectors out.

#### Code: run a real model on an OrbitMart email

```python
from transformers import AutoTokenizer, AutoModel
import torch

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

email = "My laptop dock is not detected by Windows"
inputs = tok(email, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)

print("input shape :", inputs["input_ids"].shape)   # [1, 11]  (1 email, 11 tokens)
print("output shape:", out.last_hidden_state.shape)  # [1, 11, 768]
```

For each of the 11 tokens, the model produced 768 numbers. That `[1, 11, 768]` tensor is what "thinking" looks like to a computer — it's the email, but enriched so that every token now "knows" about every other token in context.

---

## Step 4 — Logits: the model's raw guess

After the model thinks, it produces one final list of numbers — one number per possible answer. Those raw numbers are called **logits**.

For our OrbitMart classifier with 6 teams:

```text
logits = [2.1, -0.5, 0.3, 1.8, -1.2, 0.9]
           ↓     ↓    ↓    ↓     ↓    ↓
       order  return tech billing addr  other
```

The team with the highest number wins. Here `2.1` (order_status) is highest, so the model predicts "send this email to the orders team."

If you want clean percentages instead of raw numbers, run the logits through **softmax**:

```text
logits   = [2.1, -0.5, 0.3, 1.8, -1.2, 0.9]
softmax →  [42%, 3%,   7%,  31%,  2%,   13%]   ← always adds to 100%
```

> **Real-life analogy.** A panel of judges scores a contestant: 9.2, 7.5, 8.1, 9.8, 6.3. Those raw scores are logits. Curving them into "this contestant has a 38% chance of winning" — that's softmax.

#### Code: route an OrbitMart email using logits + softmax

```python
import torch

teams = ["order_status", "return_request", "technical_issue",
         "billing_issue", "address_change", "other"]

# Pretend the model just produced these scores for an incoming email:
logits = torch.tensor([2.1, -0.5, 0.3, 1.8, -1.2, 0.9])
probs  = torch.softmax(logits, dim=0)

for team, p in zip(teams, probs):
    print(f"{team:<18} {p.item():.1%}")
# order_status       42.4%
# return_request      3.2%
# technical_issue     7.0%
# billing_issue      31.5%
# address_change      1.6%
# other              12.8%

picked = teams[probs.argmax().item()]
print(f"\n→ route to: {picked}")
# → route to: order_status
```

**Production tip:** if the top probability is below a threshold (say 60%), don't auto-route — send the email to a human. The model just told you it's not confident.

---

## Step 5 — Loss: measuring how wrong the guess was

Now compare the guess to the correct answer. **Loss** is a single number: bigger = the model was more wrong.

For classification we use a function called **CrossEntropyLoss**. You don't need to know the formula. You just need to know:

- Right answer + confident → tiny loss (good)
- Wrong answer + confident → huge loss (bad — get punished hard)
- Right answer + unsure → small loss
- Wrong answer + unsure → moderate loss

> **Real-life analogy.** A weather app says "100% chance of sunshine" and it pours rain. People are furious — that's huge loss. If it had said "60% sunshine" and it rained, people would just shrug — moderate loss. Cross-entropy punishes confident wrong predictions the most.

```python
import torch
criterion = torch.nn.CrossEntropyLoss()
logits = torch.tensor([[2.1, -0.5, 0.3, 1.8, -1.2, 0.9]])
true_label = torch.tensor([0])   # correct team is index 0
loss = criterion(logits, true_label)
print(loss.item())   # ~0.86
```

#### See how loss reacts when the model is right vs wrong

```python
import torch
criterion = torch.nn.CrossEntropyLoss()

# Truth: this email belongs to team 3 (billing_issue).
true = torch.tensor([3])

scenarios = {
    "Confidently right": torch.tensor([[-1.0, -1.0, -1.0,  5.0, -1.0, -1.0]]),
    "Unsure but right" : torch.tensor([[ 0.5,  0.4,  0.3,  0.6,  0.2,  0.1]]),
    "Unsure and wrong" : torch.tensor([[ 0.6,  0.5,  0.4,  0.3,  0.2,  0.1]]),
    "Confidently WRONG": torch.tensor([[ 5.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
}

for name, logits in scenarios.items():
    print(f"{name:<20} loss = {criterion(logits, true).item():.3f}")
# Confidently right    loss = 0.005
# Unsure but right     loss = 1.690
# Unsure and wrong     loss = 1.790
# Confidently WRONG    loss = 6.005   ← huge penalty
```

This is exactly why you should be careful when telling a customer "there's definitely no charge" — confident wrong answers hurt the most, in models *and* in business.

---

## Step 6 — Backpropagation & gradients: figuring out what to change

Now we know the model was off by `0.86`. But the model has *millions* of internal knobs (parameters). Which ones do we turn? And which direction?

That's what **backpropagation** answers. It works backwards through the model and produces, for every single knob, a tiny number called a **gradient** that says: *"turn this knob this much in this direction to reduce the loss."*

> **Real-life analogy.** A pastry chef bakes a cake and it comes out too dense. They don't randomly change everything. They walk back through the recipe and figure out: "the flour was a little too much, the baking powder was a little too low, the oven was a little too cool." Those tiny corrections-per-ingredient are gradients.

You'll never write the math yourself. PyTorch does it with one line:

```python
loss.backward()   # computes gradients for every parameter
```

#### Code: see gradients show up on real parameters

```python
import torch
import torch.nn as nn

# Tiny model that scores an OrbitMart email into 6 teams.
# Pretend the email has already been turned into 4 numbers (its embedding).
classifier = nn.Linear(in_features=4, out_features=6)

email_vec  = torch.tensor([[0.2, -0.1, 0.7, 0.3]])   # "my order is late"
true_team  = torch.tensor([0])                       # order_status

logits = classifier(email_vec)
loss   = nn.CrossEntropyLoss()(logits, true_team)
loss.backward()

print("weight gradient shape:", classifier.weight.grad.shape)
# torch.Size([6, 4])     ← one number per knob, telling us how to adjust it

print("first row :", classifier.weight.grad[0])
# tensor([ 0.04, -0.02,  0.13,  0.06])
```

Every one of those 24 numbers is a tiny instruction: *"to make this email score higher for order_status next time, nudge me this way."*

---

## Step 7 — The optimizer: actually turning the knobs

The **optimizer** takes the gradients and updates the parameters. The default for modern AI is called **AdamW**, but the idea is simple:

```text
new_parameter = old_parameter  -  (learning_rate × gradient)
```

The **learning rate** controls how big each adjustment is.

- Too big (`0.1`): the model overshoots, loss explodes, training fails.
- Just right (`0.0003`): steady improvement.
- Too small (`0.00000001`): the model barely moves, training takes forever.

> **Real-life analogy.** Adjusting a shower temperature. Tiny twists of the knob = takes forever to find the right temp. Huge twists = you alternate between freezing and scalding. Goldilocks twists = comfortable shower in seconds.

```python
optimizer.step()        # apply the update
optimizer.zero_grad()   # clear gradients before next batch
```

#### Code: one full "guess → check → adjust" cycle on an OrbitMart email

```python
import torch
import torch.nn as nn

classifier = nn.Linear(4, 6)
optimizer  = torch.optim.AdamW(classifier.parameters(), lr=3e-4)
loss_fn    = nn.CrossEntropyLoss()

email_vec = torch.tensor([[0.2, -0.1, 0.7, 0.3]])   # "where is my order?"
true_team = torch.tensor([0])                       # order_status

for step in range(5):
    optimizer.zero_grad()                # 1. clear yesterday's notes
    logits = classifier(email_vec)       # 2. guess
    loss   = loss_fn(logits, true_team)  # 3. check
    loss.backward()                      # 4. figure out what to fix
    optimizer.step()                     # 5. actually fix it
    print(f"step {step}  loss={loss.item():.4f}")
# step 0  loss=2.0117
# step 1  loss=2.0086
# step 2  loss=2.0055
# step 3  loss=2.0024
# step 4  loss=1.9994   ← going down, learning is happening
```

That's it. That's the entire heart of training. Repeat for thousands of emails and watch the loss drop.

---

## Step 8 — Loop it: epochs and batches

You don't show the model one email at a time, and you don't show all of them at once. You show them in **batches** (e.g. 32 emails at a time). When you've seen every email in your dataset once, that's one **epoch**. You usually train for several epochs.

```text
1 epoch over 1,000 emails with batch_size = 32
   = 32 batches
   = 32 rounds of (guess → check → adjust)

5 epochs = 160 rounds total
```

> **Real-life analogy.** Studying for an exam. You don't read the whole textbook in one shot. You read a chapter (a batch), test yourself, then move on. After finishing the textbook once, you start over (next epoch) — and each pass you understand a little more.

#### Code: batches of OrbitMart emails with a DataLoader

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Pretend each email has been turned into 4 numbers, and we have 5 emails.
email_vecs = torch.tensor([
    [ 0.2, -0.1,  0.7,  0.3],   # "where is my order?"
    [ 0.5,  0.0, -0.2,  0.8],   # "refund please"
    [-0.1,  0.4,  0.6, -0.3],   # "charger broken"
    [ 0.9,  0.2,  0.1,  0.0],   # "change shipping address"
    [ 0.1,  0.1,  0.1,  0.1],   # "do you sell HDMI cables?"
])
labels = torch.tensor([0, 3, 2, 4, 5])   # team indices

loader = DataLoader(TensorDataset(email_vecs, labels), batch_size=2, shuffle=True)

for epoch in range(2):
    print(f"--- epoch {epoch} ---")
    for batch_idx, (x, y) in enumerate(loader):
        print(f"  batch {batch_idx}: x.shape={tuple(x.shape)} y={y.tolist()}")
# --- epoch 0 ---
#   batch 0: x.shape=(2, 4) y=[3, 0]
#   batch 1: x.shape=(2, 4) y=[5, 4]
#   batch 2: x.shape=(1, 4) y=[2]    ← leftover "odd" batch
# --- epoch 1 ---
#   batch 0: x.shape=(2, 4) y=[0, 4]
#   ...                                ← reshuffled this time
```

Notice the shuffle every epoch — that prevents the model from memorizing the order of the emails.

---

## Step 9 — Don't fool yourself: train / validation / test

A model that memorizes its training data is useless on new data. To catch this, split your data into three buckets **before** you start:

| Split | Used for | Real-life equivalent |
|---|---|---|
| **Train** (~70%) | Teach the model | Practice problems with answers |
| **Validation** (~15%) | Tune your choices, watch for overfitting | Mock exams during the semester |
| **Test** (~15%) | Final score — touch ONCE at the very end | The actual final exam |

> **Real-life analogy.** A medical student practices on textbook cases (train), takes mock exams (val), and only sees the real licensing exam once (test). If they peek at the licensing exam answers while practicing, the score becomes meaningless.

**The most common beginner mistake:** repeatedly checking test accuracy while tuning. Don't. The moment you tune to your test set, it stops being a test.

#### Code: split OrbitMart emails into train / val / test

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Pretend we exported 1000 OrbitMart tickets from the helpdesk.
df = pd.DataFrame({
    "text":  [f"sample email {i}" for i in range(1000)],
    "label": ["order_status"] * 600 + ["return_request"] * 200
           + ["billing_issue"] * 100 + ["other"] * 100,
})

# 70 / 15 / 15 split, keeping label proportions equal in every bucket.
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
print("\nlabel mix in val:")
print(val_df["label"].value_counts(normalize=True).round(2))
# train=700  val=150  test=150
#
# label mix in val:
# order_status     0.60
# return_request   0.20
# billing_issue    0.10
# other            0.10
```

The `stratify=` argument is the magic that keeps the rare classes (like `billing_issue`) appearing in every split — otherwise the test set might have zero billing emails by random chance.

---

## Step 10 — Beyond accuracy: precision, recall, F1

"Accuracy" alone can lie to you. Imagine 100 emails: 95 are "order_status", only 5 are "billing_issue". A lazy model that always predicts "order_status" gets 95% accuracy — and is useless for billing.

So we use richer metrics:

| Metric | What it answers | Real-life version |
|---|---|---|
| **Precision** | "Of the ones I flagged as billing, how many really were?" | A spam filter: out of emails it flagged as spam, how many actually were spam? |
| **Recall** | "Of all the real billing emails, how many did I catch?" | A metal detector at airport security: of all the real weapons, how many did it beep on? |
| **F1** | The balance between precision and recall (one number) | A combined report card grade |
| **Macro F1** | F1 averaged across every class equally | Forces the model to do well on rare classes too |

> **Real-life analogy for precision vs recall.** A fishing net.
> - **High precision, low recall** = a small, fancy net. Everything you catch is the right fish, but you miss most of them.
> - **Low precision, high recall** = a giant net dragged across the ocean. You catch every fish, but also seaweed, garbage, and old boots.
> - **Good F1** = a smart net that catches most of the right fish without too much junk.

#### Code: see why "95% accuracy" can be a lie

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

labels = ["order_status", "return_request", "billing_issue",
          "address_change", "technical_issue", "other"]

# 100 OrbitMart tickets, mostly order_status.
truth = ["order_status"] * 80 + ["return_request"] * 10 \
      + ["billing_issue"] * 5  + ["technical_issue"] * 5

# A LAZY model that always guesses "order_status".
lazy_preds = ["order_status"] * len(truth)

print("Accuracy :", accuracy_score(truth, lazy_preds))
print("Macro F1 :", round(f1_score(truth, lazy_preds, average="macro",
                                   labels=labels, zero_division=0), 2))
# Accuracy : 0.80      ← looks great in a dashboard
# Macro F1 : 0.15      ← reveals the model is useless on rare classes

print(classification_report(truth, lazy_preds, labels=labels, zero_division=0))
# billing_issue   recall = 0.00   ← catches ZERO billing tickets
# return_request  recall = 0.00   ← catches ZERO returns
# technical_issue recall = 0.00   ← catches ZERO tech tickets
```

This is exactly why every OrbitMart dashboard should show **macro F1**, not just accuracy. A 95% accurate spam filter that lets through every phishing email is still a disaster.

---

## Step 11 — Two flavors of language modeling

There are two main ways AI models learn from text. You'll meet both.

### A. Causal LM ("predict the next word")

The model reads left-to-right and tries to guess the next word. Used by ChatGPT, Llama, Claude, etc.

```text
"The order arrived on ___"     → model predicts "time"
```

> **Real-life analogy.** Autocomplete on your phone. It only knows what you've typed so far — it can't peek at what you'll type next.

#### Code: let GPT-2 finish an OrbitMart product description

```python
from transformers import pipeline

generate = pipeline("text-generation", model="gpt2")

prompt = "The OrbitMart NovaBuds are wireless earbuds with"
out = generate(prompt, max_new_tokens=20, do_sample=False)[0]["generated_text"]
print(out)
# The OrbitMart NovaBuds are wireless earbuds with a built-in microphone
# and a built-in speaker. They are also compatible with...
```

This is exactly how ChatGPT works under the hood — just a much, much bigger model trained on much more text.

### B. Masked LM ("fill in the blank")

The model sees the full sentence with a word hidden, and guesses the hidden word. Used by BERT and similar "understanding" models.

```text
"The order arrived [MASK] time"   → model predicts "on"
```

> **Real-life analogy.** A crossword puzzle. You can use clues from both sides of the blank to figure out the missing word.

#### Code: let BERT fill in an OrbitMart support email

```python
from transformers import pipeline

fill = pipeline("fill-mask", model="bert-base-uncased")

for guess in fill("my orbitmart charger is not [MASK].")[:3]:
    print(f"  {guess['score']:.2f}  →  {guess['sequence']}")
# 0.71  →  my orbitmart charger is not working.
# 0.09  →  my orbitmart charger is not charging.
# 0.04  →  my orbitmart charger is not connected.
```

BERT looked at *both* sides of the blank to make those guesses. That bidirectional view is why BERT is great at **understanding** text (search, classification, extraction) but cannot generate fluent paragraphs the way GPT can.

| Use it when you want... | Use the model type |
|---|---|
| To **generate** new text (chat, code, writing) | Causal LM (GPT, Llama, Claude) |
| To **understand** existing text (classify, search, extract) | Masked LM (BERT, RoBERTa) |

---

## Putting it all together: the full training loop

This is the entire heart of training, in pseudocode. Read it slowly.

```text
for each epoch:
    for each batch of examples:
        guess     = model(inputs)              # forward pass
        wrongness = loss(guess, true_answer)   # how wrong?
        wrongness.backward()                   # compute gradients
        optimizer.step()                       # turn the knobs
        optimizer.zero_grad()                  # reset for next batch

    check validation loss → are we still improving, or overfitting?
```

Every framework, every model, every modern AI tutorial — they all boil down to this loop. That's it.

---

## Practical setup (5 minutes)

```bash
python -m venv .venv
source .venv/bin/activate         # macOS/Linux
# .venv\Scripts\activate          # Windows

pip install --upgrade pip
pip install torch transformers datasets scikit-learn pandas
```

For API keys (OpenAI, Anthropic, Hugging Face), see the brief setup in [day_zero_prerequisites.md](./day_zero_prerequisites.md). Short version: put them in a `.env` file in your project, never paste them into code, never commit them to git.

---

## A complete, runnable mini-example

This single script ties together every idea above into a tiny working email classifier. Save it as `ticket_classifier.py` and run it.

```python
"""
OrbitMart Support Ticket Classifier — the entire foundations chapter
in one runnable file.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter

# 1. Tiny dataset (in real life: thousands of emails)
texts = [
    "Where is my delivery? Shipped 3 days ago.",
    "When will my package arrive?",
    "Track my order 88421 please.",
    "I want to return my headphones.",
    "How do I return a product?",
    "Return request for order 1122.",
    "My laptop dock is not detected.",
    "The charger stopped working.",
    "Bluetooth keeps disconnecting.",
    "Update my delivery address please.",
    "Wrong address on my order.",
    "Change address to 123 Main St.",
    "I was charged twice for order 88421.",
    "Why is there an extra charge?",
    "Refund for duplicate payment please.",
    "What colors do the NovaBuds come in?",
    "Do you sell USB-C to HDMI adapters?",
    "What are your store hours?",
]
labels = [
    "order_status", "order_status", "order_status",
    "return_request", "return_request", "return_request",
    "technical_issue", "technical_issue", "technical_issue",
    "address_change", "address_change", "address_change",
    "billing_issue", "billing_issue", "billing_issue",
    "other", "other", "other",
]

# 2. Map labels to integers
label_names = sorted(set(labels))
label2id = {n: i for i, n in enumerate(label_names)}

# 3. Train / validation split
train_x, val_x, train_y, val_y = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# 4. Tiny tokenizer (turns words into integers)
def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in counter:
        vocab[w] = len(vocab)
    return vocab

def encode(text, vocab, max_len=20):
    ids = [vocab.get(w, 1) for w in text.lower().split()][:max_len]
    return ids + [0] * (max_len - len(ids))

vocab = build_vocab(train_x)

# 5. PyTorch dataset
class TicketDS(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (
            torch.tensor(encode(self.X[i], vocab)),
            torch.tensor(label2id[self.Y[i]]),
        )

# 6. Tiny model: embed → average → linear classifier
class TinyClassifier(nn.Module):
    def __init__(self, vocab_size, dim, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.head = nn.Linear(dim, n_classes)
    def forward(self, x):
        v = self.embed(x)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (v * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)

model = TinyClassifier(len(vocab), 64, len(label_names))
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 7. Train
loader = DataLoader(TicketDS(train_x, train_y), batch_size=4, shuffle=True)
for epoch in range(40):
    model.train()
    total = 0
    for x, y in loader:
        optim.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optim.step()
        total += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:>2}  loss={total/len(loader):.3f}")

# 8. Evaluate
model.eval()
preds, gold = [], []
with torch.no_grad():
    for x, y in DataLoader(TicketDS(val_x, val_y), batch_size=8):
        preds.extend(model(x).argmax(-1).tolist())
        gold.extend(y.tolist())

print("\nAccuracy :", round(accuracy_score(gold, preds), 2))
print("Macro F1 :", round(f1_score(gold, preds, average="macro", zero_division=0), 2))
print(classification_report(gold, preds, target_names=label_names, zero_division=0))
```

Every concept in this chapter — tokens, embeddings, batches, forward pass, loss, backpropagation, optimizer, train/val split, accuracy/F1 — appears in those ~80 lines.

---

## Are you ready for the next chapter?

You should be able to answer these in plain English. No code required.

1. **Why do we tokenize text instead of feeding raw sentences to the model?**
   *Computers only understand numbers, so we chop text into pieces and assign each piece an ID.*

2. **What's an embedding, in one sentence?**
   *A list of numbers that represents the "meaning" of one token, so similar words end up with similar lists.*

3. **What does loss measure?**
   *A single number that says how wrong the model's guess was. Lower is better.*

4. **What's the difference between training and validation data?**
   *The model learns from training data. Validation data is held back so you can check if it actually generalizes — or just memorized.*

5. **Why might 95% accuracy still be a bad model?**
   *If 95% of examples belong to one class, a lazy model can hit 95% accuracy by always predicting that class. Macro F1 catches this.*

6. **Causal LM vs masked LM in one sentence each?**
   *Causal = predict the next word using only the past (chat models). Masked = fill in a missing word using both sides (understanding models).*

If those feel comfortable, you're ready.

---

## Common beginner mistakes (read these once)

| Mistake | Why it hurts |
|---|---|
| Looking at test accuracy while tuning | Quietly turns the test set into training |
| Only checking accuracy | Hides failures on rare/important classes |
| Making the model bigger before fixing the data | The data is almost always the real problem |
| Not reading misclassified examples by hand | Numbers tell you *that* something is wrong, not *why* |
| Hardcoding API keys | Leaks the moment you push to GitHub |

---

## What to study next

- [01_fine_tuning_models.md](./01_fine_tuning_models.md) — take a strong pretrained model and adapt it to your task
- [02_training_models.md](./02_training_models.md) — go deeper into the full training stack
- [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) — give your model a search engine instead of retraining it

---

## References

- PyTorch docs — <https://docs.pytorch.org/docs/stable/torch.html>
- Hugging Face quickstart — <https://huggingface.co/docs/transformers/quicktour>
- Hugging Face tokenizers — <https://huggingface.co/docs/transformers/tokenizer_summary>
