# 02 — Training Models  
**Goal:** understand the full training loop, from tensors to distributed optimization  
**Case study:** OrbitMart ticket routing and product-manual language modeling  
**Updated:** 2026-04-10

---

## Why this topic matters

Fine-tuning lets you adapt models.  
Training teaches you how models actually learn.

If you understand training deeply, you can:
- debug quality issues faster
- estimate memory and compute needs
- scale from one GPU to many GPUs
- reason about throughput and bottlenecks
- create smaller or custom models intelligently

---

## The training stack in one picture

```text
dataset
  -> tokenizer / preprocessing
  -> batches
  -> forward pass
  -> loss
  -> backward pass
  -> optimizer step
  -> scheduler step
  -> validation
  -> checkpoint
  -> profiling / scaling
```

---

## Objectives you must know

### Classification
Predict one or more labels.

Examples:
- ticket intent
- spam vs non-spam
- fraud signal

### Token classification
Predict a label per token.

Examples:
- named entity recognition
- invoice field labeling

### Causal LM
Predict next token.

Examples:
- assistant generation
- code generation
- drafting tools

### Masked LM
Predict masked tokens.

Examples:
- contextual representation learning
- encoder pretraining

---

## Tutorial 1 — Build a clean PyTorch training loop

### Real-world use case
Train OrbitMart’s support ticket intent model from scratch.

This time, we are not focusing on the task.  
We are focusing on the **training loop itself**.

---

## Minimal training loop structure

```python
for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(...)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            ...
```

### What every part does
- `model.train()` activates training behavior like dropout
- `optimizer.zero_grad()` clears old gradients
- `loss.backward()` computes gradients
- `optimizer.step()` applies parameter updates
- `model.eval()` switches to evaluation behavior

---

## Production-style classifier loop

```python
import torch
from sklearn.metrics import f1_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, gold = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        gold.extend(labels.cpu().tolist())

    macro_f1 = f1_score(gold, preds, average="macro")
    return total_loss / len(loader), macro_f1
```

### Why gradient clipping matters
For some models and schedules, gradients can spike.  
Clipping improves stability.

---

## Optimizers and schedulers you should know

### AdamW
The default optimizer you will use often for transformers.

Why:
- works well in practice
- stable baseline
- widely supported

### Learning rate schedulers
Popular options:
- linear decay
- cosine decay
- warmup + decay

### Warmup
Helps avoid unstable early steps, especially in transformer training.

---

## Tutorial 2 — Train a tiny causal language model

### Business problem
OrbitMart wants a domain-specific text model that understands product-manual phrasing:
- ports
- battery specs
- charging standards
- cable compatibility
- return notes

This is a good educational domain because the vocabulary is technical and repetitive.

---

## Data pipeline for causal LM

### Source documents
Use:
- product manuals
- internal product descriptions
- spec sheets
- FAQ answers

### Preprocess
- remove exact duplicates
- normalize broken OCR if needed
- preserve headings when they matter
- chunk into sequences

### Tokenization
Use a subword tokenizer once you move beyond the educational baseline.

---

## Causal LM training objective

Given token IDs:

```text
[101, 205, 33, 78, 910]
```

Create:
- inputs  = `[101, 205, 33, 78]`
- targets = `[205, 33, 78, 910]`

The model predicts each next token.

---

## Hugging Face training flow for causal LM

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("text", data_files={"train": "manuals_train.txt", "validation": "manuals_val.txt"})

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained(model_name)

args = TrainingArguments(
    output_dir="orbitmart-clm",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator
)

trainer.train()
```

### What this teaches you
- tokenization for LM training
- causal LM data collation
- training arguments
- standard evaluation loop

---

## Perplexity: what it means

Perplexity is a common LM metric derived from loss.

Intuition:
- lower perplexity means the model is less surprised by the true next token
- it is useful for comparing training runs
- it does **not** fully capture end-user quality

### Practical warning
A lower perplexity model can still be worse for business use if:
- it hallucinates more
- it is too verbose
- it follows instructions worse

---

## Tutorial 3 — Scale to multiple GPUs with DDP

### Why DDP matters
Single-GPU training hits limits:
- model too large
- batch too small
- training too slow

Distributed Data Parallel (DDP) is the standard first step into multi-GPU training.

### What DDP does
Each GPU:
- gets a copy of the model
- processes a different mini-batch
- synchronizes gradients after backpropagation

---

## Basic DDP launch

```bash
torchrun --nproc_per_node=4 train_ddp.py
```

### Why `torchrun`
It is the standard launch mechanism for modern PyTorch distributed jobs.

---

## DDP training skeleton

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for batch in train_loader:
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()

    dist.destroy_process_group()
```

### DDP gotchas
- use `DistributedSampler`
- save checkpoints carefully, usually only on rank 0
- be careful with random seeds
- logging from every process can become noisy

### Using DistributedSampler correctly

Without `DistributedSampler`, every process would iterate over the **entire** dataset.  
With it, each process sees only its assigned shard.

```python
from torch.utils.data import DataLoader, DistributedSampler

# In your training script, after DDP setup:
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True,
    seed=42,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=per_gpu_batch_size,
    sampler=train_sampler,   # replaces shuffle=True
    num_workers=4,
    pin_memory=True,
)

# Important: call set_epoch each epoch so shuffling is different per epoch
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)   # ensures non-repeating order across epochs
    train_one_epoch(model, train_loader, optimizer, criterion, device)
```

> `set_epoch()` is critical.  
> Without it every epoch sees the same data order on every GPU, reducing effective randomness.

---

## Tutorial 4 — FSDP vs DeepSpeed

When DDP is not enough, the next question is:

> Do I shard model states with FSDP, or use DeepSpeed-style ZeRO approaches?

### FSDP
Good when:
- you want close alignment with PyTorch native distributed primitives
- you want parameter/gradient/optimizer sharding
- you are already in the PyTorch ecosystem

### DeepSpeed
Good when:
- you want advanced ZeRO configurations
- you need certain ecosystem features or workflows
- your team already uses it successfully

### Practical beginner advice
Start with:
1. single GPU
2. DDP
3. FSDP or DeepSpeed only when memory becomes the main blocker

---

## Example Accelerate FSDP config

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_processes: 4
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
```

### Launch
```bash
accelerate launch train.py
```

### Why Accelerate is useful
It gives you a gentler path into:
- FSDP
- DeepSpeed
- mixed precision
- launch config management

---

## Tutorial 5 — Mixed precision and low precision

### Mixed precision
Common choices:
- **bf16**
- **fp16**

Benefits:
- faster training
- lower memory use

### bf16 vs fp16: which to choose

Both use 16 bits, but the bit layout differs:

| Property | bf16 | fp16 |
|---|---|---|
| Exponent bits | 8 (same as fp32) | 5 |
| Mantissa bits | 7 | 10 |
| Max representable value | ~3.4 × 10³⁸ | 65504 |
| Overflow risk | very low | moderate |
| Precision | lower | higher |
| Requires loss scaling | rarely | often |
| Supported hardware | A100, H100, modern Ampere+ | all CUDA GPUs |

**Practical rule:** use `bf16` when your hardware supports it (Ampere generation or newer).  
`fp16` is fine on older hardware but you may need `fp16_opt_level` loss scaling to avoid NaN gradients.

If you see NaN losses with fp16, switch to bf16 or add gradient scaling:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()   # fp16 loss scaler

for batch in train_loader:
    optimizer.zero_grad()
    with autocast(dtype=torch.float16):
        logits = model(batch["input_ids"].to(device))
        loss = criterion(logits, batch["label"].to(device))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

> With bf16 and `autocast(dtype=torch.bfloat16)`, you usually do not need a `GradScaler` at all.

### Low precision / FP8 ecosystem
The current stack now includes stronger support for lower precision training paths, but do not start there as a beginner.  
First become stable in bf16/fp16 training.

### Rule
- beginner: fp32 on CPU/small GPU for learning, then bf16 on capable hardware
- intermediate: bf16 mixed precision
- advanced: explore FP8 and hardware-specific low precision flows

---

## Tutorial 6 — `torch.compile` and profiling

### Why `torch.compile` matters
It can improve training/inference performance, but you must understand:
- graph breaks
- unsupported patterns
- diminishing returns on small models

### Safe mindset
- get correctness first
- then benchmark
- keep the uncompiled baseline

---

## Simple compile pattern

```python
model = MyModel().to(device)
model = torch.compile(model)
```

### Where it helps
- repeated steady-state workloads
- models without too many dynamic graph breaks
- GPU training loops where compute dominates overhead

### Where it may disappoint
- tiny models
- unstable/dynamic Python-heavy code
- workflows dominated by data loading bottlenecks

---

## Profiling checklist

Before blaming the model, inspect:
- GPU utilization
- data loader throughput
- CPU bottlenecks
- sequence length
- batch size
- padding waste
- gradient accumulation overhead
- checkpoint save time

### Practical tools
- PyTorch profiler
- `nvidia-smi`
- throughput logs (samples/sec, tokens/sec)
- memory usage logs

---

## Reproducibility checklist

- set seeds
- log package versions
- log dataset snapshot/hash
- log tokenizer version
- log hyperparameters
- log hardware type
- log exact checkpoint used
- save best checkpoint, not only last checkpoint

---

## Training failure patterns you will actually see

### Validation loss goes up while train loss falls
Likely overfitting.

### Loss is NaN
Possible causes:
- LR too high
- unstable mixed precision
- bad data
- exploding gradients

### Training is slow
Possible causes:
- data loading bottleneck
- too much padding
- batch too small
- CPU preprocessing overhead
- saving too often

### Multi-GPU speedup is disappointing
Possible causes:
- communication overhead
- tiny model
- small batches
- inefficient data pipeline

---

## Project ladder

### Project A — Ticket classifier
Focus: loop correctness, validation, metrics

### Project B — Tiny product-manual CLM
Focus: next-token training and perplexity

### Project C — DDP run
Focus: distributed launch and reproducibility

### Project D — FSDP or DeepSpeed experiment
Focus: memory scaling and checkpoint handling

### Project E — performance report
Focus: compare baseline, mixed precision, compile, and distributed variants

---

## When you are ready to move on

You are ready for the next topic when you can:
- explain the full training loop from memory
- compute and interpret validation metrics
- train on one GPU without confusion
- run a multi-GPU DDP job
- explain when to use FSDP/DeepSpeed
- debug at least one training instability

---

## Next file

If you now want to build your own tokenizer, config, or custom model package, read:
- [03_creation_of_models.md](./03_creation_of_models.md)

---

## References

- PyTorch distributed: <https://docs.pytorch.org/docs/stable/distributed.html>
- PyTorch compiler FAQ: <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html>
- PyTorch optimizers: <https://docs.pytorch.org/docs/stable/optim.html>
- Hugging Face causal LM tutorial: <https://huggingface.co/docs/transformers/tasks/language_modeling>
- Hugging Face masked LM tutorial: <https://huggingface.co/docs/transformers/tasks/masked_language_modeling>
- Accelerate FSDP guide: <https://huggingface.co/docs/accelerate/usage_guides/fsdp>
- FSDP vs DeepSpeed: <https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed>
- Low precision training: <https://huggingface.co/docs/accelerate/usage_guides/low_precision_training>