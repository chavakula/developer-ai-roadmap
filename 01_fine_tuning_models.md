# 01 — Fine-Tuning Models  
**Goal:** learn how to adapt pretrained models correctly for real tasks  
**Case study:** OrbitMart support replies, invoice extraction, and analytics SQL generation  
**Updated:** 2026-04-10

---

## What fine-tuning is

Fine-tuning means taking a pretrained model and **adapting** it to your task or behavior requirements.

Examples:
- teach a model to reply in your company’s brand voice
- improve structured extraction quality
- align responses to preferences
- improve multi-step reasoning on a narrow task
- reduce cost by pushing good behavior into a smaller model

---

## The most important question first

Before fine-tuning, ask:

> **Do I really need fine-tuning, or would prompting or RAG solve this more cheaply and safely?**

This question alone saves teams a huge amount of wasted effort.

---

## Decision framework: prompt vs RAG vs SFT vs DPO vs RFT

| Need | Best first move | Why |
|---|---|---|
| Better instructions on a known task | Prompting | cheapest and fastest |
| Private or fresh knowledge | RAG | changes knowledge without changing weights |
| Consistent formatting or style | SFT | teaches repeatable output patterns |
| Preference alignment between good/better responses | DPO | learns from chosen vs rejected answers |
| Complex reasoning where a grader can score outcomes | RFT | optimizes behavior through rewards |
| Smaller model that imitates a stronger pipeline | Distillation + SFT | pushes behavior down into cheaper models |

### Practical rule
- **knowledge problem** -> RAG  
- **behavior problem** -> fine-tuning  
- **workflow problem** -> orchestration/agents  
- **ambiguous** -> start with prompting + evals

---

## Current best-practice snapshot

The current official guidance emphasizes:

- **evals first**
- start with **SFT**
- use **SFT + DPO** together when preference data exists
- use **RFT** when you can grade outcomes well
- use **graders** carefully and monitor for reward hacking
- for open models, start with **LoRA/PEFT** before full-weight tuning

### Why evals first matters
If you do not know what “better” means, you cannot optimize reliably.

---

## Fine-tuning data principles

### 1. Write down the task clearly
Bad:
> “Make the model better at customer support.”

Good:
> “Given a support email, produce a JSON object with intent, urgency, and reply draft in OrbitMart’s tone.”

### 2. Build an error taxonomy
For example:
- wrong intent
- incomplete extraction
- invalid JSON
- too verbose
- unsafe policy violation
- hallucinated order facts

### 3. Start small but high quality
Current guidance recommends starting with a small, clean set of good demonstrations rather than rushing into huge noisy datasets.

### 4. Keep prompts in the data
When examples are few, include the system/developer behavior you actually want inside the examples.

### 5. Preserve a held-out test set
Never train on everything.

---

## Tutorial 1 — Supervised fine-tuning for brand-consistent support replies

### Business problem
OrbitMart wants support drafts that:
- sound calm and professional
- do not over-promise refunds
- request missing order details politely
- produce consistent structure

### Why SFT fits
You already know what a good answer looks like.  
You have examples of correct outputs.

---

## Training data shape

### Example chat-style training item
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are OrbitMart support. Be concise, empathetic, and policy-compliant."
    },
    {
      "role": "user",
      "content": "My earbuds stopped charging after two weeks. Can you replace them?"
    },
    {
      "role": "assistant",
      "content": "I'm sorry you're dealing with that. I can help. Please send your order number and a photo of the charging case contacts. Once we verify the order and warranty window, we'll guide you through replacement or return options."
    }
  ]
}
```

### What good examples should show
- tone
- structure
- policy boundaries
- what to do when information is missing
- what **not** to say

---

## Data collection workflow

### Step 1 — Pull source conversations
Choose conversations that:
- were successfully resolved
- represent common intents
- are written by high-quality agents

### Step 2 — Clean them
Remove:
- PII
- sensitive IDs
- unrelated thread noise
- inconsistent policy versions

### Step 3 — Normalize them
Standardize:
- greeting style
- closing style
- policy language
- allowed promises

### Step 4 — Label slices
Examples:
- damaged item
- delayed shipment
- billing issue
- return request
- missing information
- angry customer

### Step 5 — Create train/val/test splits
Make sure rare slices appear in validation and test too.

---

## Hosted SFT workflow example

> Use the currently supported SFT base models listed in the fine-tuning docs/dashboard. The official SFT guide currently shows `gpt-4.1-*` snapshots in examples.

### Upload data and start a job
```python
from openai import OpenAI
client = OpenAI()

job = client.fine_tuning.jobs.create(
    training_file="file-REPLACE_WITH_YOUR_TRAIN_FILE",
    model="gpt-4.1-nano-2025-04-14"
)

print(job.id)
```

### Poll until complete
```python
job = client.fine_tuning.jobs.retrieve("ftjob-...")
print(job.status)
print(job.fine_tuned_model)
```

### Use the tuned model
```python
response = client.responses.create(
    model="ft:gpt-4.1-nano-2025-04-14:your-org:orbitmart-support",
    input="Customer says: My monitor arrived cracked. Write a support reply."
)
print(response.output_text)
```

---

## Eval plan for this SFT job

### Offline rubric
Score each answer on:
- tone compliance
- policy compliance
- helpfulness
- completeness
- hallucination rate
- unnecessary verbosity

### Good eval set slices
- friendly customer
- angry customer
- missing order ID
- out-of-policy request
- warranty edge case
- multilingual inquiry

### Watch for these failures
- invented refund approvals
- invalid policy claims
- too much empathy but no action
- structure drift
- saying “replacement approved” when only troubleshooting is allowed

---

## Tutorial 2 — Open-model LoRA fine-tuning for invoice extraction

### Business problem
OrbitMart receives supplier invoices in inconsistent formats.  
The finance team wants structured output:

```json
{
  "supplier_name": "...",
  "invoice_number": "...",
  "invoice_date": "...",
  "currency": "...",
  "line_items": [...],
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0
}
```

### Why LoRA fits
- you want a domain-specific adaptation
- you may not want or need full-parameter training
- you want to reduce GPU cost
- you want faster experimentation

---

## Why PEFT/LoRA is the default open-model entry point

LoRA updates only small trainable adapter matrices instead of the whole model.

That means:
- less memory
- less storage
- faster runs
- easier experimentation

### When LoRA is especially attractive
- 7B to low tens-of-billions parameter models
- moderate dataset size
- instruction tuning
- extraction/summarization/classification
- teams with limited GPU budget

---

## Example extraction training row

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Extract the invoice fields as valid JSON only.\n\nINVOICE TEXT:\nSupplier: Delta Components Ltd\nInvoice No: DC-8842\nDate: 2026-02-03\nItem: USB-C controller chips x 120 @ 3.20\nTax: 46.08\nTotal: 430.08"
    },
    {
      "role": "assistant",
      "content": "{\"supplier_name\":\"Delta Components Ltd\",\"invoice_number\":\"DC-8842\",\"invoice_date\":\"2026-02-03\",\"currency\":null,\"line_items\":[{\"description\":\"USB-C controller chips\",\"quantity\":120,\"unit_price\":3.2}],\"subtotal\":384.0,\"tax\":46.08,\"total\":430.08}"
    }
  ]
}
```

---

## Open-model LoRA training pattern with TRL + PEFT

> Replace the base model with a model you are licensed to use and can fit on your hardware.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

base_model = "REPLACE_WITH_YOUR_INSTRUCT_MODEL"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

args = TrainingArguments(
    output_dir="orbitmart-invoice-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    bf16=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,   # `tokenizer=` is deprecated in TRL v0.9+; use processing_class
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=args
)

trainer.train()
model.save_pretrained("orbitmart-invoice-lora")
tokenizer.save_pretrained("orbitmart-invoice-lora")
```

### What you should learn here
- adapter setup
- hyperparameter sensitivity
- evaluation on valid JSON and numeric correctness
- cost/quality tradeoffs

---

## Tutorial 2b — QLoRA: fine-tuning large models on consumer hardware

### Why QLoRA matters
LoRA alone still requires loading full-precision model weights.  
For a 7B model that is around 14 GB in bf16 — difficult on a single consumer GPU.

**QLoRA** (Quantized LoRA) loads the base model in **4-bit NF4 quantization**, which cuts memory roughly 4×.  
You then attach LoRA adapters in higher precision (bf16) on top.  
This means a 7B model can be fine-tuned on a 16 GB GPU.

### What you need
```bash
pip install bitsandbytes accelerate peft trl transformers
```

### Example: QLoRA invoice extractor (consumer GPU)

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

base_model = "REPLACE_WITH_YOUR_INSTRUCT_MODEL"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NF4 is the recommended QLoRA quant type
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,    # double quantization saves another ~0.4 bits/param
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)

# Required step before attaching LoRA adapters to a quantized model
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# typical output: trainable params: 6M || all params: 3.7B || trainable%: 0.16

dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

args = TrainingArguments(
    output_dir="orbitmart-invoice-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=args,
)

trainer.train()
model.save_pretrained("orbitmart-invoice-qlora")
tokenizer.save_pretrained("orbitmart-invoice-qlora")
```

### Key differences from plain LoRA
| Aspect | LoRA | QLoRA |
|---|---|---|
| Base model precision | bf16 / fp16 | 4-bit NF4 |
| Memory saving | moderate | high (3–4×) |
| `prepare_model_for_kbit_training` | not needed | required |
| GPU requirement (7B model) | ~14 GB | ~6–8 GB |
| Quality cost | none | small and often negligible |

### When to use QLoRA
- 7B–13B models on a single 16 GB GPU or smaller
- rapid iteration with low hardware budget
- when memory is the bottleneck, not quality

### When plain LoRA is enough
- models already fit in memory
- you want simpler dependency stack (no `bitsandbytes`)
- running on hardware where 4-bit ops are not supported

---

## Extraction eval checklist

For structured extraction, measure more than “looks okay”.

### Track:
- exact match for invoice number
- exact match for date
- numeric tolerance for totals
- valid JSON rate
- schema-pass rate
- line-item recall
- hallucinated-field rate

### Real-world warning
A model that produces beautiful JSON but the wrong totals is still a bad finance system.

---

## Tutorial 3 — DPO for response preference alignment

### Business problem
OrbitMart has multiple acceptable support answers, but some are better:
- shorter
- calmer
- more policy-compliant
- less repetitive
- more direct

You have preference pairs:
- **chosen** response
- **rejected** response

### Why DPO fits
DPO learns from pairwise preferences rather than plain demonstrations.

---

## Example preference record

```json
{
  "input": "Customer: Your refund process is too slow. This is unacceptable.",
  "preferred_output": "I'm sorry for the delay. I checked your case and the refund is currently in review. I'll outline the remaining steps and what you can expect next.",
  "non_preferred_output": "We are processing many refunds right now. Please wait."
}
```

### What DPO teaches the model
Not just “what is acceptable,” but “what is preferred among valid options.”

---

## OpenAI DPO API shape

Use the DPO method in the fine-tuning job request.

```python
from openai import OpenAI
client = OpenAI()

job = client.fine_tuning.jobs.create(
    training_file="file-all-about-the-weather",
    model="REPLACE_WITH_CURRENT_DPO_ELIGIBLE_MODEL",
    method={
        "type": "dpo",
        "dpo": {
            "hyperparameters": {"beta": 0.1}
        }
    }
)
print(job.id)
```

### Important workflow advice
A strong pattern is:
1. do SFT first on preferred outputs
2. then run DPO on chosen/rejected pairs

That gives DPO a better base to refine.

---

## Open-source DPO pattern with TRL

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer

base_model = "REPLACE_WITH_YOUR_INSTRUCT_MODEL"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

dataset = load_dataset("json", data_files={"train": "dpo_train.jsonl", "validation": "dpo_val.jsonl"})

args = TrainingArguments(
    output_dir="orbitmart-dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    report_to="none"
)

trainer = DPOTrainer(
    model=model,
    args=args,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

trainer.train()
```

### DPO failure modes
- preference pairs are inconsistent
- “rejected” answers are still too good
- pairs do not represent the real traffic
- preference policy changes mid-dataset

---

## Tutorial 4 — RFT for SQL generation with graders

### Business problem
OrbitMart wants a BI assistant that converts analytics questions into SQL.

Examples:
- “What were laptop returns by week last month?”
- “Show top 10 products by net revenue in Q1.”
- “Which suppliers had delayed purchase orders above 7 days?”

### Why RFT fits
There is often not a single exact text answer, but there **is** a way to score correctness:
- SQL runs
- returns expected columns
- produces correct result set
- respects table access limits

---

## RFT mindset

In reinforcement fine-tuning:
- the model generates samples
- graders score the samples
- the reward signal guides learning

### Good RFT use cases
- reasoning tasks
- tool-calling tasks
- SQL/code-like tasks
- tasks with verifiable outcomes

### Bad RFT use cases
- vague tasks with no clear grader
- tasks where evaluation is almost entirely subjective
- tasks with hidden safety issues that graders miss

---

## Example RFT dataset item concept

```json
{
  "prompt": "Show total refund amount by category for March 2026.",
  "reference_answer": {
    "must_use_tables": ["refunds", "orders", "products"],
    "expected_columns": ["category", "total_refund_amount"]
  }
}
```

---

## Example grader ideas

### String check grader
Useful for:
- exact JSON keys
- exact function names
- exact tool names

```json
{
  "type": "string_check",
  "name": "table_guard",
  "operation": "like",
  "input": "refunds",
  "reference": "{{ sample.output_text }}"
}
```

### Python grader
Useful for:
- execute SQL safely in a sandbox
- compare returned rows to expected result
- validate schema
- assign partial credit

```python
grading_function = '''
import sqlite3

def grade(sample, item) -> float:
    sql = sample["output_text"]
    if "drop " in sql.lower() or "delete " in sql.lower():
        return 0.0

    expected_columns = item["expected_columns"]

    # In real usage, run against a sandbox DB
    # and compare columns / results.
    for col in expected_columns:
        if col.lower() not in sql.lower():
            return 0.2

    return 1.0
'''
grader = {"type": "python", "source": grading_function}
```

---

## Reward hacking warning

A model can learn to **game the grader**.

Example:
- it learns that including certain phrases boosts reward
- it outputs a tool name correctly but wrong arguments
- it returns SQL that matches the keyword test but not the actual query intent

### How to defend against this
- combine multiple graders
- manually inspect outputs
- compare grader results with human review
- use held-out eval sets
- include adversarial examples

---

## Fine-tuning hyperparameters: what beginners should care about

### Learning rate
Too high:
- unstable training
- style drift
- sudden collapse

Too low:
- no visible improvement

### Epochs
Too many:
- overfitting
- memorization
- reduced generalization

### Batch size
Larger effective batch sizes can stabilize training, but they cost memory.

### Gradient accumulation
Lets you simulate bigger batches when GPU memory is limited.

### Sequence length
Longer sequences may improve context learning, but increase cost.

---

## Hosted vs open-model fine-tuning: when to choose which

| Situation | Better fit |
|---|---|
| You want fast time-to-value | hosted |
| You want less infra work | hosted |
| You want full weight/control access | open |
| You need custom training code or libraries | open |
| You want to experiment with PEFT adapters locally | open |
| You want strong built-in product integration | hosted |

---

## Fine-tuning project ladder

### Project A — Brand voice support model
Goal: better reply consistency.

### Project B — Invoice extractor
Goal: valid JSON and correct numeric fields.

### Project C — Preference-aligned response ranker
Goal: reduce verbose or defensive answers.

### Project D — SQL assistant with graders
Goal: verifiable correctness on analytics questions.

---

## Fine-tuning checklist before production

- baseline prompt or RAG system exists
- held-out evaluation set exists
- failure taxonomy exists
- PII removed or governed
- policy version frozen for labeling
- evaluator rubric defined
- safety review completed
- cost/latency measured
- rollback path defined

---

## Common mistakes

### Mistake 1: trying to fix missing knowledge with fine-tuning
Use RAG for private/fresh knowledge.

### Mistake 2: training on noisy historical outputs
Your model will learn your organization’s bad habits too.

### Mistake 3: skipping a baseline
You need to know whether tuning actually helped.

### Mistake 4: using only one metric
Especially for extraction and agent/tool tasks, one metric is never enough.

### Mistake 5: ignoring data slices
A model can look good overall and still fail badly on the most important business slice.

---

## What you should do after this guide

If you want deeper control over the optimization process, go next to:
- [02_training_models.md](./02_training_models.md)

If your problem is mostly private or changing knowledge, go next to:
- [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md)

---

## References

- OpenAI evaluation best practices: <https://developers.openai.com/api/docs/guides/evaluation-best-practices>
- OpenAI supervised fine-tuning: <https://developers.openai.com/api/docs/guides/supervised-fine-tuning>
- OpenAI DPO: <https://developers.openai.com/api/docs/guides/direct-preference-optimization>
- OpenAI RFT: <https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning>
- OpenAI graders: <https://developers.openai.com/api/docs/guides/graders>
- PEFT LoRA docs: <https://huggingface.co/docs/peft/package_reference/lora>
- TRL docs: <https://huggingface.co/docs/trl/index>