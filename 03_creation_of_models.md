# 03 — Creation of Models  
**Goal:** move from “using models” to “designing model artifacts”  
**Case study:** OrbitMart manuals, support logs, and custom tokenizer/model packaging  
**Updated:** 2026-04-10

---

## Plain English: what this chapter is about

So far you have **used** models that someone else built. This chapter is about **building your own model package**.

Helpful analogy: shipping a model is like shipping a mobile app, not just the binary. An app needs an icon, a settings screen, install instructions, a version, and screenshots. A model package needs a tokenizer, a config, weights, a class that knows how to load them, and a model card telling people how to use it.

```text
JUST WEIGHTS  (not enough)              A REAL MODEL PACKAGE  (what we ship)
┌──────────────┐                        ┌────────────────────────────────┐
│ pytorch_model│                        │  tokenizer/   ← splits text   │
│   .bin       │                        │  config.json  ← architecture  │
└──────────────┘                        │  model.safetensors ← weights  │
   "What is this?                       │  modeling_*.py ← class code   │
    How do I run it?"                   │  README.md     ← model card   │
                                        │  generation_config.json       │
                                        └────────────────────────────────┘
                                              loaded with one line:
                                              AutoModel.from_pretrained(...)
```

A usable model is a **package**, not just a weight file.

---

## Mini-glossary: jargon in this chapter

| Term | One-line meaning |
|---|---|
| Tokenizer | Splits text into IDs the model can read. |
| Vocabulary | The fixed list of tokens the tokenizer knows. |
| BPE / WordPiece / Unigram | Three ways tokenizers learn how to split words. |
| Special token | Reserved IDs like `[CLS]`, `[PAD]`, `<eos>` with a defined role. |
| Config | A small JSON describing model shape (layers, hidden size, vocab size, etc.). |
| Architecture | The class of layers (e.g. transformer decoder) the weights plug into. |
| Task head | A small layer on top that turns hidden states into the answer (class, span, token). |
| Checkpoint | A saved snapshot of weights (and sometimes optimizer state). |
| `save_pretrained` / `from_pretrained` | Hugging Face convention to save and reload a full package. |
| Model card | A README describing what the model is, how it was trained, and how to use it safely. |
| Hidden size (d_model) | Width of each token's vector inside the model. |
| Num layers / depth | How many transformer blocks are stacked. |
| Context length | Max tokens the model can read at once. |
| Embedding | The lookup table that turns a token ID into a vector. |
| Tying weights | Reusing the embedding matrix as the output projection (saves params). |
| safetensors | A safer, faster file format for weights than `.bin`. |

Read this table once and refer back as terms appear.

---

## The stack you are creating

```text
corpus
 -> tokenizer
 -> config
 -> model class
 -> training code
 -> checkpoint
 -> save_pretrained package
 -> model card
 -> reuse / deployment
```

---

## When this topic matters

You should study model creation if you want to:
- build a domain-specific tokenizer
- modify model depth/width/context length
- add custom task heads
- package a custom architecture for reuse
- experiment with architecture tradeoffs
- understand what is inside a Hugging Face model repo

---

## Tutorial 1 — Train a domain tokenizer

### Business problem
OrbitMart’s internal text has many domain terms:
- USB-C PD 3.1
- GaN charger
- SKU-8842-A
- RMA
- Qi2
- firmware rollback
- PoE injector

A generic tokenizer may split these awkwardly.

### Why tokenizer training helps
A domain tokenizer can:
- reduce fragmentation of frequent domain terms
- improve packing efficiency
- improve downstream learning quality
- reduce token counts for repeated business text

---

## Tokenizer design choices

### Vocabulary size
Larger vocabulary:
- fewer splits for common words
- larger embedding matrix

Smaller vocabulary:
- more token splits
- smaller embedding matrix

### Special tokens
At minimum, decide on:
- `<pad>`
- `<unk>`
- `<bos>`
- `<eos>`
- task-specific markers if needed

### Normalization
Decide how to handle:
- case
- accents
- punctuation
- special symbols
- serial numbers

---

## Hugging Face custom tokenizer training pattern

`train_new_from_iterator` is available on **fast tokenizers** only (backed by the `tokenizers` Rust library).
`GemmaTokenizer` is SentencePiece-based and is **not** a fast tokenizer, so it does not expose this method.
Use any fast tokenizer as an architecture template — `gpt2` is a good, dependency-free choice for BPE.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# gpt2 is a BPE-based fast tokenizer — safe to use as the architecture template
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("text", data_files={"train": "orbitmart_corpus.txt"})["train"]

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]["text"]

trained_tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(),
    vocab_size=32000
)

trained_tokenizer.save_pretrained("./orbitmart-tokenizer")
```

### Alternative: using the `tokenizers` library directly
For full control over normalization, pre-tokenization, and merges, build from scratch:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    vocab_size=32000,
    min_frequency=2,
)

dataset = load_dataset("text", data_files={"train": "orbitmart_corpus.txt"})["train"]

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]["text"]

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
tokenizer.save("./orbitmart-tokenizer/tokenizer.json")
```

> Use the first pattern when you want a drop-in compatible Hugging Face tokenizer.  
> Use the second when you want explicit control over every tokenization decision.

### Corpus suggestions
Train on:
- manuals
- FAQs
- support notes
- invoice text
- policy documents
- product catalogs

Do not train a tokenizer on a tiny sample unless it really represents your future workload.

---

## How to evaluate whether your tokenizer is better

### Measure:
- average tokens per document
- fragmentation of key domain terms
- out-of-vocabulary behavior
- downstream model performance on your real tasks

### Example comparison question
How does each tokenizer split:

```text
USB-C PD 3.1 140W charger with GaN and foldable prongs
```

If your domain tokenizer produces fewer awkward pieces, that is often a good sign.

---

## Tutorial 2 — Build a tiny transformer config

### What a config defines
A config tells the framework how to build the model.

Typical fields:
- `vocab_size`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `intermediate_size`
- `max_position_embeddings`

### Example config class
```python
from transformers import PretrainedConfig

class OrbitConfig(PretrainedConfig):
    model_type = "orbit"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
```

### Why this matters
You are learning that architecture is a set of concrete design decisions, not magic.

---

## Tutorial 3 — Create a custom model class

### Minimal custom model idea
For learning, create a tiny decoder-only model with:
- token embedding
- position embedding
- transformer blocks
- LM head

### Skeleton
```python
import torch
import torch.nn as nn
from transformers import PreTrainedModel

class OrbitTinyLM(PreTrainedModel):
    config_class = OrbitConfig

    def __init__(self, config):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        logits = self.lm_head(x)
        return {"logits": logits}
```

### Important note
This skeleton is **intentionally incomplete**: it has embedding and an LM head but **no transformer blocks**.  
Its purpose is to teach the packaging interface, not to produce a working LLM.

### Adding real transformer blocks

Replace the gap between embeddings and the LM head with stacked `TransformerDecoderLayer` blocks:

```python
import torch.nn as nn
from transformers import PreTrainedModel

class OrbitTinyLMWithBlocks(PreTrainedModel):
    config_class = OrbitConfig

    def __init__(self, config):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_hidden_layers)

        # Causal mask: each position can only attend to itself and earlier positions
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(config.max_position_embeddings),
            persistent=False,
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Decoder in self-attention-only mode: memory == x (standard for decoder-only LMs)
        mask = self.causal_mask[:seq_len, :seq_len]
        x = self.decoder(tgt=x, memory=x, tgt_mask=mask, memory_mask=mask)

        logits = self.lm_head(x)
        return {"logits": logits}
```

> For production-quality implementations, look at the Hugging Face transformers source for GPT-2 or Llama.  
> These show the full attention, RoPE, normalization, and weight-tying details.

---

## Tutorial 4 — Save and reload like a real model package

One powerful Hugging Face convention is `save_pretrained()` and `from_pretrained()`.

### Save
```python
config = OrbitConfig(vocab_size=32000)
model = OrbitTinyLM(config)

model.save_pretrained("./orbit-tiny-lm")
trained_tokenizer.save_pretrained("./orbit-tiny-lm")
```

### Reload
```python
model = OrbitTinyLM.from_pretrained("./orbit-tiny-lm")
tokenizer = trained_tokenizer.from_pretrained("./orbit-tiny-lm")
```

### Why this matters
Reusable model artifacts reduce chaos across:
- training
- evaluation
- deployment
- collaboration

---

## Tutorial 5 — Add task-specific heads

Many real model packages are:
- a shared backbone
- plus one or more heads

### Examples
- classification head
- token classification head
- retrieval embedding head
- regression head

### OrbitMart example
A shared text backbone with:
- ticket intent classifier head
- urgency score head
- language-modeling head

### Why this is useful
It lets you reuse the same learned representation across business tasks.

---

## Tutorial 6 — Build a Hugging Face-compatible package layout

A typical custom package can look like:

```text
orbit_model/
├── configuration_orbit.py
├── modeling_orbit.py
├── tokenization_orbit.py
├── __init__.py
├── config.json
├── tokenizer.json
├── model.safetensors
└── README.md
```

### Minimum packaging goals
- clean load path
- deterministic config
- tokenizer packaged with weights
- model card describing intended use and risks

---

## Model card template

A minimal model card should answer:

### What is this model?
OrbitTinyLM trained on product manuals and support text.

### Intended use
- internal experimentation
- drafting product copy
- classification prototype

### Not intended for
- legal advice
- medical decisions
- autonomous refunds

### Training data
- manuals
- support notes
- product descriptions

### Risks
- hallucinations
- outdated information
- policy mismatch
- bias from historical support interactions

---

## Architecture tradeoffs you should understand

### Depth vs width
- deeper models learn hierarchical abstractions
- wider models can increase capacity
- there is no universal best setting

### Context length
Longer context:
- more memory
- slower training
- sometimes necessary for manuals/contracts/code

### Vocabulary size
Larger vocab:
- bigger embedding matrix
- fewer splits

### Initialization choice
Random initialization:
- expensive to train to usefulness
- useful for education/research

Pretrained initialization:
- better for most practical work

---

## From-scratch training vs continued pretraining vs fine-tuning

### From scratch
Use when:
- new modality or research direction
- very special tokenizer/objective
- serious compute budget

### Continued pretraining
Use when:
- you want stronger domain adaptation at the language-model level
- you have lots of in-domain raw text

### Fine-tuning
Use when:
- you already have a strong base
- you need task behavior more than new world knowledge

---

## Large-scale path: Megatron Core / Bridge

When you move beyond small educational models, you enter the world of:
- tensor parallelism
- pipeline parallelism
- distributed checkpointing
- conversion between ecosystems

The modern NVIDIA path for this is:
- **Megatron Core** for optimized large-scale training primitives
- **Megatron Bridge** for interoperability and conversion workflows

### When to study this
- multi-node large model work
- research or platform teams
- training beyond hobby/prototype scale

---

## Practical OrbitMart project ideas

### Project A — Tokenizer comparison report
Compare a general tokenizer vs your domain tokenizer.

### Project B — Tiny decoder package
Create tokenizer + config + model + save/reload flow.

### Project C — Multi-head support model
Same backbone, separate heads for intent and urgency.

### Project D — model card + packaging
Turn your research code into a reusable internal artifact.

---

## Common mistakes

### Mistake 1: training a tokenizer on the wrong corpus
Your tokenizer should match your future workload.

### Mistake 2: saving weights without config/tokenizer
That makes reuse painful.

### Mistake 3: changing config fields without documenting them
Reproducibility dies quickly here.

### Mistake 4: thinking “custom model” must mean giant research invention
Many useful custom models are small packaging or architecture variations.

---

## When you are ready to move on

You are ready for the next file when you can:
- train a tokenizer
- define a config
- create a model class
- save and reload a package
- explain when to build from scratch vs start from a pretrained base

---

## Next files

To use the model with external knowledge, continue to:
- [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md)

---

## References

- Hugging Face create a custom architecture: <https://huggingface.co/docs/transformers/main/create_a_model>
- Hugging Face custom tokenizers: <https://huggingface.co/docs/transformers/main/custom_tokenizers>
- Megatron Core language models: <https://docs.nvidia.com/megatron-core/developer-guide/latest/models/llms.html>
- Megatron Core overview: <https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/overview.html>