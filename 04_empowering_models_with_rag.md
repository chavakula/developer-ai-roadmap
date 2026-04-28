# 04 — Empowering Models with RAG  
**Goal:** make models useful on private, fresh, and domain-specific knowledge without changing weights  
**Case study:** OrbitMart return policies, product manuals, and supplier docs  
**Updated:** 2026-04-28

> **First time here?** Read [00_foundations.md](./00_foundations.md) first — it explains embeddings, which are the heart of RAG.

---

## The whole chapter in one paragraph

A pretrained AI model only knows what it saw during training. It does not know your company's return policy, last week's price list, or yesterday's product launch. Instead of retraining the model every time something changes (slow and expensive), **RAG** (Retrieval-Augmented Generation) hands the model an *open book* at answer time: search the right documents → paste the relevant passage into the prompt → let the model write a grounded answer with citations. No retraining required.

> **Real-life analogy.** A new employee on their first day. Without RAG, you ask them "what's our return policy for laptops?" and they guess: *"30 days?"* (wrong, and they don't know it). With RAG, they grab the policy binder off the shelf, flip to the right page, read "laptops: 14 days," and tell you the correct answer — pointing at the page they used. Same employee, but now they have access to the right information at the right time.

### Code teaser: a 25-line RAG pipeline for OrbitMart's policy binder

The simplest possible RAG pipeline, end to end. We embed OrbitMart's policies, retrieve the most relevant one for a customer question, and ask the model to answer using only that snippet.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Our "policy binder" — in production this is hundreds of PDF pages.
policies = [
    "Laptops may be returned within 14 days of delivery in original packaging.",
    "NovaBuds and other earbuds have a 30-day return window for hygiene reasons.",
    "AlphaDock and FlexCharge accessories carry a 1-year limited warranty.",
    "Refunds are processed within 3-5 business days after we receive the return.",
    "Shipping is free on orders over $50; standard delivery takes 2-4 business days.",
]

encoder = SentenceTransformer("all-MiniLM-L6-v2")
policy_vecs = encoder.encode(policies, normalize_embeddings=True)   # shape: [5, 384]

def answer(question: str, top_k: int = 2) -> str:
    q_vec = encoder.encode([question], normalize_embeddings=True)[0]
    scores = policy_vecs @ q_vec                  # cosine similarity
    top    = np.argsort(scores)[::-1][:top_k]
    context = "\n- ".join(policies[i] for i in top)
    # In a real app, send this prompt to GPT/Claude/etc.
    return f"Use ONLY this policy text:\n- {context}\n\nQ: {question}\nA:"

print(answer("How long do I have to return a laptop?"))
# Use ONLY this policy text:
# - Laptops may be returned within 14 days of delivery in original packaging.
# - Refunds are processed within 3-5 business days after we receive the return.
#
# Q: How long do I have to return a laptop?
# A:
```

That's the entire idea of RAG. The rest of the chapter is about doing this *well*: chunking long PDFs, hybrid search, rerankers, and forcing the model to refuse when the binder doesn't have the answer.

---

## Plain English: what this chapter is about

A pretrained LLM only knows what it saw during training. It does not know your company's return policy, last week's price list, or the manual for the product you launched yesterday.

**RAG = give the model an open book at answer time.**

```text
CLOSED-BOOK EXAM (no RAG)              OPEN-BOOK EXAM (RAG)
┌──────────────────────────┐           ┌──────────────────────────┐
│  Question: "What is      │           │  Question: "What is      │
│  OrbitMart's return      │           │  OrbitMart's return      │
│  window for laptops?"    │           │  window for laptops?"    │
│                          │           │                          │
│        ↓                 │           │  Step 1: search policy   │
│   model guesses          │           │   docs → finds:          │
│   from memory            │           │   "Laptops: 14 days"     │
│                          │           │                          │
│   "Probably 30 days?"    │           │  Step 2: model reads     │
│   ❌ might be wrong      │           │   passage + answers:     │
│   ❌ no citation         │           │   "14 days [policy.pdf]" │
└──────────────────────────┘           │   ✅ correct + cited     │
                                       └──────────────────────────┘
```

RAG = **Retrieval-Augmented Generation**: retrieve relevant text → put it in the prompt → let the model write an answer that is grounded in that text.

### The story we'll follow in this chapter

Imagine OrbitMart's customer support team has a binder full of policies: returns, warranties, shipping, RMA. The chatbot they shipped last month keeps **inventing** policies that don't exist ("30-day return on laptops" — actually it's 14). Customers complain, ops yells.

Your job: stop the chatbot from making things up. The fix is RAG.

| Step | Tutorial | Real-life equivalent |
|---|---|---|
| Use a hosted service (one-line setup) | Tutorial 1 (managed RAG) | Hire a service that scans the binder and answers for you |
| Build it yourself with embeddings | Tutorial 2 (self-managed) | Photocopy the binder, file the pages, look up by topic |
| Combine keyword + meaning search and re-rank | Tutorial 3 (hybrid + reranker) | First the librarian narrows the shelf, then a senior agent picks the best page |
| Make sure the model **only** uses the binder | Tutorial 4 (hallucination control) | "Don't quote rules that aren't in the binder" |

By the end the chatbot answers "You have **14 days** to return your laptop" and shows the exact policy paragraph it used.

---

## Mini-glossary: jargon in this chapter

| Term | One-line meaning |
|---|---|
| Chunk | A small piece of a document (a paragraph or two) used as a retrieval unit. |
| Embedding | A vector that represents the meaning of a chunk so similar meanings are nearby. |
| Vector store | A database that stores embeddings and lets you search by similarity. |
| Index | The data structure inside the vector store that makes search fast. |
| Top-k | How many chunks to retrieve per query (e.g. top-5). |
| Semantic search | Search by meaning (using embeddings). |
| Lexical / BM25 search | Search by keyword overlap (classic IR). |
| Hybrid retrieval | Combine semantic + lexical for better recall. |
| Reranker | A second model that re-orders the top-k for higher precision. |
| Grounding | The retrieved text the model is required to base its answer on. |
| Citation | The source pointer the model returns alongside the answer. |
| Hallucination | The model inventing facts not in the retrieved text. |
| Context window | Max tokens the model can read at once (limits how many chunks fit). |
| Recall@k | Fraction of queries where the right chunk is in the top-k. |
| MRR | Mean reciprocal rank of the right chunk in the result list. |
| Faithfulness | Whether the answer is actually supported by the cited chunks. |

Read once, refer back as terms appear.

---

## Why RAG matters

RAG is often the right answer when the problem is:
- private knowledge
- frequently changing knowledge
- large document collections
- need for citations
- auditability

### Examples
- internal policies
- manuals
- knowledge bases
- contracts
- support documentation
- compliance documents

---

## When RAG is a better choice than fine-tuning

Use RAG when:
- facts change often
- source documents matter
- citations matter
- you want to restrict the model to approved documents
- you want updates without retraining

Use fine-tuning when:
- behavior needs to change
- formatting needs to be consistent
- you want a style or decision policy baked in

### In practice
Many strong systems are:
- prompt + RAG
- or fine-tuned model + RAG

---

## RAG pipeline in one picture

```text
documents
 -> chunking
 -> embeddings / indexing
 -> vector or hybrid search
 -> reranking / filtering
 -> response synthesis
 -> citations / grounded answer
```

### Visual: same pipeline, with a real example

```text
1. INGEST (done once, ahead of time)
   ┌──────────────────────────────────────────────────────┐
   │ policy.pdf  "Laptops can be returned within 14 days  │
   │              of delivery in original packaging..."   │
   └──────────────────────────────────────────────────────┘
                          │
                          ▼  chunk into ~300-token pieces
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ chunk 1  │ │ chunk 2  │ │ chunk 3  │ │ chunk 4  │
   │ "Returns"│ │"Laptops: │ │ "Phones: │ │ "RMA..." │
   │          │ │  14 days"│ │  7 days" │ │          │
   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
        │            │            │            │
        ▼            ▼            ▼            ▼  embed each chunk
       [v1]         [v2]         [v3]         [v4]
        │            │            │            │
        └────────────┴─────┬──────┴────────────┘
                           ▼
                   ┌───────────────┐
                   │ vector store  │  (stores vectors + chunk text + metadata)
                   └───────────────┘

2. ANSWER (every time a user asks)
   user: "How long do I have to return my laptop?"
            │
            ▼ embed the question
          [vq]
            │
            ▼  similarity search → top-3 chunks
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ chunk 2  │ │ chunk 1  │ │ chunk 4  │   ← ranked
   │"14 days" │ │"Returns" │ │ "RMA..." │
   └────┬─────┘ └────┬─────┘ └────┬─────┘
        └────────────┼────────────┘
                     ▼
        prompt = system + question + retrieved chunks
                     │
                     ▼
                ┌──────────┐
                │   LLM    │
                └────┬─────┘
                     ▼
        "You have 14 days from delivery. [policy.pdf, p.2]"
```

This is the entire mental model. Everything else in the chapter is **how to make each box reliable**.

---

## Core RAG terms

### Chunk
A piece of a document used for retrieval.

### Embedding
A vector representation used for semantic search.

### Vector store
A database that stores embeddings and metadata.

### Hybrid retrieval
Semantic + keyword retrieval together.

### Reranker / postprocessor
A step that improves the ordering or filtering of retrieved passages.

### Response synthesis
How the final answer is generated from retrieved evidence.

---

## Tutorial 1 — Managed RAG with OpenAI file search

### Real-life analogy

This is **"hire a serviced apartment, don't build the house"**. You upload your binder of PDFs to OpenAI; they handle the chunking, embedding, indexing, search, and answer generation. You write maybe 10 lines of code and have a working system. Tradeoff: less control, ongoing per-query cost, your docs live on their servers.

### Business problem
OrbitMart wants an assistant that answers questions about:
- return policy
- warranty rules
- shipping commitments
- accessory compatibility

The documents already exist as PDFs and markdown files.

### Why managed file search is attractive
You do not need to hand-build:
- chunking pipeline
- embedding pipeline
- vector index lifecycle
- semantic search orchestration

The platform handles a large part of the retrieval path for you.

---

## Step 1 — Create a vector store and upload files

Your knowledge base should include:
- policy PDFs
- markdown help-center docs
- return instructions
- warranty documents
- product compatibility notes

### Why metadata matters
Store metadata like:
- `category = policy | manual | faq`
- `language = en | fr | de`
- `product_family = accessories | laptops | audio`
- `version = 2026-03`

That makes filtering possible later.

### Code: create a vector store and upload files

```python
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# Step 1: create the vector store
vector_store = client.vector_stores.create(
    name="OrbitMart Knowledge Base"
)
print("vector_store_id:", vector_store.id)

# Step 2: upload files with metadata
file_paths = [
    "docs/returns_policy.pdf",
    "docs/warranty_guide.md",
    "docs/accessory_compatibility.md",
]

file_streams = [open(p, "rb") for p in file_paths]

batch = client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=file_streams,
)

print("status:", batch.status)
print("file_counts:", batch.file_counts)

for stream in file_streams:
    stream.close()
```

> The `upload_and_poll` helper uploads all files and waits until indexing is complete.  
> For large corpora, use the async variant and poll `file_batches.retrieve()` on a schedule.

### Attaching metadata per file

The platform attaches metadata when you set file attributes after uploading.  
If you need category/version filters, include those in the file name or store them in a sidecar mapping, and apply filters at query time using the `filters` parameter (shown in Step 2).

---

## Step 2 — Query with file search

### Basic example
```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="What is OrbitMart's policy for returning opened earbuds?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_123"]
    }]
)

print(response.output_text)
```

### Include retrieved results for debugging
```python
response = client.responses.create(
    model="gpt-4.1",
    input="What is OrbitMart's warranty period for charging docks?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_123"],
        "max_num_results": 3
    }],
    include=["file_search_call.results"]
)
```

### Add metadata filters
```python
response = client.responses.create(
    model="gpt-4.1",
    input="Summarize the opened-box return policy for headphones.",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_123"],
        "filters": {
            "type": "in",
            "key": "category",
            "value": ["policy", "faq"]
        }
    }]
)
```

---

## What this managed path is good for

- fast prototypes
- high-quality internal Q&A
- document-grounded assistants
- lower engineering overhead
- teams that want retrieval without building a vector stack first

---

## Tutorial 2 — Self-managed semantic retrieval

### Real-life analogy

This is **"build your own kitchen"**. You handle every step: chop docs into chunks (paragraphs), turn each chunk into an embedding (a list of numbers representing meaning), store them in a vector database, and search by similarity. More work, but full control: your data stays on your machine, no per-query cost, you can swap any piece (embedder, store, retriever).

### Why you may still want self-managed RAG
You may want:
- custom ranking logic
- custom storage
- lower vendor lock-in
- direct vector DB control
- advanced ACL behavior
- custom ingestion or document parsing

---

## Step 1 — Chunk your documents

### Bad chunk
Too long, mixed topics, poor boundaries.

### Better chunk
One coherent section or subsection, such as:
- “Refund timelines”
- “Warranty exclusions”
- “USB-C compatibility notes”

### Chunking heuristics
- start with 300–800 tokens for prose
- use headings and subheadings as boundaries
- keep tables together when possible
- avoid mixing unrelated sections in the same chunk

---

## Step 2 — Store metadata

A retrieved chunk without useful metadata is much harder to use.

### Metadata examples
```json
{
  "doc_id": "policy_returns_v12",
  "title": "Return Policy",
  "section": "Opened items",
  "category": "policy",
  "version": "2026-03",
  "language": "en",
  "product_family": "audio"
}
```

---

## Step 3 — Retrieval example concept

```python
# pseudocode
results = vector_db.search(
    query_embedding=embed(user_query),
    top_k=5,
    filters={"category": ["policy", "manual"]}
)
```

### What to inspect
- top-k relevance
- missing obvious chunks
- duplicate chunks
- wrong metadata
- stale content

---

## Tutorial 3 — Hybrid retrieval and reranking

### Real-life analogy

Hybrid search = **"two librarians at the front desk"**. One librarian only matches **exact words** (BM25 / keyword search) — great when you ask about "SKU-8842-A". The other only matches **meaning** (vector / semantic search) — great when you ask "how do I send back my laptop?". You combine both lists and get the best of both.

Reranker = **"ask a senior agent to pick the best 3 pages from the top 50"**. Initial search is fast but imprecise; the reranker is slow but accurate, so you only run it on a small shortlist.

Pure semantic retrieval is not always enough.

### Example failure
User asks:
> “What is the warranty for USB-C PD 3.1 chargers?”

If the exact term “PD 3.1” is important, keyword signals can help.

### Hybrid retrieval
Combine:
- dense semantic search
- sparse/keyword retrieval

### Reranking / postprocessing
After retrieval, apply:
- similarity cutoff
- metadata replacement
- cross-encoder reranking
- deduplication
- recency weighting

LlamaIndex describes postprocessors as the stage **after retrieval and before response synthesis**.

---

## Prompting the generator with retrieved context

### Good grounded prompt structure
```text
You are an OrbitMart assistant.
Answer only from the retrieved sources below.
If the answer is not supported by the sources, say you do not have enough evidence.

Retrieved context:
[1] ...
[2] ...
[3] ...

Question:
...
```

### Why this works
It makes the model’s job clearer:
- ground answer in evidence
- avoid overconfident guessing
- cite or point back to sources

---

## Tutorial 4 — Hallucination control patterns

### Real-life analogy

This is **"closed-book exam rules"**. You tell the model: *"Only quote facts from the pages I gave you. If the answer isn't in those pages, say 'I don't know'."* Combined with required citations ("point to the page you used"), the model can no longer make up policies.

The difference between a fun demo and a system you can put in front of customers is exactly this: clear rules + required citations + a refusal path when the answer isn't in the documents.

### Pattern 1 — “answer only from evidence”
Good default.

### Pattern 2 — explicit abstention
Tell the model to say:
> “I don’t have enough evidence in the retrieved sources.”

### Pattern 3 — require citations
Helps auditing and debugging.

### Pattern 4 — split retrieval and answer synthesis
Log retrieval output separately before asking the model to answer.

---

## Evaluation checklist for RAG enablement

### Questions to test
- Did retrieval include the right chunk?
- Did the answer stay within evidence?
- Did the model cite the right source?
- Did the answer ignore contradictory evidence?
- Did metadata filters work?

### Useful metrics
- top-k hit rate
- answer relevance
- groundedness / faithfulness
- retrieval precision
- retrieval recall on labeled questions

---

## Real-world RAG use cases you can build

### 1. Policy assistant
Answer employee or customer policy questions.

### 2. Manual assistant
Answer product setup and troubleshooting questions.

### 3. Supplier document assistant
Extract or explain procurement terms from supplier docs.

### 4. Internal knowledge assistant
Ground answers on team wikis and SOPs.

---

## Common mistakes

### Mistake 1: giant chunks
This hurts retrieval quality.

### Mistake 2: retrieving too many passages
Too much context can bury the answer.

### Mistake 3: no metadata
Filtering becomes impossible.

### Mistake 4: expecting retrieval to fix poor prompt design
Generation still matters.

### Mistake 5: not testing freshness/staleness
Old policy docs can create confident but wrong answers.

---

## When you are ready to move on

You are ready for the next file when you can:
- explain embeddings and retrieval clearly
- chunk documents sensibly
- use metadata filters
- distinguish semantic vs hybrid retrieval
- build a grounded Q&A prototype with citations

---

## Next file

To turn RAG into a real production system, continue to:
- [05_building_rag_systems.md](./05_building_rag_systems.md)

---

## References

- OpenAI file search guide: <https://developers.openai.com/api/docs/guides/tools-file-search>
- LlamaIndex node postprocessors: <https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/>
- OpenAI evaluation best practices: <https://developers.openai.com/api/docs/guides/evaluation-best-practices>