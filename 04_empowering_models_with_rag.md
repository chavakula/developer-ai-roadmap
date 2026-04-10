# 04 — Empowering Models with RAG  
**Goal:** make models useful on private, fresh, and domain-specific knowledge without changing weights  
**Case study:** OrbitMart return policies, product manuals, and supplier docs  
**Updated:** 2026-04-10

---

## What RAG is

RAG stands for **Retrieval-Augmented Generation**.

The basic idea is simple:

```text
user question
 -> retrieve relevant external context
 -> give context to the model
 -> generate grounded answer
```

Instead of hoping the model memorized the answer during pretraining, you fetch the answer source at runtime.

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