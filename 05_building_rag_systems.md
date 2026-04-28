# 05 — Building RAG Systems  
**Goal:** turn retrieval from a prototype into a production-style system  
**Case study:** OrbitMart enterprise knowledge assistant  
**Updated:** 2026-04-28

> **First time here?** Read [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) first — this chapter assumes you've built a basic RAG pipeline already.

---

## The whole chapter in one paragraph

Chapter 04 was a notebook: one PDF, one question, one answer. This chapter is the real world: thousands of documents that change every day, hundreds of users with different permissions, an on-call engineer when things break, and the need to actually *prove* the answers are good. You'll learn how to ingest from real sources (S3, Notion, wikis), keep the index fresh as docs change, enforce who-sees-what, evaluate quality with real metrics, and observe everything when something goes wrong at 2 AM.

> **Real-life analogy.** Chapter 04 was building a single bookshelf in your living room. This chapter is running an entire **public library**: getting deliveries from publishers (ingestion), shelving and re-shelving as new editions arrive (freshness), making sure children can't borrow adult books (access control), tracking what's checked out (logging), measuring whether visitors actually find what they came for (eval), and having a librarian on call when the catalog system crashes (observability).

### Code teaser: production touches for an OrbitMart RAG system

The difference between "works in a notebook" and "works for 10,000 OrbitMart employees" is in the boring details: metadata, permissions, evaluation, and traces. Here is what each looks like in code.

```python
from dataclasses import dataclass, field
from typing import Literal
import time, uuid, json

# 1. CHUNK + METADATA — every chunk knows where it came from and who can see it.
@dataclass
class Chunk:
    id: str
    text: str
    source: str                       # "notion://policies/refunds"
    updated_at: float                 # for freshness checks
    acl: list[Literal["support", "warehouse", "finance", "public"]]
    tags: list[str] = field(default_factory=list)

binder = [
    Chunk("c1", "Laptops: 14-day return window.", "notion://policies/returns",
          time.time(), ["support", "public"], ["returns"]),
    Chunk("c2", "Wholesale pricing tier B: 18% off list.", "sap://pricing/b",
          time.time(), ["finance"], ["pricing"]),     # finance only!
    Chunk("c3", "Warehouse SOP: scan SKU at bay 4 before forklift move.", "wiki://wh/sop",
          time.time(), ["warehouse"], ["sop"]),
]

# 2. PERMISSION-AWARE RETRIEVAL — a customer-support agent must NOT see finance docs.
def retrieve(query: str, user_role: str, k: int = 3) -> list[Chunk]:
    visible = [c for c in binder if user_role in c.acl]
    # In real life: vector search across `visible`. Here we just keyword-match.
    return [c for c in visible if any(w in c.text.lower() for w in query.lower().split())][:k]

# 3. TRACE — log every query for evals + debugging at 2am.
def answer(query: str, user_role: str) -> dict:
    trace_id = str(uuid.uuid4())
    t0 = time.time()
    hits = retrieve(query, user_role)
    response = {
        "trace_id": trace_id,
        "user_role": user_role,
        "query": query,
        "retrieved": [c.id for c in hits],
        "answer": f"Based on {len(hits)} policy snippet(s)...",
        "latency_ms": int((time.time() - t0) * 1000),
    }
    print("TRACE:", json.dumps(response))
    return response

# 4. RUN — same query, two different users, two different result sets.
answer("laptop return",   user_role="support")    # sees c1
answer("wholesale pricing", user_role="support")  # sees nothing (correctly!)
answer("wholesale pricing", user_role="finance")  # sees c2
```

Metadata, ACLs, traces, and a clear input/output contract — those four things turn a notebook into a system. The rest of this chapter scales each of them up.

---

## Plain English: what this chapter is about

The previous chapter was "I have one PDF and one question — can the model answer it?" That is a notebook.

This chapter is "thousands of documents, hundreds of users, the docs change every day, finance can see finance docs but warehouse cannot, and someone is on call when it breaks." That is a **system**.

```text
NOTEBOOK (chapter 04)                  PRODUCT (this chapter)
┌───────────────────────┐              ┌──────────────────────────────────────┐
│ load 1 pdf            │              │ ingest from Notion + S3 + wiki + ... │
│ chunk it              │              │ schedule re-ingest when docs change  │
│ embed                 │              │ permission-aware retrieval           │
│ retrieve top-3        │              │ hybrid search + reranker             │
│ ask LLM               │              │ answer with citations                │
│ print answer          │              │ log every query for evals + tracing  │
└───────────────────────┘              │ alerting when quality drops          │
   one user, one shot                  └──────────────────────────────────────┘
                                          many users, always-on, auditable
```

The skills are different: ingestion, freshness, access control, evaluation, observability.

---

## Mini-glossary: jargon in this chapter

| Term | One-line meaning |
|---|---|
| Ingestion | The job that pulls documents from source systems into your pipeline. |
| Parsing | Turning a raw file (PDF/HTML/DOCX) into clean text + structure. |
| Enrichment | Adding metadata to chunks (source, author, date, tags). |
| Indexing | Writing embeddings + metadata into the vector store. |
| ACL | Access Control List — who is allowed to see this chunk. |
| Multi-tenant | One system serving many customers/teams without leaking data between them. |
| Freshness | How quickly new/updated docs become searchable. |
| Stale chunk | A chunk whose source has changed but the index has not. |
| Reranker | Second-stage model that re-orders top-k for higher precision. |
| Faithfulness / Groundedness | Whether the answer actually follows from the cited chunks. |
| Recall@k / MRR / nDCG | Standard retrieval quality metrics. |
| Trace | A logged record of one user query: retrieved chunks, prompt, answer, latency. |
| Eval set | A fixed list of question/expected-answer pairs used to score the system. |
| Golden answer | The known-correct answer for an eval question. |
| Guardrail | A check that blocks unsafe / off-policy answers. |
| SLA | Service Level Agreement — promised latency / uptime numbers. |

---

## System architecture view

```text
sources
  -> ingestion
  -> parsing
  -> chunking
  -> enrichment / metadata
  -> indexing
  -> retrieval API
  -> generation API
  -> answer + citations
  -> evals + traces + monitoring
```

### Visual: the same architecture, expanded

```text
   SOURCES                INGESTION              INDEX                SERVING                OBSERVABILITY
┌────────────┐         ┌─────────────┐       ┌────────────┐       ┌──────────────┐       ┌──────────────┐
│ Notion     │──┐      │ pull        │       │            │       │ /retrieve    │       │ traces       │
│ S3 PDFs    │──┼─────▶│ parse       │──────▶│ vector     │◀──────│ /rerank      │──────▶│ evals        │
│ Wiki       │──┤      │ chunk       │       │ store      │       │ /generate    │       │ dashboards   │
│ Tickets    │──┘      │ enrich+ACL  │       │ + metadata │       │ /answer      │       │ alerting     │
└────────────┘         └─────────────┘       └────────────┘       └──────────────┘       └──────────────┘
                          ▲                                            │
                          │                                            ▼
                       schedules                                   end users
                       (hourly/daily,                              (chat, support
                        on-change webhooks)                         tools, agents)
```

Every box in this diagram becomes a tutorial below.

### The story we'll follow in this chapter

In chapter 04 you built a RAG **demo**: one PDF, one question, one answer. It worked on your laptop.

Now imagine OrbitMart says: *"Make this work for the whole company."* Suddenly:
- Docs come from **5 different sources** (Notion, S3, wiki, ticket system, Drive).
- Docs **change every day** — yesterday's policy is wrong today.
- **Finance** can see finance docs but **warehouse** must not.
- **Hundreds of users** ask questions every minute.
- When the bot is wrong, you need to know **why** — which chunks did it pull?

This chapter is the journey from the demo to the production system:

| Step | Tutorial | Real-life equivalent |
|---|---|---|
| Pull docs from many sources automatically | Tutorial 1 (ingestion) | A library cart that visits every department daily |
| Cut docs into the right size | Tutorial 2 (chunking) | Decide whether to file by paragraph or by page |
| Combine search methods | Tutorial 3 (retrieval) | Two librarians at the desk (keyword + meaning) |
| Permission-aware search | Tutorial 4 (multi-tenant + ACL) | A library card that only opens certain rooms |
| Re-index when docs change | Tutorial 5 (freshness) | Throw out yesterday's newspaper, file today's |
| Have the model write the final answer | Tutorial 6 (generation) | The senior agent dictates the customer reply |
| Score the system on a fixed test set | Tutorial 7 (evals) | Mystery-shopper QA program |
| Log every query and watch quality drift | Tutorial 8 (observability) | CCTV + monthly QA review |
| Make it cheap and fast | Tutorial 9 (cost / latency) | Optimize the kitchen so dishes arrive hot and on budget |

By the end you have a system you can hand to ops without crossing your fingers.

---

## OrbitMart production use case

OrbitMart wants one internal assistant for:
- customer support agents
- finance ops
- warehouse operations
- product specialists
- legal/compliance reviewers

Knowledge sources include:
- markdown policies
- PDFs
- spreadsheets converted to text
- wiki pages
- product manuals
- release notes
- vendor contracts

That means the RAG system must handle:
- multiple source types
- multiple audiences
- metadata and permissions
- document freshness
- conflicting versions

---

## Tutorial 1 — Design the ingestion pipeline

### Real-life analogy

Ingestion = **"the library acquisitions team"**. Every morning they visit each department, collect anything new (memos, contracts, manuals), strip the staples, photocopy what's needed, label each page with where it came from and who is allowed to read it, and file it in the catalog.

In RAG terms: a scheduled job pulls from each source (Notion, S3, wiki, ticket system), parses each file (PDF → text), tags it with metadata (source, author, date, ACL), and writes it into the vector store. Without this, your search index goes stale within days.

### Step 1 — list your source systems
Example:
- Notion / wiki
- shared drive
- policy repository
- product manual repository
- CRM export
- support macros

### Step 2 — define canonical document schema
Every document should become a normalized internal object.

Example:
```json
{
  "doc_id": "returns_policy_v12",
  "title": "Returns Policy",
  "source_type": "pdf",
  "body_markdown": "...",
  "created_at": "2026-03-01",
  "updated_at": "2026-03-10",
  "department": "support",
  "access_level": "internal",
  "version": "12"
}
```

### Why this matters
Without a canonical schema, multi-source RAG systems become inconsistent and brittle.

---

## Parsing and normalization

### Goals
- preserve headings
- preserve tables when possible
- extract metadata
- remove duplicate boilerplate
- keep source traceability

### Practical advice
Store:
- raw original file
- parsed normalized text
- chunks
- chunk metadata
- indexing timestamp

That gives you re-index flexibility later.

---

## Tutorial 2 — Chunking strategy as a design decision

### Real-life analogy

Chunking = **"how do you cut up a long document for index cards?"** Too small (one sentence per card) and you lose context ("refunds are processed" — by whom? when?). Too big (one whole policy per card) and the search returns way more text than fits in the prompt and the answer drowns in noise.

The sweet spot is usually **a paragraph or two with overlap** — like cutting a recipe book into one-recipe cards rather than one-step or one-chapter chunks.

Chunking is not a boring preprocessing step.  
It changes system quality.

### Recommended first-pass heuristics
- split on headings
- keep sections coherent
- use overlap only when needed
- keep tables or lists intact
- keep source offsets

### Store this per chunk
```json
{
  "chunk_id": "returns_policy_v12::opened_items::0003",
  "doc_id": "returns_policy_v12",
  "title": "Returns Policy",
  "section": "Opened items",
  "text": "...",
  "start_char": 1420,
  "end_char": 1888,
  "updated_at": "2026-03-10",
  "access_level": "internal"
}
```

### Why source offsets matter
They help with:
- citations
- debugging
- document viewers
- grounding explanations

---

## Tutorial 3 — Retrieval architecture

### Base retrieval stage
This stage should answer:
- which chunks are plausible candidates?

### Post-retrieval stage
This stage should answer:
- which of those candidates are actually best?

### Generation stage
This stage should answer:
- how do we produce a final answer from evidence?

### A healthy system separates these three steps.

---

## Retrieval patterns you should know

### Dense retrieval
Good for semantic similarity.

### Sparse / keyword retrieval
Good for:
- exact product names
- error codes
- policy numbers
- SKU strings

### Hybrid retrieval
Often the practical default for enterprise knowledge systems.

### Routed retrieval
Choose different indexes based on:
- user role
- department
- language
- document type
- query intent

---

## Tutorial 4 — Multi-tenant and access control design

### Real-life analogy

ACL = **"a library card that only opens certain rooms"**. Finance staff swipe and the door to finance opens; warehouse staff swipe and only their room opens. The exact same library, but each visitor's catalog search returns only what their card allows.

In code: every chunk is tagged with `acl: ["finance"]` or `acl: ["warehouse", "support"]`. Every search filters by the user's permissions before it computes similarity. Get this wrong and you've leaked private documents — it must be enforced at the **store**, not the prompt.

This is where many demos fail in real organizations.

### Questions you must answer
- Should support see finance docs?
- Should contractors see internal policy drafts?
- Should EU staff see region-specific docs only?
- Should archived policies remain searchable?

### Access-control strategies
- per-index isolation
- per-chunk metadata filters
- tenant-specific vector stores
- role-based filters at query time

### Minimum rule
Never rely only on the model prompt for access control.  
Enforce ACLs in retrieval.

---

## Tutorial 5 — Freshness and re-indexing

### Real-life analogy

Freshness = **"throw out yesterday's newspaper, file today's"**. The return-policy PDF was edited last night; if your index still contains the old version, the bot confidently quotes outdated rules.

Three common patterns: re-index on a **schedule** (every hour), re-index on a **webhook** (the moment a doc changes), or re-index on **read** (lazy). Most production systems use a mix.

### Why freshness matters
RAG fails badly when:
- old policy remains indexed
- old pricing survives
- deprecated process docs outrank new docs

### Strategies
- version all documents
- mark deprecated docs explicitly
- schedule re-indexing
- keep updated timestamps in metadata
- optionally boost recency for change-prone sources

### Useful operational pattern
When a new policy version goes live:
1. ingest new version
2. index it
3. de-rank or archive old version
4. validate top queries on that topic

---

## Tutorial 6 — Response generation layer

The answer generator should know:
- the user’s question
- the retrieved chunks
- the desired answer format
- the citation or source policy
- what to do if evidence is weak

### Good answer requirements
- answer the actual question
- cite evidence
- abstain if evidence is insufficient
- avoid merging conflicting sources without explanation

---

## Example response policy prompt

```text
You are the OrbitMart Knowledge Assistant.

Answer only using the retrieved evidence.
If multiple sources conflict, mention the conflict and prefer the newest approved policy.
If the evidence is insufficient, say so clearly.
Return:
1. short answer
2. evidence bullets
3. source references
```

---

## Tutorial 7 — Evaluate the RAG system

### Real-life analogy

Evals = **"mystery shopping for your bot"**. You write a fixed list of 50 realistic customer questions and the answer you'd accept (the "golden answer"). Every time you change something — swap embedder, change chunk size, tweak the prompt — you re-run the 50 questions and see if the score went up or down.

Without evals you are guessing. With evals you know whether yesterday's change made the bot better or worse, before customers find out.

### The four evaluation layers

#### 1. Retrieval quality
Did we fetch the right chunks?

#### 2. Groundedness
Did the answer stay inside the evidence?

#### 3. Answer correctness
Was the final answer right for the user?

#### 4. User/task success
Did it actually help the workflow?

---

## Example RAG eval dataset format

```json
{
  "question": "Can opened earbuds be returned within 30 days?",
  "expected_answer": "Opened earbuds may be returned within 30 days if hygiene seal is intact.",
  "must_hit_doc_ids": ["returns_policy_v12"],
  "must_mention": ["30 days", "hygiene seal"]
}
```

### Metrics to track
- hit@k
- MRR or ranking score
- answer relevance
- groundedness
- correctness
- abstention quality

### LangSmith-style evaluation dimensions
A practical RAG eval commonly checks:
- answer relevance
- answer accuracy/correctness
- retrieval quality

### Dedicated RAG evaluation frameworks

Beyond hand-rolled metrics, purpose-built frameworks accelerate eval setup:

| Framework | Strengths | When to use |
|---|---|---|
| **RAGAS** | retrieval + generation metrics out of the box (faithfulness, context recall, precision, answer relevance) | quick end-to-end quality score; LangChain/LlamaIndex-friendly |
| **TrueLens** | TruEra's RAG triad (context relevance, groundedness, answer relevance); integrates with multiple frameworks | when you want human-readable scorecards |
| **DeepEval** | unit-test style assertions; supports LLM-as-judge and custom metrics | CI/CD pipelines; regression testing |
| **LangSmith evals** | trace-native; easy to annotate runs and build datasets from production traffic | teams already using LangChain/LangGraph |

#### RAGAS quick start

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Build a dataset from your eval set
eval_data = {
    "question": ["Can opened earbuds be returned within 30 days?"],
    "answer":   ["Yes, opened earbuds may be returned within 30 days if the hygiene seal is intact."],
    "contexts": [["Return Policy section 3: Opened items may be returned within 30 days if hygiene seal is intact."]],
    "ground_truth": ["Opened earbuds may be returned within 30 days if hygiene seal is intact."],
}

dataset = Dataset.from_dict(eval_data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result)
```

#### What each metric means (RAGAS)
- **faithfulness** — is the answer supported by retrieved context? (detects hallucinations)
- **answer_relevancy** — does the answer address the question? (detects off-topic replies)
- **context_recall** — did retrieval surface evidence needed for the ground truth?
- **context_precision** — are retrieved passages actually relevant? (penalizes noisy top-k)

> Start with `faithfulness` and `context_recall` — they catch the two most common RAG failure modes.

---

## Tutorial 8 — Observability and regression testing

### Real-life analogy

Observability = **"CCTV + receipts in a restaurant"**. For every dish that goes out you can see: which customer ordered it, what ingredients went in, how long it took, and how the customer rated it. When a dish goes wrong, you can replay exactly what happened.

For RAG, every single query should be logged with: the user, the question, which chunks were retrieved, the prompt sent to the model, the answer, the latency, and (if available) the user's thumbs up/down. This is what lets you debug the **one** bad answer out of 10,000.

### Log each stage
For every request, log:
- query
- rewritten query if any
- retrieved chunk IDs
- scores
- applied filters
- final answer
- citations
- latency by stage

### Why traces matter
They let you identify if a failure came from:
- bad retrieval
- bad ranking
- bad answer synthesis
- bad document parse
- wrong metadata filter

---

## Tutorial 9 — Cost and latency optimization

### Cost levers
- smaller model for synthesis when possible
- fewer retrieved chunks
- better filtering
- caching
- cheaper reranking path
- better prompt compression

### Latency levers
- precomputed embeddings
- smaller top-k
- async retrieval
- split fast/slow paths
- cache popular queries

### Golden rule
Never optimize cost before you know what quality target you must preserve.

---

## Example production architecture

```text
[Source connectors]
    -> [Parser/Normalizer]
    -> [Chunker + Metadata Enricher]
    -> [Indexer]
    -> [Retriever Service]
    -> [Reranker / Postprocessor]
    -> [Answer Synthesis Service]
    -> [Eval + Trace + Monitoring Layer]
```

---

## Project ladder

### Project A — Ingestion pipeline
Build canonical doc objects and chunk storage.

### Project B — Retrieval benchmark
Compare chunk sizes, top-k values, and filters.

### Project C — Policy assistant
Cited answers with abstention behavior.

### Project D — Multi-tenant assistant
Add role-based filtering.

### Project E — Freshness pipeline
Handle updates, archives, and document replacement.

---

## Production checklist

- canonical schema defined
- metadata consistent
- ACL enforcement outside the model
- freshness policy defined
- retrieval logs enabled
- eval dataset exists
- regression suite exists
- answer abstention path exists
- source citation path exists
- incident rollback process exists

---

## Common mistakes

### Mistake 1: treating chunking as a one-time afterthought
It changes ranking quality dramatically.

### Mistake 2: mixing old and new policies without version control
This causes high-confidence contradictions.

### Mistake 3: no retrieval-stage logging
Then every failure looks mysterious.

### Mistake 4: trusting “looks good in a few demos”
Real systems need structured evals.

### Mistake 5: using the model prompt as the only security boundary
Never do this for document access.

---

## When you are ready for agents

You are ready to add agent workflows when:
- your retrieval pipeline is dependable
- you have clear source governance
- you can debug retrieval vs synthesis errors
- you know where human approval is needed

---

## Next file

Now continue to:
- [06_creating_agents.md](./06_creating_agents.md)

---

## References

- LangSmith RAG evaluation tutorial: <https://docs.langchain.com/langsmith/evaluate-rag-tutorial>
- LangGraph overview: <https://docs.langchain.com/oss/python/langgraph/overview>
- LangGraph persistence: <https://docs.langchain.com/oss/python/langgraph/persistence>
- LlamaIndex node postprocessors: <https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/>
- OpenAI production best practices: <https://developers.openai.com/api/docs/guides/production-best-practices>