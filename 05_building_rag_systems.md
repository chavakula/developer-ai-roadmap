# 05 — Building RAG Systems  
**Goal:** turn retrieval from a prototype into a production-style system  
**Case study:** OrbitMart enterprise knowledge assistant  
**Updated:** 2026-04-10

---

## Empowering with RAG vs building RAG

The previous file focused on **using** retrieval to ground a model.

This file focuses on **engineering the system around retrieval**:
- ingestion
- indexing
- search
- answer generation
- evaluation
- monitoring
- freshness
- access control

This is the difference between:
- “I made a doc Q&A notebook”
- and
- “I built a system people can rely on”

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

---

## Tutorial 8 — Observability and regression testing

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