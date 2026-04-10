# Zero-to-Hero AI Engineering Tutorials  
**From beginner to expert in fine-tuning, model training, model creation, RAG, and agents**  
**Updated:** 2026-04-10  
**Format:** master index + topic-wise tutorial guides  
**Core case study:** *OrbitMart*, a fictional electronics retailer with support tickets, invoices, product manuals, internal policies, and operations workflows.

---

## What this tutorial set gives you

This is not just a plan. It is a guided learning set that turns the roadmap into actual study material.

You will learn how to:

1. **Fine-tune models**
2. **Train models**
3. **Create model artifacts**
4. **Empower models with RAG**
5. **Build full RAG systems**
6. **Create agents**
7. **Build the foundations needed for all of the above**

Every guide includes:

- concept explanations in plain English
- real-world use cases
- practical architecture choices
- code patterns and markdown-friendly snippets
- exercises and mini projects
- production checklists
- “what to do next” paths

---

## How to use this tutorial set

### If you are starting from zero
Read the files in this order:

1. [00_foundations.md](./00_foundations.md)
2. [01_fine_tuning_models.md](./01_fine_tuning_models.md)
3. [02_training_models.md](./02_training_models.md)
4. [03_creation_of_models.md](./03_creation_of_models.md)
5. [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md)
6. [05_building_rag_systems.md](./05_building_rag_systems.md)
7. [06_creating_agents.md](./06_creating_agents.md)

### If you already know some ML
Jump directly to:

- fine-tuning → [01_fine_tuning_models.md](./01_fine_tuning_models.md)
- distributed training → [02_training_models.md](./02_training_models.md)
- tokenizers/custom architectures → [03_creation_of_models.md](./03_creation_of_models.md)
- retrieval and production RAG → [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) and [05_building_rag_systems.md](./05_building_rag_systems.md)
- workflows, tools, handoffs, approvals, MCP → [06_creating_agents.md](./06_creating_agents.md)

---

## Current stack snapshot

This tutorial set is aligned to the **current official/public stack checked on 2026-04-10**.

### OpenAI platform
- OpenAI’s current model pages recommend starting with **GPT-5.4** for complex reasoning/coding workflows, and **GPT-5.4 mini / nano** for faster, cheaper workloads.
- OpenAI’s optimization guides currently emphasize **evals first**, then **SFT**, **DPO**, **RFT**, and **graders**.
- OpenAI’s file search tool in the Responses API provides a managed retrieval path using **semantic + keyword search** over vector stores.
- OpenAI’s agent stack centers on the **Agents SDK**, **tools**, **handoffs**, **guardrails**, **sessions**, and **traces**.

### Open-source stack
- **PyTorch 2.11 docs** are the reference for `torch.compile`, DDP, profiling, and training primitives.
- **Transformers v5.x docs** are the reference for custom architectures and custom tokenizers.
- **PEFT** remains the practical default for adapter tuning, especially **LoRA**.
- **TRL** is a primary open-source training library for SFT, DPO, and other alignment workflows.
- **Accelerate** remains the easiest practical path into **FSDP**, **DeepSpeed**, and low-precision training.
- **LangSmith / LangGraph** and **LlamaIndex** remain useful for evaluating and structuring RAG/agent systems.
- **MCP** is a major interoperability layer for connecting tools and context to model applications.

> Important: model eligibility and exact API support change over time. Before production use, re-check the current model page or dashboard for any feature such as fine-tuning eligibility, tool support, or price.

---

## Learning path map

| Phase | File | Main output |
|---|---|---|
| Foundations | [00_foundations.md](./00_foundations.md) | You can read training code and build small models |
| Fine-tuning | [01_fine_tuning_models.md](./01_fine_tuning_models.md) | You can decide between prompting, RAG, SFT, DPO, RFT |
| Training | [02_training_models.md](./02_training_models.md) | You can build and scale training loops |
| Model creation | [03_creation_of_models.md](./03_creation_of_models.md) | You can create tokenizers, configs, and model packages |
| RAG enablement | [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md) | You can ground a model on external knowledge |
| RAG systems | [05_building_rag_systems.md](./05_building_rag_systems.md) | You can build production-style retrieval systems |
| Agents | [06_creating_agents.md](./06_creating_agents.md) | You can build tool-using, approval-aware, traceable agents |

---

## Suggested study rhythm

### Standard pace
- **36 weeks**
- **8 to 10 hours/week**

### Intensive pace
- **24 weeks**
- **15 to 20 hours/week**

### Weekly structure
- **2 sessions**: theory + docs
- **2 sessions**: coding + experiments
- **1 session**: evals, debugging, notes

---

## The OrbitMart case study

To make the tutorials feel real, most examples reuse a single fictional company:

### OrbitMart
A mid-sized electronics retailer that has:
- customer support tickets
- invoices and purchase orders
- product manuals
- return and refund policies
- internal knowledge base
- ops workflows such as refunds, order lookup, ticket routing, and compliance review

Why this matters:
- it gives you a realistic setting for **fine-tuning**
- it gives you text corpora for **training and tokenizer creation**
- it gives you documents and metadata for **RAG**
- it gives you operational workflows for **agents**

---

## Capstone sequence

By the end of this set, your capstone should be:

1. train a domain tokenizer  
2. pretrain or continue-train a small text model  
3. fine-tune it for one downstream business task  
4. add RAG over internal documents  
5. wrap it in an agent with tools and approvals  
6. add evals, traces, and deployment notes  

Suggested capstone themes:
- support operations copilot
- finance document assistant
- internal policy assistant
- engineering knowledge agent

---

## Section summaries

## 00 Foundations
Read this if you are brand new to the stack.  
You will learn Python-for-ML basics, PyTorch basics, tokenization, evaluation, and how to train two small models.

See: [00_foundations.md](./00_foundations.md)

## 01 Fine-Tuning Models
This guide teaches when to use prompting, RAG, SFT, DPO, RFT, or PEFT.  
Includes hosted and open-model workflows.

See: [01_fine_tuning_models.md](./01_fine_tuning_models.md)

## 02 Training Models
This guide teaches training loops, batching, losses, perplexity, mixed precision, DDP, FSDP, DeepSpeed, and profiling.

See: [02_training_models.md](./02_training_models.md)

## 03 Creation of Models
This guide teaches tokenizer training, configuration design, custom architectures, Hugging Face packaging, and large-model toolchains.

See: [03_creation_of_models.md](./03_creation_of_models.md)

## 04 Empowering Models with RAG
This guide teaches how to ground models on private and fresh knowledge using managed and self-managed retrieval.

See: [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md)

## 05 Building RAG Systems
This guide teaches how to turn retrieval into a real production system with ingestion, metadata, evals, freshness, ACLs, and observability.

See: [05_building_rag_systems.md](./05_building_rag_systems.md)

## 06 Creating Agents
This guide teaches tool calling, structured outputs, sessions, handoffs, guardrails, approvals, tracing, MCP integration, and agent evals.

See: [06_creating_agents.md](./06_creating_agents.md)

---

## Production habits that apply everywhere

- always build a baseline before optimizing
- always keep a held-out validation/test set
- always add evals before rollout
- always log experiments
- always track data lineage
- always know which lever you are pulling:
  - prompt
  - retrieval
  - fine-tuning
  - architecture change
  - workflow change
  - agent/tool change

---

## File inventory

- [00_foundations.md](./00_foundations.md)
- [01_fine_tuning_models.md](./01_fine_tuning_models.md)
- [02_training_models.md](./02_training_models.md)
- [03_creation_of_models.md](./03_creation_of_models.md)
- [04_empowering_models_with_rag.md](./04_empowering_models_with_rag.md)
- [05_building_rag_systems.md](./05_building_rag_systems.md)
- [06_creating_agents.md](./06_creating_agents.md)

---

## Official references used to keep this set current

### OpenAI
- Models: <https://developers.openai.com/api/docs/models>
- GPT-5.4 model page: <https://developers.openai.com/api/docs/models/gpt-5.4>
- Evaluation best practices: <https://developers.openai.com/api/docs/guides/evaluation-best-practices>
- Supervised fine-tuning: <https://developers.openai.com/api/docs/guides/supervised-fine-tuning>
- DPO: <https://developers.openai.com/api/docs/guides/direct-preference-optimization>
- RFT: <https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning>
- Graders: <https://developers.openai.com/api/docs/guides/graders>
- File search: <https://developers.openai.com/api/docs/guides/tools-file-search>
- Agents SDK guide: <https://developers.openai.com/api/docs/guides/agents-sdk>
- Agent evals: <https://developers.openai.com/api/docs/guides/agent-evals>
- Practical guide to building agents: <https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/>

### PyTorch
- Compiler FAQ: <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html>
- torch.distributed: <https://docs.pytorch.org/docs/stable/distributed.html>
- torch.optim: <https://docs.pytorch.org/docs/stable/optim.html>

### Hugging Face
- Create a custom architecture: <https://huggingface.co/docs/transformers/main/create_a_model>
- Custom tokenizers: <https://huggingface.co/docs/transformers/main/custom_tokenizers>
- Causal language modeling: <https://huggingface.co/docs/transformers/tasks/language_modeling>
- Masked language modeling: <https://huggingface.co/docs/transformers/tasks/masked_language_modeling>
- LoRA / PEFT: <https://huggingface.co/docs/peft/package_reference/lora>
- TRL: <https://huggingface.co/docs/trl/index>
- Accelerate FSDP: <https://huggingface.co/docs/accelerate/usage_guides/fsdp>
- FSDP vs DeepSpeed: <https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed>
- Low precision training: <https://huggingface.co/docs/accelerate/usage_guides/low_precision_training>

### RAG / orchestration / MCP
- LangSmith RAG evaluation tutorial: <https://docs.langchain.com/langsmith/evaluate-rag-tutorial>
- LangGraph overview: <https://docs.langchain.com/oss/python/langgraph/overview>
- LangGraph persistence: <https://docs.langchain.com/oss/python/langgraph/persistence>
- LlamaIndex node postprocessors: <https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/>
- MCP specification (latest): <https://modelcontextprotocol.io/specification/2025-11-25>
- MCP roadmap: <https://modelcontextprotocol.io/development/roadmap>

---

## Next practical step

Start with [00_foundations.md](./00_foundations.md), finish the two mini projects there, and only then move to fine-tuning.