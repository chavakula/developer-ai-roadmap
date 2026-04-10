# 06 — Creating Agents  
**Goal:** build tool-using, stateful, approval-aware, traceable agents  
**Case study:** OrbitMart support operations copilot  
**Updated:** 2026-04-10

---

## What an agent is

An agent is more than a single completion.

A practical agent combines:
- a model
- instructions
- tools
- state/session
- execution loop
- validations / guardrails
- observability
- optional handoffs to other agents
- optional human approval for risky actions

---

## The correct mindset

Do **not** start with:
> “I want a fully autonomous multi-agent system.”

Start with:
> “What workflow am I automating, what tools are needed, and where should humans stay in control?”

This is the difference between a demo and a production system.

---

## Current agent stack snapshot

The current practical stack emphasizes:
- **single-agent systems first**
- reusable tool definitions
- human intervention for risky/irreversible steps
- traces for debugging
- sessions/state management
- handoffs only when needed
- MCP for standardized tool/context integration

---

## The simplest useful agent loop

```text
user input
 -> model decides
 -> maybe calls a tool
 -> tool result returns
 -> model continues
 -> final answer or action request
```

That alone is enough for many real workflows.

---

## OrbitMart agent use case

OrbitMart wants an internal operations copilot that can:
- classify support requests
- look up policies
- fetch order status
- draft a reply
- request approval before issuing refunds
- hand off rare finance questions to a specialist agent

This is a great agent case because it has:
- data retrieval tools
- action tools
- clear approval boundaries
- multi-step reasoning
- measurable success/failure

---

## Tutorial 1 — Build a single tool-using agent

### Why start here
A single agent with good tools solves more than many people expect.

### Minimal agent
```python
from agents import Agent, Runner, function_tool

@function_tool
def order_status(order_id: str) -> str:
    """Look up the latest order status."""
    # Replace with real database/API call
    return f"Order {order_id}: shipped on 2026-04-02, carrier = BlueShip"

agent = Agent(
    name="OrbitMart Support Agent",
    instructions=(
        "Help internal support agents. "
        "Use tools when needed. "
        "Do not invent order details. "
        "If you lack evidence, say so."
    ),
    tools=[order_status],
)

result = Runner.run_sync(agent, "Check order 88421 and summarize status.")
print(result.final_output)
```

### What this teaches you
- tool schema comes from the function signature/docstring
- the model decides whether to call the tool
- tool output becomes part of the reasoning loop

---

## Tool design rules

A good tool has:
- a clear name
- a clear description
- well-defined arguments
- one responsibility
- predictable output

### Bad tool
`do_everything(order_id, mode, query, extra, flag)`

### Better tools
- `get_order_status(order_id)`
- `lookup_return_policy(product_family)`
- `create_refund_case(order_id, reason_code)`

### Why this matters
If the model cannot clearly distinguish tools, tool choice degrades quickly.

---

## Tutorial 2 — Structured outputs and operational boundaries

### Business problem
OrbitMart wants the agent to return a structured action plan, not free text only.

### Example schema
```python
from pydantic import BaseModel, Field

class SupportPlan(BaseModel):
    intent: str = Field(description="Ticket intent")
    needs_order_lookup: bool
    needs_human_approval: bool
    reply_draft: str
```

### Why structured outputs matter
They make downstream automation safer and easier.

Use them when:
- another service consumes the output
- a UI must render predictable fields
- you need explicit approval routing
- you want clean evals

---

## Tutorial 3 — Human approval for risky actions

### Business problem
The agent can suggest a refund, but must not issue one without approval.

### Marking tools that need approval
```python
from agents import Agent, Runner, function_tool

@function_tool(needs_approval=True)
async def cancel_order(order_id: int) -> str:
    return f"Cancelled order {order_id}"
```

### Why this is critical
Refunds, cancellations, escalations, and account changes are exactly where automation can become expensive.

### Good approval candidates
- refunds
- order cancellation
- address changes after shipment
- supplier payment changes
- policy override exceptions

---

## Tutorial 4 — Sessions and conversation state

### Why session management matters
Many workflows are multi-turn:
- first ask order number
- then confirm customer identity
- then inspect status
- then draft response

The agent needs memory.

### Local session example
```python
import asyncio
from agents import Agent, Runner, SQLiteSession

async def main():
    agent = Agent(
        name="OrbitMart Support Agent",
        instructions="Reply concisely and remember prior turns in the session."
    )

    session = SQLiteSession("support_case_88421")

    result = await Runner.run(agent, "Customer says package never arrived.", session=session)
    print(result.final_output)

    result = await Runner.run(agent, "The order number is 88421.", session=session)
    print(result.final_output)

asyncio.run(main())
```

### Why sessions matter in production
They help with:
- multi-turn context
- resumability
- agent continuity
- better user experience

---

## Tutorial 5 — Tracing and observability

### Why traces are essential
Without traces, debugging agents becomes guesswork.

The current agent stack emphasizes traces for:
- model calls
- tool calls
- handoffs
- guardrails
- custom events

### What a trace lets you inspect
- which tool was chosen
- what arguments were sent
- whether a handoff occurred
- where the workflow failed
- how long each stage took

### Practical rule
Do not ship serious agent workflows without traces.

---

## Tutorial 6 — Guardrails

Guardrails are validations that protect workflow boundaries.

### Input guardrails
Check user input before or during execution.

Examples:
- off-topic requests
- abuse
- sensitive-data leakage
- out-of-scope usage

### Output guardrails
Check the final answer.

Examples:
- policy violation
- missing citation
- unsupported claim
- disallowed action recommendation

### Tool guardrails
Wrap custom function-tool calls.

This matters because agent-level input/output guardrails do not automatically cover every intermediate tool invocation in multi-step flows.

---

## Example input guardrail shape

```python
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    input_guardrail,
)

@input_guardrail(name="support_scope_guard", run_in_parallel=False)
def support_scope_guard(context: RunContextWrapper, agent: Agent, input_text):
    text = str(input_text).lower()
    blocked = "do my homework" in text or "write malware" in text
    return GuardrailFunctionOutput(
        output_info={"blocked": blocked},
        tripwire_triggered=blocked,
    )
```

### When to run in blocking mode
Use blocking guardrails when you want to prevent:
- tool calls
- token spend
- side effects

before the workflow starts.

---

## Tutorial 7 — Multi-agent systems and handoffs

### When to consider multiple agents
Only after a single agent becomes hard to manage.

Good reasons:
- prompt too complex
- tool set too overloaded
- different specialists need different policies
- different approval rules apply
- failure modes become clearer when split

### Not-good reasons
- “multi-agent sounds advanced”
- “I saw a cool diagram online”
- “we have 5 tools so we need 5 agents”

---

## Handoff pattern

Handoffs let one agent delegate to another specialized agent.

### Example roles
- triage agent
- refund agent
- logistics agent
- policy agent
- finance agent

### Concept
A handoff behaves like a tool that transfers control.

### Example scenario
User asks:
> “Customer says the refund never arrived and also wants to change the supplier payment account.”

This likely spans:
- refund operations
- finance controls

A triage agent can route/handoff rather than trying to own both deeply.

---

## Tutorial 8 — Agents as tools vs handoffs

### Agents as tools
A manager agent calls specialist agents as tools, then stays in control.

Use when:
- you want one agent to own the user relationship
- you want central orchestration
- specialists should not take over the conversation

### Handoffs
Execution is delegated to another agent.

Use when:
- the next specialist should own the next steps
- the workflow naturally changes domain

### Practical advice
Start with **agents as tools** or a single agent.  
Use handoffs only when the workflow benefits from transferred control.

---

## Tutorial 9 — MCP integration

### What MCP is
MCP standardizes how applications expose:
- tools
- prompts
- resources/context

Think of it as a standardized connector layer between model apps and external capabilities.

### Why it matters
Without MCP-style standards, every integration becomes bespoke.

### Good MCP use cases
- internal knowledge connectors
- enterprise tool catalogs
- shared integrations across multiple agent apps
- external service bridges

### Security mindset for MCP
MCP can expose powerful capabilities, so you must design:
- user consent
- tool authorization
- audit trails
- least privilege
- clear trust boundaries

---

## Tutorial 10 — Evaluate agent workflows

### Start with traces
Trace grading is often the fastest way to debug an agent workflow.

Questions to evaluate:
- Did the agent pick the right tool?
- Did it ask for approval when required?
- Did it stay in scope?
- Did it hand off correctly?
- Did it violate policy?

### Then move to datasets and repeatable evals
Build evaluation sets that represent real workflows, not only toy prompts.

### Example eval dataset row
```json
{
  "user_input": "Customer wants a refund for order 88421 and asks to expedite it.",
  "expected_tool_sequence": ["get_order_status", "lookup_return_policy"],
  "must_request_approval": true,
  "allowed_final_actions": ["draft_reply", "open_refund_case"]
}
```

---

## Attack surface and safety review

### Common risks
- prompt injection through retrieved docs
- bad tool arguments
- unsafe autonomous actions
- approval bypass
- cross-tenant data leakage
- stale or conflicting policy sources

### Defenses
- strong tool descriptions and schemas
- tool/output validation
- retrieval filtering
- session/user scoping
- approval gates
- traces and audits
- explicit business-rule checks outside the model

---

## Example production pattern for OrbitMart

```text
User / Support Agent UI
    -> Triage Agent
        -> order tools
        -> policy retrieval
        -> refund specialist
        -> finance specialist
    -> approval service for risky actions
    -> trace + eval layer
```

### Business-safe behavior
- look up data automatically
- draft actions automatically
- require approval for refund issuance
- keep audit records
- log handoffs and tool use

---

## Project ladder

### Project A — single-agent support copilot
Tools: order lookup + policy search

### Project B — approval-aware refund workflow
Add `needs_approval` tools and structured outputs

### Project C — multi-agent support desk
Triage + refund + finance specialists

### Project D — MCP-backed integration layer
Expose internal services through a standardized tool/context layer

### Project E — agent evaluation suite
Trace reviews + dataset-based regression tests

---

## Production checklist

- single-agent baseline tried first
- tool definitions standardized
- risky actions require approval
- retrieval sources governed
- traces enabled
- session strategy defined
- eval dataset exists
- rollback/disable path exists
- audit logs retained
- security review completed

---

## Common mistakes

### Mistake 1: going multi-agent too early
This creates complexity faster than value.

### Mistake 2: treating tools like vague helper functions
Agents need crisp tool contracts.

### Mistake 3: no approval path for side effects
This is how expensive mistakes happen.

### Mistake 4: no traces
Then you cannot explain failures.

### Mistake 5: trusting the model with policy enforcement alone
Critical controls belong outside the model too.

---

## Final expert-level goal

You are at a strong expert level when you can:
- choose between single-agent and multi-agent designs
- define high-quality tools
- add approval boundaries intelligently
- integrate retrieval and tools without confusion
- evaluate behavior from traces and datasets
- explain safety, cost, and maintainability tradeoffs clearly

---

## References

- OpenAI Agents SDK guide: <https://developers.openai.com/api/docs/guides/agents-sdk>
- OpenAI agent evals: <https://developers.openai.com/api/docs/guides/agent-evals>
- OpenAI practical guide to building agents: <https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/>
- OpenAI Agents SDK docs: <https://openai.github.io/openai-agents-python/>
- Running agents: <https://openai.github.io/openai-agents-python/running_agents/>
- Tools: <https://openai.github.io/openai-agents-python/tools/>
- Handoffs: <https://openai.github.io/openai-agents-python/handoffs/>
- Guardrails: <https://openai.github.io/openai-agents-python/guardrails/>
- Human in the loop: <https://openai.github.io/openai-agents-python/human_in_the_loop/>
- Tracing: <https://openai.github.io/openai-agents-python/tracing/>
- MCP specification: <https://modelcontextprotocol.io/specification/2025-11-25>
- MCP roadmap: <https://modelcontextprotocol.io/development/roadmap>