# 06 — Creating Agents  
**Goal:** build tool-using, stateful, approval-aware, traceable agents  
**Case study:** OrbitMart support operations copilot  
**Updated:** 2026-04-28

> **First time here?** Read [00_foundations.md](./00_foundations.md) for the basics, then come back. Agents build on every prior chapter.

---

## The whole chapter in one paragraph

A **chatbot answers**. An **agent does**. An agent can call tools (look up an order, send an email, refund a payment), take multi-step actions, remember what it did, ask a human before risky moves, and leave a trail of what it did and why. This chapter teaches you the agent loop (model decides → call tool → read result → decide again), how to define safe tools, when to require human approval, and how to observe and debug what the agent is actually doing.

> **Real-life analogy.** A chatbot is like a **receptionist** who tells you "I'll have someone call you back." An agent is like a **junior assistant** who actually picks up the phone, looks up your order in the system, sends the refund, and emails you the confirmation — all while keeping a log of every step so their manager can review it later. The chatbot describes the world; the agent changes the world.

### Code teaser: an OrbitMart support agent in 40 lines

This is the entire agent recipe: define safe tools, let the model pick which one to call, run it, feed the result back, repeat. Real agents add memory, approvals, and tracing on top — but the core loop is exactly this.

```python
import json
from openai import OpenAI

client = OpenAI()

# 1. SAFE TOOLS — tiny Python functions the agent is allowed to call.
ORDERS = {
    "88421": {"status": "shipped", "eta": "2026-04-30", "customer": "Alex"},
    "99102": {"status": "processing", "eta": "2026-05-02", "customer": "Maya"},
}

def get_order_status(order_id: str) -> dict:
    return ORDERS.get(order_id, {"error": f"order {order_id} not found"})

def issue_refund(order_id: str, amount: float) -> dict:
    # In real life: call payments API. Refunds usually require human approval first!
    return {"order_id": order_id, "refunded": amount, "status": "refunded"}

TOOLS = {
    "get_order_status": get_order_status,
    "issue_refund":     issue_refund,
}

# 2. TOOL SCHEMAS — tell the model what each tool does and what arguments it takes.
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "Look up the status and ETA of an OrbitMart order.",
        "parameters": {"type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "issue_refund",
        "description": "Refund an OrbitMart order. Requires manager approval over $100.",
        "parameters": {"type": "object",
            "properties": {"order_id": {"type": "string"},
                           "amount":   {"type": "number"}},
            "required": ["order_id", "amount"]}}},
]

# 3. THE AGENT LOOP
def agent(user_msg: str, max_steps: int = 4) -> str:
    messages = [
        {"role": "system", "content": "You are an OrbitMart support agent. Use tools when needed."},
        {"role": "user", "content": user_msg},
    ]
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOL_SPECS,
        ).choices[0].message
        messages.append(resp)

        if not resp.tool_calls:                # model is done
            return resp.content

        for call in resp.tool_calls:           # run each requested tool
            args = json.loads(call.function.arguments)
            result = TOOLS[call.function.name](**args)
            print(f"  ↳ {call.function.name}({args}) → {result}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})

    return "(agent gave up after max_steps)"

print(agent("Where is order 88421 and what's the ETA?"))
#   ↳ get_order_status({'order_id': '88421'}) → {'status': 'shipped', 'eta': '2026-04-30', ...}
# Your order 88421 has shipped and is expected to arrive on April 30, 2026.
```

That's an agent. Add memory, traces, human-approval gates for risky tools (`issue_refund` over $100!), and evals — and you have a production system. Everything else in this chapter is making this loop safe, observable, and trustworthy.

---

## Plain English: what this chapter is about

A chatbot **answers**. An agent **does**.

```text
PLAIN CHATBOT                          AGENT
┌───────────────────────────┐          ┌─────────────────────────────────────┐
│ user: "where is order 42?"│          │ user: "where is order 42?"          │
│                           │          │     ↓                               │
│ model: "I cannot look     │          │ model decides: I should call        │
│         up orders. Please │          │   get_order_status(order_id=42)     │
│         contact support." │          │     ↓                               │
└───────────────────────────┘          │ tool runs → returns                 │
   one turn,                           │   {"status":"shipped",              │
   no actions,                         │    "eta":"2026-04-12"}              │
   no real-world effect.               │     ↓                               │
                                       │ model: "Your order shipped on       │
                                       │  Apr 10 and arrives Apr 12."        │
                                       └─────────────────────────────────────┘
                                          multi-step, calls tools,
                                          can take real-world actions.
```

The recipe for an agent:

```text
   ┌────────────────────────────────────────────────────────┐
   │                  AGENT LOOP                            │
   │                                                        │
   │   user msg ──▶ model + instructions + tool list        │
   │                          │                             │
   │                          ▼                             │
   │              decide: answer OR call a tool             │
   │                          │                             │
   │             ┌────────────┴────────────┐                │
   │             ▼                         ▼                │
   │       call tool                 final answer ──▶ user  │
   │             │                                          │
   │             ▼                                          │
   │       tool result ──▶ feed back into model             │
   │             │                                          │
   │             └─────────── loop ◀────────────────┐       │
   │                                                │       │
   │   (optional: human approval before risky tool) ┘       │
   └────────────────────────────────────────────────────────┘
```

Start with **one** agent, **a few** tools, and **clear approval points**. That solves more real problems than people expect.

---

## Mini-glossary: jargon in this chapter

| Term | One-line meaning |
|---|---|
| Agent | A model + instructions + tools + a loop that can take multi-step actions. |
| Tool / function | A typed function the model can call (e.g. `get_order_status`). |
| Function-calling | The model returning a structured request to invoke a tool. |
| Tool schema | The JSON description of a tool's name, args, and types. |
| Session / state | Memory of the current conversation/run (messages, tool results). |
| Loop | The cycle: model → tool → result → model → ... until done. |
| Handoff | Passing the conversation from one agent to a specialist agent. |
| Multi-agent | A team of agents (router + specialists), each with focused tools. |
| Guardrail | A check that blocks unsafe inputs/outputs (PII, policy, scope). |
| Approval gate | A human "yes/no" required before a risky tool runs. |
| Trace | A logged record of one agent run (every step + tool call). |
| Eval | Scored test cases for the agent (did it pick the right tool? right answer?). |
| MCP | Model Context Protocol — a standard way to expose tools/data to any agent. |
| System prompt / instructions | The fixed text describing the agent's role and rules. |
| Tool-use loop limit | Max steps before forcing the agent to stop (prevents runaway loops). |

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

### The story we'll follow in this chapter

OrbitMart's support team is drowning. For every customer email an agent has to: look up the order, check the policy, draft a reply, sometimes issue a refund, sometimes hand off to finance. You will build the digital coworker that does this safely — not a fully autonomous robot, a careful coworker that **asks for approval** before doing anything risky.

| Step | Tutorial | Real-life equivalent |
|---|---|---|
| One agent + a few tools | Tutorial 1 (single agent) | A new hire with a phone book and a few buttons |
| Make the output predictable | Tutorial 2 (structured outputs) | The hire writes on a printed form, not free-text |
| Block risky actions until a human says yes | Tutorial 3 (approvals) | "Refunds over $100 need supervisor signature" |
| Remember the conversation | Tutorial 4 (sessions) | Same hire, same chat, doesn't ask your name 3 times |
| Log every step | Tutorial 5 (tracing) | A logbook so you can replay any case |
| Block bad inputs / outputs | Tutorial 6 (guardrails) | "Don't paste customer credit cards into the chat" |
| Bring in specialists when needed | Tutorial 7 (multi-agent + handoffs) | Transfer rare finance cases to the finance team |
| Choose: call a specialist as a tool, or hand over | Tutorial 8 (tools vs handoffs) | "Ask a colleague" vs "transfer the call" |
| Plug in standard tools from anywhere | Tutorial 9 (MCP) | Universal USB-C for tools |
| Score the whole workflow | Tutorial 10 (evals) | Mystery-shopper test plus QA review |

By the end you have a coworker, not a chatbot.

---

## Tutorial 1 — Build a single tool-using agent

### Real-life analogy

A single agent with tools = **"a new hire with a phone book and a few buttons"**. They can look things up (read tools) and press buttons (action tools). They are not creative, not autonomous — they follow instructions and use the right tool for the right question.

This sounds simple, but solves a surprising amount of real work. Don't jump to multi-agent until this isn't enough.

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

### Real-life analogy

Approval gates = **"refunds over $100 need supervisor signature"**. The agent can do small, safe things on its own (read order status, draft a reply). For anything **irreversible** (issuing a refund, sending an email to a customer, changing a payment account), it must **stop and ask** a human first.

This is the single most important production pattern. It turns a scary autonomous bot into a safe assistant.

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

### Real-life analogy

Sessions = **"the same hire, the same chat"**. Without sessions the agent has amnesia: every message is treated as a stranger. With sessions, when the customer replies "yes please", the agent still remembers what "yes" is referring to.

A session is just a conversation ID + the list of messages and tool results so far.

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

### Real-life analogy

Tracing = **"flight recorder for every agent run"**. After the fact, when a customer complains "the bot promised me a refund and never delivered", you can pull the trace and see, step-by-step: what the user said, which tools the agent called, what each tool returned, what the agent decided next, and what it told the customer.

Without traces, debugging an agent is guesswork. With traces, every failure is reproducible.

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

### Real-life analogy

Guardrails = **"the bumpers on a bowling lane"**. They don't roll the ball for you, they just stop it from going into the gutter. Examples: block prompts containing customer credit cards, refuse off-topic questions, refuse to discuss competitors, refuse to give legal advice.

Guardrails run **around** the agent — before the input goes in (input guardrails) and before the output goes out (output guardrails).

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

### Visual: single-agent vs multi-agent with handoffs

```text
SINGLE AGENT (start here)              MULTI-AGENT WITH HANDOFFS
┌────────────────────────────┐         ┌──────────────────────────────────────┐
│         user               │         │              user                    │
│           │                │         │                │                     │
│           ▼                │         │                ▼                     │
│   ┌────────────────┐       │         │       ┌────────────────┐             │
│   │  one agent     │       │         │       │ triage agent   │             │
│   │  many tools:   │       │         │       │ (router only)  │             │
│   │  - get_order   │       │         │       └───┬────────┬───┘             │
│   │  - get_policy  │       │         │           │        │                 │
│   │  - issue_refund│       │         │     handoff      handoff             │
│   └────────────────┘       │         │           ▼        ▼                 │
│                            │         │   ┌────────────┐ ┌────────────┐      │
│  good for: 80% of cases    │         │   │ refund     │ │ finance    │      │
│  one prompt, one loop      │         │   │ agent      │ │ agent      │      │
└────────────────────────────┘         │   │ (issue_    │ │ (update_   │      │
                                       │   │  refund)   │ │  payment)  │      │
                                       │   └────────────┘ └────────────┘      │
                                       │                                      │
                                       │  good when: distinct domains,        │
                                       │  different tool sets, different      │
                                       │  approval rules per specialist       │
                                       └──────────────────────────────────────┘
```

Rule of thumb: do **not** start multi-agent. Move to it only when one agent's prompt + tools become unwieldy or when specialist agents need different guardrails / approval rules.

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

### Real-life analogy

MCP = **"USB-C for tools"**. Before MCP, every agent framework had its own way to expose tools (different code, different schemas). With MCP, any tool that speaks the protocol can be plugged into any compatible agent. Need a calendar tool? A database tool? A payments tool? If they speak MCP, they just snap in.

This is what turns ad-hoc agent demos into a real ecosystem.

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

### Real-life analogy

Agent evals = **"mystery shopper for your coworker"**. You write a fixed set of realistic scenarios ("customer asks about a missing order", "customer asks for a refund on an opened item") and the expected behavior ("agent should call get_order_status, then ask for the order ID, then…"). You replay these scenarios after every change.

For agents you score not just the **final answer** but the **path**: did it pick the right tool? did it stop at the approval gate? did it hand off when it should have?

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