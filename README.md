# LangGraph Starter Template

A minimal, modern **LangGraph 1.x** hello-world project. It implements a
tiny two-stage chatbot — collect the user's name, then chat — purely as a
vehicle for the architecture patterns below. Replace the agents with your
own and keep the skeleton.

```
langgraph-template/
├── main.py                 # CLI entry point (streaming chat loop)
├── LICENSE                 # 0BSD — permissive, no attribution required
├── langgraph.json          # LangGraph Studio / platform config
├── requirements.txt        # runtime dependencies
├── requirements-dev.txt    # + pytest, langgraph-cli (Studio)
├── .env.example            # copy to .env and fill in
├── tests/
│   └── test_graph.py       # end-to-end graph test with a fake LLM
└── app/
    ├── state.py            # typed state schema + reducers
    ├── llm.py              # single chat-model factory
    ├── graph.py            # graph assembly, retry policy, Studio entry point
    ├── visualization.py    # Mermaid export helpers
    └── agents/
        ├── base.py         # BaseAgent: shared LLM + structured-output plumbing
        ├── greeter.py      # onboarding stage (node + gate)
        └── chat.py         # main conversation stage
```

## Starting a new project from this template

1. **Copy the template** (don't develop inside it):

   ```powershell
   # Windows
   robocopy D:\www\langgraph-template D:\www\my-new-project /E /XD .venv __pycache__
   ```

   ```bash
   # macOS / Linux
   rsync -a --exclude .venv --exclude __pycache__ langgraph-template/ my-new-project/
   ```

   (If you host the template on GitHub, mark it as a *template repository*
   and use "Use this template", or `npx degit you/langgraph-template my-new-project`.)

2. **Initialise git and a virtual environment:**

   ```bash
   cd my-new-project
   git init
   python -m venv .venv
   .venv\Scripts\activate        # Windows  (source .venv/bin/activate elsewhere)
   pip install -r requirements-dev.txt
   ```

   (`requirements.txt` alone is enough at runtime; the dev file adds
   `pytest` and the LangGraph CLI for Studio.)

3. **Configure the environment:**

   ```bash
   copy .env.example .env        # then edit .env
   ```

   Set `OPENAI_API_KEY`, and optionally `MODEL_NAME` (defaults to
   `gpt-4o-mini`). Uncomment the `LANGSMITH_*` lines to get full traces of
   every run at [smith.langchain.com](https://smith.langchain.com) — no
   code changes needed.

4. **Run it:**

   ```bash
   python main.py            # interactive chat loop (streams tokens)
   python main.py --graph    # print the graph as Mermaid source
   pytest                    # run the example test (no API key needed)
   langgraph dev             # open the graph in LangGraph Studio
   ```

5. **Make it yours:**

   * Rename/replace `app/agents/greeter.py` and `app/agents/chat.py` with
     your own agents.
   * Add your domain fields to `Profile` / `AppState` in `app/state.py`.
   * Register your nodes and edges in `app/graph.py`.
   * Keep `main.py` for local testing, or call `build_graph()` from your
     web framework of choice (see "Serving over HTTP" below).

## Architecture patterns

### 1. One graph run per chat turn

The graph is **not** a long-running loop. Every incoming user message
triggers exactly one graph run, which flows from `START` to `END` and
returns the updated state:

```
START → collect_name ──(name set?)──> chat → END
                 │
                 └────── False ────────────> END
```

If the bot needs information from the user (e.g. their name), the run
simply ends after asking the question. This maps naturally onto
request/response transports such as HTTP — no long-lived process or
websocket is required.

### 2. Checkpointer + thread id = sessions

State is persisted per `thread_id` by a **checkpointer**, passed at
compile time and selected by the runtime that owns the process:

```python
graph = build_graph(checkpointer=InMemorySaver())   # CLI / dev server
config = {"configurable": {"thread_id": session_id}}
state = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
```

The next invoke on the same thread resumes with the full message history
and everything collected so far — you never manage a session store by
hand. For production, swap in a durable checkpointer
(`langgraph-checkpoint-sqlite` or `langgraph-checkpoint-postgres`) so
sessions survive restarts.

The module-level `graph` in `app/graph.py` is compiled **without** a
checkpointer: it is the entry point declared in `langgraph.json`, and
LangGraph Studio / the platform inject their own persistence.

### 3. Typed state with reducers

`app/state.py` defines the state as a `TypedDict`. Each key can carry a
*reducer* that controls how node return values merge into state:

* `messages` uses LangGraph's `add_messages` reducer — nodes return only
  their **new** messages and LangGraph appends them.
* `profile` has no reducer, so returning it replaces it (nodes return a
  copied, updated dict).

Nodes return **partial updates** (`{"profile": {...}}`), never the whole
state. This replaces hand-rolled deep-merge + jsonschema validation with
typed, declarative merging.

### 4. Agents as classes: node methods + gate methods

Each stage of the flow is a class in `app/agents/` extending `BaseAgent`.
An agent contributes two kinds of methods:

* **Node methods** (`collect_name`, `respond`) — take the state, do the
  work, return a partial state update. Registered with `add_node`.
* **Gate methods** (`is_name_set`) — pure predicates over the state used
  as routing functions. Registered with `add_conditional_edges`.

This keeps prompt/LLM logic, routing logic, and graph wiring separated:
the graph in `app/graph.py` reads as a table of contents for the flow.
One caveat: agent instances are shared across threads/sessions, so keep
them **stateless** — anything per-conversation belongs in the graph state.

### 5. Gated sequential onboarding

Stages that must complete before the main conversation are chained with
conditional edges:

```python
builder.add_conditional_edges(
    "collect_name", greeter.is_name_set, {True: "chat", False: END}
)
```

Each stage is **idempotent**: if its fact is already collected it returns
`{}` and the gate passes straight through. So re-running the whole graph
every turn is cheap, and adding a stage is just a new node + gate pair
inserted into the chain — e.g. `collect_name → collect_company → chat`.

> Alternative: LangGraph also supports pausing *mid-run* with
> `interrupt()` / `Command(resume=...)`. The gate pattern here is simpler
> and stateless-transport-friendly; reach for `interrupt()` when a node
> must resume from its exact position rather than re-enter from `START`.

### 6. Structured output via Pydantic

When an agent needs machine-readable answers (extracting the name), it
declares a Pydantic model and calls `self.query_structured(...)`
(`app/agents/base.py`):

```python
class NameCheck(BaseModel):
    name: str | None = Field(default=None, description="...")
    reply: str = Field(default="", description="...")
```

`with_structured_output` pushes the schema to the provider's native
structured-output mode, so the reply arrives as a **validated object** —
no JSON parsing, schema-validation retries, or "the LLM added extra keys"
clean-up code.

### 7. Token streaming

`main.py` consumes the graph with `stream_mode="messages"`, printing LLM
tokens as they are generated. Two details worth copying:

* **Filter by node.** Every LLM call in the graph emits chunks, including
  the greeter's structured-output extraction, whose deltas are not
  user-facing. `STREAMING_NODES` whitelists the nodes whose tokens reach
  the user (chunk metadata carries `langgraph_node`).
* **Fall back to state.** Turns that end in a non-streaming node (the
  onboarding question) print the final message from the checkpointed
  state instead.

The same loop works server-side for Server-Sent Events or websockets.

### 8. Retries for transient failures

Nodes that call an LLM are registered with a `RetryPolicy`
(`app/graph.py`), so rate limits and timeouts are retried with
exponential backoff at the graph level instead of ad-hoc try/except
inside every agent:

```python
builder.add_node("chat", chat.respond, retry_policy=RetryPolicy(max_attempts=3))
```

### 9. Single LLM factory

All agents get their model from `app/llm.py`. Model name and credentials
come from the environment, instances are cached, and switching providers
(e.g. to `langchain-anthropic`) is a one-file change.

### 10. Graph visualisation & Studio

`app/visualization.py` exports the compiled graph as Mermaid:

```python
to_mermaid(graph)          # Mermaid source — paste into https://mermaid.live
to_png(graph, "graph.png") # PNG bytes via the mermaid.ink API
```

`python main.py --graph` prints the Mermaid source. For interactive
debugging, `langgraph dev` starts a local server (configured by
`langgraph.json`) and opens **LangGraph Studio**, where you can step
through runs node by node, inspect state at each step, and replay turns.

### 11. Testing with a fake LLM

`tests/test_graph.py` drives three full conversation turns through the
compiled graph with no network access: it monkeypatches the LLM factory
at the point agents import it (`app.agents.base.get_llm`) and substitutes
a fake supporting both plain and structured-output calls. This tests the
real routing, reducers, and checkpointer behaviour — including that
onboarding is idempotent — while keeping the suite fast and free.

## Serving over HTTP

The graph is transport-agnostic. A minimal Flask/FastAPI handler is:

```python
graph = build_graph(checkpointer=my_durable_checkpointer)

def chat_endpoint(session_id: str, text: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    state = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    return {"reply": state["messages"][-1].content}
```

Derive `thread_id` from your authenticated session (hash it if it is a
raw cookie value) and the checkpointer does the rest. For streaming
responses, adapt the `run_turn` loop in `main.py` to yield SSE events.

## Requirements

* Python 3.10+
* Pinned majors: `langgraph 1.x`, `langchain-core 1.x`,
  `langchain-openai 1.x` (verified against langgraph 1.2.9).

## License

Released under the [0BSD](LICENSE) license (BSD Zero Clause) — a permissive
license with **no attribution requirement**. Copy this template into your
own projects, public or private, and do whatever you like with it; you do
not need to retain the copyright notice or credit the original.

> Update the copyright line in `LICENSE` to your own name or organisation
> before publishing.
