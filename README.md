# LangGraph Starter Template

A minimal, modern **LangGraph 1.x** agent project. It implements a small
two-stage chatbot — collect the user's name, then chat (with tool
calling) — purely as a vehicle for the architecture patterns below.
Replace the agents with your own and keep the skeleton.

```
langgraph-template/
├── main.py                 # async CLI entry point (streaming chat loop)
├── pyproject.toml          # packaging, deps, ruff + pytest config
├── langgraph.json          # LangGraph Studio / platform config
├── LICENSE                 # 0BSD — permissive, no attribution required
├── .env.example            # copy to .env and fill in
├── .github/workflows/ci.yml# lint + format + tests on 3.10/3.12/3.14
├── examples/
│   └── human_approval.py   # standalone interrupt() demo   [removable]
├── tests/
│   ├── conftest.py         # the fake-LLM fixture
│   ├── fakes.py            # recording FakeLLM + helpers
│   ├── test_graph.py       # end-to-end graph tests
│   ├── test_persistence.py # SQLite durability             [removable]
│   └── test_examples.py    # interrupt demo tests          [removable]
└── app/
    ├── state.py            # typed state schema + reducers
    ├── llm.py              # provider-agnostic model factory
    ├── graph.py            # graph assembly, retries, Studio entry point
    ├── tools.py            # tools for the chat agent       [removable]
    ├── visualization.py    # Mermaid export helpers
    └── agents/
        ├── base.py         # BaseAgent: async LLM + structured-output plumbing
        ├── greeter.py      # onboarding stage (node + gate)
        └── chat.py         # main conversation stage (tools + trimming)
```

## Starting a new project from this template

1. **Copy the template** (don't develop inside it): on GitHub use
   **"Use this template"** (or `npx degit you/langgraph-template
   my-new-project`); locally:

   ```powershell
   robocopy D:\www\langgraph-template D:\www\my-new-project /E /XD .venv __pycache__ .git
   ```

2. **Initialise git and a virtual environment:**

   ```bash
   cd my-new-project
   git init
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (source .venv/bin/activate elsewhere)
   pip install -e ".[dev]"
   ```

   (Plain `pip install -e .` is enough at runtime; the `dev` extra adds
   pytest, ruff, the LangGraph CLI for Studio, and the SQLite saver.)

3. **Configure the environment:**

   ```bash
   copy .env.example .env          # then edit .env
   ```

   Set `OPENAI_API_KEY`. `MODEL_NAME` takes any `provider:model` string —
   see "Swapping model providers" below. Uncomment the `LANGSMITH_*`
   lines for full run traces with no code changes.

4. **Run it:**

   ```bash
   python main.py                  # interactive chat (streams tokens)
   python main.py --db chat.db     # same, sessions survive restarts
   python main.py --graph          # print the graph as Mermaid source
   pytest                          # 12 tests, no API key needed
   ruff check . && ruff format .   # lint + format
   langgraph dev                   # open the graph in LangGraph Studio
   python examples/human_approval.py   # interrupt() demo
   ```

5. **Make it yours:**

   * Rename `pyproject.toml`'s `name`, update `LICENSE`'s copyright line.
   * Replace the agents in `app/agents/`, add your tools in
     `app/tools.py`, extend `Profile`/`AppState` in `app/state.py`,
     register nodes in `app/graph.py`.
   * Remove the optional features you don't need — each one lists its
     removal steps where it lives (see "Optional features" below).

## Architecture patterns

### 1. One graph run per chat turn

Every incoming user message triggers exactly one graph run:

```
START → collect_name ──(name set?)──> chat ──(tool calls?)──> END
                 │                     ↑  └──> tools ──┘
                 └────── False ──────> END
```

If the bot needs information from the user (e.g. their name), the run
simply ends after asking. This maps naturally onto request/response
transports — no long-lived process or websocket required.

### 2. Checkpointer + thread id = sessions

State is persisted per `thread_id` by a **checkpointer**, chosen by
whoever owns the runtime:

```python
graph = build_graph(checkpointer=InMemorySaver())   # or a SQLite/Postgres saver
config = {"configurable": {"thread_id": session_id}}
state = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config)
```

The next invoke on the same thread resumes with everything collected so
far — you never manage a session store by hand. The module-level `graph`
in `app/graph.py` is compiled **without** a checkpointer: it is the
entry point declared in `langgraph.json`, and LangGraph Studio / the
platform inject their own persistence.

### 3. Async nodes

All node methods are `async def` and call the model with `ainvoke` /
`astream`. Async is the deployment-ready default — behind FastAPI or the
LangGraph platform, sync nodes serialize requests. The CLI drives the
graph with `asyncio.run`; tests wrap calls in a one-line `run()` helper.

### 4. Typed state with reducers

`app/state.py` defines the state as a `TypedDict`. `messages` carries
the `add_messages` reducer — nodes return only their *new* messages and
LangGraph appends them; `profile` has no reducer, so returning it
replaces it. Nodes return **partial updates**, never the whole state.

### 5. Agents as classes: node methods + gate methods

Each stage is a class in `app/agents/` extending `BaseAgent`:
**node methods** (async, do the work, return a state update) and
**gate methods** (sync predicates used by `add_conditional_edges`).
Prompt logic, routing logic, and wiring stay separated — `app/graph.py`
reads as a table of contents. Agent instances are shared across
sessions, so keep them **stateless**; per-conversation data belongs in
the graph state.

### 6. Tool calling (the chat ⇄ tools loop)

`app/tools.py` defines plain `@tool` functions; the chat agent binds
them and `ToolNode` executes whatever the model requests, looping back
to `chat` until it answers without tool calls (`tools_condition` does
the routing). Add a tool = write one decorated function and append it to
`TOOLS`; nothing else needs wiring.

### 7. Gated sequential onboarding

Stages that must complete before the main conversation are chained with
conditional edges. Each stage is **idempotent**: if its fact is already
collected it returns `{}` and the gate passes through, so re-running the
whole graph every turn is cheap. Adding a stage is a new node + gate
pair inserted into the chain.

> LangGraph also supports pausing *mid-run* with `interrupt()` — see
> `examples/human_approval.py` for a working demo and when to prefer it.

### 8. Structured output via Pydantic

Agents needing machine-readable answers declare a Pydantic model and
call `self.query_structured(...)`. The schema is enforced by the
provider's native structured-output mode — no JSON parsing or
validation-retry code.

### 9. Prompt-window trimming

`ChatAgent.respond` sends only the most recent `MAX_HISTORY_MESSAGES`
messages (via `trim_messages`) while the full history stays in state.
Long conversations stop growing the prompt without losing data. For
token-based budgets, pass the model itself as `token_counter`.

### 10. Provider-agnostic model factory

`app/llm.py` uses `init_chat_model`, so the model — including the
provider — is just the `MODEL_NAME` env string. See "Swapping model
providers".

### 11. Token streaming

`main.py` consumes the graph with `stream_mode="messages"`. Two details
worth copying: **filter by node** (`STREAMING_NODES` — every LLM call
emits chunks, including structured-output extractions and tool traffic
that must not reach the user), and **fall back to state** for turns that
end in a non-streaming node. The same loop works for SSE/websockets.

### 12. Retries for transient failures

LLM-calling nodes are registered with `retry_policy=RetryPolicy(...)`,
so rate limits and timeouts retry with backoff at the graph level
instead of try/except in every agent.

### 13. Testing with a fake LLM

`tests/` drives whole conversation turns through the compiled graph with
no network: `conftest.py` monkeypatches the LLM factory at the seam all
agents use (`app.agents.base.get_llm`) and substitutes a recording fake
(`fakes.py`) that supports plain, structured, and tool-binding calls.
This exercises real routing, reducers, checkpointing, tool execution,
and trimming — 12 tests in well under a second.

### 14. Visualisation, Studio, CI

`python main.py --graph` prints Mermaid source (`app/visualization.py`
also renders PNG). `langgraph dev` opens the graph in **LangGraph
Studio** for step-through debugging. GitHub Actions runs ruff + pytest
on Python 3.10/3.12/3.14 for every push and PR.

## Optional features — how to add or remove

Each feature is self-contained and marked with a bracketed tag in code
comments. Removal never requires understanding the feature's internals.

| Feature | Lives in | Remove by |
|---|---|---|
| Tool calling `[tools]` | `app/tools.py`, 2 marked lines in `chat.py`, 4 in `graph.py` | steps listed in `app/tools.py` docstring |
| History trimming `[trim]` | `app/agents/chat.py` | delete the `trim_messages` call, pass `state["messages"]` |
| SQLite sessions `[sqlite]` | `main.py` `--db` blocks, `tests/test_persistence.py` | delete the marked blocks + test + `langgraph-checkpoint-sqlite` dep |
| interrupt() demo | `examples/`, `tests/test_examples.py` | delete both files |

## Model providers

The model — including the provider — is just the `MODEL_NAME` env string
(`provider:model`, resolved by `init_chat_model`). Four providers are
supported out of the box:

| Provider | Install | `MODEL_NAME` example | Key |
|---|---|---|---|
| OpenAI | `pip install -e "."` (default) | `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `pip install -e ".[anthropic]"` | `anthropic:claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| Gemini | `pip install -e ".[google]"` | `google_genai:gemini-2.5-flash` | `GOOGLE_API_KEY` |
| Ollama | `pip install -e ".[ollama]"` | `ollama:llama3.2` | none (local) |

No code changes to switch — install the extra, set `MODEL_NAME` and the
key in `.env`. Ollama runs models locally: start the server (`ollama
serve` or the desktop app) and pull the model (`ollama pull llama3.2`)
first. Note that the greeter relies on structured output and the chat
stage on tool calling, so pick an Ollama model that supports tools.

`main.py`'s startup check validates the provider package and key up
front and prints install/config guidance instead of a traceback. To add
another provider: one row in `_PROVIDERS` (`main.py`), one extra in
`pyproject.toml`, one example line in `.env.example`.

## Serving over HTTP

The graph is transport-agnostic. A minimal FastAPI handler:

```python
graph = build_graph(checkpointer=my_durable_checkpointer)

@app.post("/chat/{session_id}")
async def chat(session_id: str, text: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config)
    return {"reply": state["messages"][-1].text}
```

Derive `thread_id` from your authenticated session (hash it if it is a
raw cookie value). For streaming responses, adapt `run_turn` in
`main.py` to yield SSE events from `astream`.

## Requirements

* Python 3.10+ (CI covers 3.10, 3.12, 3.14)
* Pinned majors: `langgraph 1.x`, `langchain 1.x`, `langchain-core 1.x`,
  `langchain-openai 1.x` (verified against langgraph 1.2.9).

## License

Released under the [0BSD](LICENSE) license (BSD Zero Clause) — a permissive
license with **no attribution requirement**. Copy this template into your
own projects, public or private, and do whatever you like with it; you do
not need to retain the copyright notice or credit the original.

> Update the copyright line in `LICENSE` to your own name or organisation
> before publishing.
