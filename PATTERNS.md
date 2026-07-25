# Architecture patterns

The 18 patterns this template demonstrates, each with the
file that implements it. You do **not** need to read these to use the
template — start with [README.md](README.md) to run it, or
[ELI5.md](ELI5.md) if LangGraph is new to you. Come back here when you
want to know *why* something is built the way it is, or before changing
it.

1. [One graph run per chat turn](#1-one-graph-run-per-chat-turn)
2. [Checkpointer + thread id = sessions](#2-checkpointer--thread-id--sessions)
3. [Async nodes](#3-async-nodes)
4. [Typed state with reducers](#4-typed-state-with-reducers)
5. [Agents as classes: node methods + gate methods](#5-agents-as-classes-node-methods--gate-methods)
6. [Tool calling (the chat ⇄ tools loop)](#6-tool-calling-the-chat--tools-loop)
7. [Gated sequential onboarding](#7-gated-sequential-onboarding)
8. [Structured output via Pydantic](#8-structured-output-via-pydantic)
9. [Prompt-window trimming](#9-prompt-window-trimming)
10. [Provider-agnostic model factory, per-agent overrides](#10-provider-agnostic-model-factory-per-agent-overrides)
11. [Token streaming](#11-token-streaming)
12. [Retries for transient failures](#12-retries-for-transient-failures)
13. [Testing with a fake LLM](#13-testing-with-a-fake-llm)
14. [Visualisation, Studio, CI](#14-visualisation-studio-ci)
15. [Image (vision) input](#15-image-vision-input)
16. [Side-effecting agents (batch pipelines)](#16-side-effecting-agents-batch-pipelines)
17. [Logging](#17-logging)
18. [Evals — grading the model, not the wiring](#18-evals--grading-the-model-not-the-wiring)

---

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
graph = build_graph(checkpointer=InMemorySaver())  # or a SQLite/Postgres saver
config = {"configurable": {"thread_id": session_id}}
state = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config)
```

The next invoke on the same thread resumes with everything collected so
far — you never manage a session store by hand. Two runnable extensions
of this pattern live in `examples/`: `long_term_memory.py` (facts that
survive *across* threads, via the Store API) and `time_travel.py`
(listing, replaying, and forking a thread's checkpoints). The module-level `graph`
in `app/graph.py` is compiled **without** a checkpointer: it is the
entry point declared in `langgraph.json`, and LangGraph Studio / the
platform inject their own persistence.

Because that module-level `graph = build_graph()` runs at **import
time**, everything `build_graph` constructs must stay side-effect free
until first use — no filesystem writes, no network calls in
constructors. If an agent needs an output file or a client, initialise
it lazily (on first use), or importing `app.graph` will misbehave in
tests and tooling.

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

`tools/__init__.py` defines plain `@tool` functions; the chat agent binds
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
> `examples/human_approval.py` for the bare mechanism and when to prefer
> it, and `examples/tool_approval.py` for its production use: gating
> dangerous tool calls behind human approval.

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

### 10. Provider-agnostic model factory, per-agent overrides

`lib/llm.py` uses `init_chat_model`, so the model — including the
provider — is just the `MODEL_NAME` env string. See "Model providers".

Different graph stages can run **different models** as pure
configuration: pass `model_env="MY_STAGE_MODEL"` to an agent's
constructor and that env variable (any `provider:model` string)
overrides `MODEL_NAME` for that agent only — e.g. a cheap fast model for
extraction and a stronger one for generation. Name the same variable in
`check_environment(extra_model_vars=("MY_STAGE_MODEL",))` so it is
validated at startup too.

```python
class SummariserAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(temperature=0.1, model_env="SUMMARISER_MODEL")
```

### 11. Token streaming

`main.py` consumes the graph with `stream_mode="messages"`. Two details
worth copying: **filter by node** (`STREAMING_NODES` — every LLM call
emits chunks, including structured-output extractions and tool traffic
that must not reach the user), and **fall back to state** for turns that
end in a non-streaming node. The same loop works for SSE/websockets —
`examples/fastapi_server.py` is exactly that, as an SSE endpoint.

### 12. Retries for transient failures

LLM-calling nodes are registered with `retry_policy=RetryPolicy(...)`,
so rate limits and timeouts retry with backoff at the graph level
instead of try/except in every agent. Know the default scope: LangGraph
retries connection errors, HTTP 5xx, and unrecognised exceptions, but
**not** `ValueError`/`TypeError`-style programming errors — and a
structured-output parse failure (`OutputParserException`) subclasses
`ValueError`, so a model that returns malformed JSON surfaces
immediately rather than retrying. Pass a custom `retry_on` if you want
different behaviour.

### 13. Testing with a fake LLM

`tests/` drives whole conversation turns through the compiled graph with
no network: `conftest.py` monkeypatches the LLM factory at the seam all
agents use (`lib.agent.get_llm`) and substitutes a recording fake
(`fakes.py`) that supports plain, structured, and tool-binding calls.
This exercises real routing, reducers, checkpointing, tool execution,
and trimming — the whole suite runs in a few seconds with no API key.

Structured results are **keyed by schema class**, so graphs with many
structured-output agents need no fake per agent:

```python
fake.structured_results[NameCheck] = NameCheck(name="Paul", reply="")
```

A structured call whose schema has no queued result **fails the test** —
proving a node did *not* run is simply not queueing its schema (see the
"onboarding is idempotent" test). Environment-dependent behaviour is
always stubbed: the startup-check tests monkeypatch `find_spec` and the
Ollama preflight so results never depend on which packages happen to be
installed on the machine running the tests.

### 14. Visualisation, Studio, CI

`python main.py --graph` prints Mermaid source (`lib/visualization.py`
also renders PNG). `langgraph dev` opens the graph in **LangGraph
Studio** for step-through debugging. GitHub Actions runs ruff + pytest
on Python 3.10/3.12/3.14 for every push and PR.

### 15. Image (vision) input

`image_message(text, path)` in `lib/agent.py` builds the
provider-agnostic content blocks for one image + prompt, and
`self.query_image_structured(...)` combines it with structured output —
the building blocks of any "analyse this picture" agent:

```python
result = await self.query_image_structured(
    "You are a photo analyst ...",  # system prompt
    "What is in this photo?",  # user text
    image_path,
    PhotoCheck,  # Pydantic schema
)
```

The chat flow in this template never calls it, but the content-block
format is the one thing you cannot verify offline — it is unit-tested
here, and you should still run one real call against your provider (the
configured model must support vision) before trusting a new one.

### 16. Side-effecting agents (batch pipelines)

This template's agents are *pure*: state in, state out, persistence
owned by the checkpointer. Agents that write outputs themselves — files,
database rows, API calls — follow a different recipe
(`examples/batch_pipeline.py` is a runnable demo of all four rules):

* **One graph run per unit of work** (one image, one document, one
  ticket) — the driver owns the loop, ordering, `--limit`-style options
  and progress reporting; the graph only knows about a single item.
* **Inject effect dependencies through agent constructors** (a store
  object wrapping the CSV/folder/API), never module-level paths —
  that keeps the fake-LLM seam intact and lets tests point agents at
  `tmp_path`.
* **Make nodes idempotent and let outputs double as resume state**: each
  node checks its own store ("does my row/file exist?") and no-ops when
  the work is done, so re-running an interrupted batch is always safe —
  often no checkpointer is needed at all.
* **Keep stores lazy** (no directory creation or file reads in
  `__init__`) so the import-time `graph = build_graph()` stays
  side-effect free (see pattern 2).

### 17. Logging

Stdlib `logging` is the vendor-agnostic seam. Two rules, both enforced
by tests:

* **Libraries emit, drivers configure.** Every `app/` module does only
  `logging.getLogger(__name__)` and emits at standard levels (DEBUG =
  diagnostics, INFO = one line per lifecycle event, WARNING = degraded,
  ERROR = failed). Handlers/formatters are set only by
  `configure_logging()` in `lib/log.py`, called from drivers
  (`main.py`, an adapter's `set_up()`) — never at import. An invariant
  test blocks config calls elsewhere.
* **Conversation content and profile values are PII** — they never
  appear in logs at any level (`test_no_conversation_content_in_logs`
  drives a turn and greps every record). Log events and metadata
  (durations, counts, `thread_id` via `extra=`), not text.

Configure via env: `LOG_LEVEL` (default INFO) and `LOG_FORMAT=json`
(one JSON object per line on stderr — extras like `thread_id`
included). Swapping vendors is one line at the driver, zero changes in
`app/`: any `logging.Handler` works — Sentry/Datadog handlers, syslog,
or OpenTelemetry's `LoggingHandler` for OTLP export to any backend.
(LangSmith covers LLM *tracing*; this is for application logs.)

**Google Cloud Logging sample.** On GCP (Cloud Run, GKE, Agent Engine)
use `GcpJsonFormatter` — Cloud Logging reads the level from a JSON
field named `severity`, and without it everything on stderr ingests as
ERROR:

```python
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(GcpJsonFormatter())
configure_logging(handlers=[handler])
```

The Agent Engine adapter's `set_up()` does exactly this. From *outside*
GCP, send logs via the API instead: `pip install google-cloud-logging`,
then pass its handler through the same seam —
`configure_logging(handlers=[CloudLoggingHandler(google.cloud.logging.Client())])`.

### 18. Evals — grading the model, not the wiring

The test suite proves the *wiring* with a fake LLM; **evals grade the
model** with real calls. They live in `evals/`, outside pytest's
`testpaths`, so the default `pytest` (and the Stop hook, and CI's
matrix) never runs them — run `pytest evals` when you change a prompt,
a model, or a provider. Without an API key they skip with an
explanation; a manual GitHub Actions workflow (`Evals`, run from the
Actions tab with an `OPENAI_API_KEY` secret) runs them on demand.

The three example evals are the three canonical types:

* **Programmatic scoring** (`test_greeter_extraction.py`) — realistic
  phrasings vs expected extracted names, including "must NOT extract"
  cases. Catches bad edits to the extraction prompt.
* **Trajectory checking** (`test_model_uses_the_time_tool`) — asserts
  the model chose to call the tool, read from state; the *path*, not
  the words.
* **LLM-as-judge** (`test_reply_quality_judged`) — `evals/judge.py`
  grades the reply against a rubric with a structured `Verdict`
  (Pydantic, no parsing). Set `EVAL_JUDGE_MODEL` to judge a cheap
  model's answers with a stronger one. The rubric mirrors
  `_SYSTEM_PROMPT`'s promises — edit one, update the other.

Honest caveat: evals are stochastic. A judge failure prints its
reasoning — read it before blaming the code; a flaky extraction case is
signal about your prompt or model, not an invitation to add retries.
When you outgrow inline cases: LangSmith datasets + `evaluate()` for
tracked runs, and LangChain's `openevals`/`agentevals` for prebuilt
judges and trajectory evaluators.
