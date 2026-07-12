# LangGraph Starter Template

A minimal, modern **LangGraph 1.x** agent project. It implements a small
two-stage chatbot — collect the user's name, then chat (with tool
calling) — purely as a vehicle for the architecture patterns below.
Replace the agents with your own and keep the skeleton.

> **New to LangGraph?** Start with [ELI5.md](ELI5.md) — a plain-words
> intro to the concepts plus a step-by-step first-chatbot guide.

```
langgraph-template/
├── main.py                 # async CLI entry point (streaming chat loop)
├── pyproject.toml          # packaging, deps, ruff + pytest config
├── langgraph.json          # LangGraph Studio / platform config
├── LICENSE                 # 0BSD — permissive, no attribution required
├── .env.example            # copy to .env and fill in
├── .github/workflows/ci.yml# lint + format + tests on 3.10/3.12/3.14
├── AGENTS.md               # entry point for AI coding tools -> CLAUDE.md
├── .claude/                # AI-tooling config: permissions, hooks, skills
├── evals/                  # real-model evals: pytest evals  [removable]
├── examples/
│   ├── human_approval.py   # standalone interrupt() demo   [removable]
│   └── agent_engine_app.py # Google Agent Engine adapter   [removable]
├── tests/
│   ├── conftest.py         # the fake-LLM fixture
│   ├── fakes.py            # recording FakeLLM + helpers
│   ├── test_graph.py       # end-to-end graph tests
│   ├── test_agents_base.py # BaseAgent helpers (models, image input)
│   ├── test_environment.py # startup-check tests
│   ├── test_llm.py         # model factory resolution/caching
│   ├── test_template_invariants.py # architecture rules as tests
│   ├── test_logging.py     # log levels, JSON format, PII rule
│   ├── test_persistence.py # SQLite durability             [removable]
│   ├── test_examples.py    # interrupt demo tests          [removable]
│   └── test_agent_engine.py# Agent Engine adapter tests    [removable]
└── app/
    ├── state.py            # typed state schema + reducers
    ├── log.py              # logging config (the vendor seam)
    ├── llm.py              # provider-agnostic model factory
    ├── env.py              # provider registry + startup checks
    ├── graph.py            # graph assembly, retries, Studio entry point
    ├── tools.py            # tools for the chat agent       [removable]
    ├── visualization.py    # Mermaid export helpers
    └── agents/
        ├── base.py         # BaseAgent: LLM, structured-output + image plumbing
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
   pytest                          # 45 tests, no API key needed
   pytest evals                    # model-quality evals (REAL calls, costs money)
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

### 10. Provider-agnostic model factory, per-agent overrides

`app/llm.py` uses `init_chat_model`, so the model — including the
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
end in a non-streaming node. The same loop works for SSE/websockets.

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
agents use (`app.agents.base.get_llm`) and substitutes a recording fake
(`fakes.py`) that supports plain, structured, and tool-binding calls.
This exercises real routing, reducers, checkpointing, tool execution,
and trimming — the whole suite runs in well under a second.

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

`python main.py --graph` prints Mermaid source (`app/visualization.py`
also renders PNG). `langgraph dev` opens the graph in **LangGraph
Studio** for step-through debugging. GitHub Actions runs ruff + pytest
on Python 3.10/3.12/3.14 for every push and PR.

### 15. Image (vision) input

`image_message(text, path)` in `app/agents/base.py` builds the
provider-agnostic content blocks for one image + prompt, and
`self.query_image_structured(...)` combines it with structured output —
the building blocks of any "analyse this picture" agent:

```python
result = await self.query_image_structured(
    "You are a photo analyst ...",     # system prompt
    "What is in this photo?",          # user text
    image_path,
    PhotoCheck,                        # Pydantic schema
)
```

The chat flow in this template never calls it, but the content-block
format is the one thing you cannot verify offline — it is unit-tested
here, and you should still run one real call against your provider (the
configured model must support vision) before trusting a new one.

### 16. Side-effecting agents (batch pipelines)

This template's agents are *pure*: state in, state out, persistence
owned by the checkpointer. Agents that write outputs themselves — files,
database rows, API calls — follow a different recipe (proven out in a
sibling batch-pipeline project):

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
  `configure_logging()` in `app/log.py`, called from drivers
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

## Optional features — how to add or remove

Each feature is self-contained and marked with a bracketed tag in code
comments. Removal never requires understanding the feature's internals.

| Feature | Lives in | Remove by |
|---|---|---|
| Tool calling `[tools]` | `app/tools.py`, 2 marked lines in `chat.py`, 4 in `graph.py` | steps listed in `app/tools.py` docstring |
| History trimming `[trim]` | `app/agents/chat.py` | delete the `trim_messages` call, pass `state["messages"]` |
| SQLite sessions `[sqlite]` | `main.py` `--db` blocks, `tests/test_persistence.py` | delete the marked blocks + test + `langgraph-checkpoint-sqlite` dep |
| interrupt() demo | `examples/human_approval.py`, `tests/test_examples.py` | delete both files |
| Agent Engine (GCP) | `examples/agent_engine_app.py`, `tests/test_agent_engine.py`, `[vertexai]` extra | delete all three |
| Evals | `evals/`, `.github/workflows/evals.yml` | delete both |

## Model providers

The model — including the provider — is just the `MODEL_NAME` env string
(`provider:model`, resolved by `init_chat_model`). Four providers are
supported out of the box:

| Provider | Install | `MODEL_NAME` example | Key |
|---|---|---|---|
| OpenAI | `pip install -e "."` (default) | `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `pip install -e ".[anthropic]"` | `anthropic:claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| Gemini (API key) | `pip install -e ".[google]"` | `google_genai:gemini-2.5-flash` | `GOOGLE_API_KEY` |
| Gemini (Vertex AI) | `pip install -e ".[vertexai]"` | `google_vertexai:gemini-2.5-flash` | none (ADC†) |
| Ollama | `pip install -e ".[ollama]"` | `ollama:llama3.2` | none (local) |

No code changes to switch — install the extra, set `MODEL_NAME` and the
key in `.env`. Ollama runs models locally: start the server (`ollama
serve` or the desktop app) and pull the model (`ollama pull llama3.2`)
first. Note that the greeter relies on structured output and the chat
stage on tool calling, so pick an Ollama model that supports tools.

`check_environment()` in `app/env.py` validates every configured model
(the `MODEL_NAME` default plus any per-agent override variables you pass
via `extra_model_vars`) before the first run: provider package present,
API key set, and — where the provider row defines a `preflight` — extra
checks like pinging the local Ollama server. Call it from any driver you
write, not just the CLI. To add another provider: one `Provider` row in
`app/env.py`, one extra in `pyproject.toml`, one example line in
`.env.example`.

## Working with AI coding tools

The template ships configured for AI-assisted development (Claude Code
and compatible tools), and the configuration copies into every derived
project:

* **Invariant tests** (`tests/test_template_invariants.py`) encode the
  architecture rules — providers synced across config files, agents
  using the `get_llm` seam, async nodes / sync gates — so violating a
  pattern fails `pytest` instead of slipping through review. Fix the
  code, not the test.
* **`.claude/settings.json`** (committed) pre-approves the safe
  verification commands (`pytest`, `ruff`, `python main.py`,
  `langgraph dev`) and denies reading `.env` — the AI can run the
  verify loop without permission prompts and without your secrets.
* **Hooks** make the two core habits deterministic: every file the AI
  writes is auto-formatted with ruff (PostToolUse), and the AI cannot
  declare itself done while `pytest` is red (Stop hook — affordable
  because the fake-LLM suite runs in a few seconds with no API key).
  Hook scripts are Python for Windows/macOS/Linux parity.
* **Skills** (`.claude/skills/`) encode the four recipes — `add-stage`,
  `add-tool`, `add-provider`, `remove-feature` — so the sanctioned path
  is also the easiest one.
* **`AGENTS.md`** points non-Claude tools at the same CLAUDE.md
  instructions.
* **Comment anchors** — grep-able conventions in the source: `[tag]`
  marks an optional feature's lines, `enforced by tests/...` marks a
  contract with its invariant test, and `Customisation knob` marks
  lines meant to be edited freely (e.g. the chat system prompt). Full
  explanations live in exactly one place; everything else points to it.
  Preserve and extend these markers when editing.

Personal overrides go in `.claude/settings.local.json` (gitignored).

## Deploying to Google Agent Engine (Gemini Enterprise Agent Platform)

`examples/agent_engine_app.py` wraps the graph in the platform's
custom-agent contract: a pickle-able class with a config-only
`__init__`, graph construction in `set_up()` (server-side), and
`query()`/`async_query()` entry points returning JSON-serialisable
results. Sessions map the platform conversation onto the checkpointer's
`thread_id`, exactly like every other driver in this template.

```bash
pip install -e ".[vertexai]"
gcloud auth application-default login
```

then follow the deploy snippet in the module docstring
(`vertexai.init(...)` + `agent_engines.create(AgentEngineApp(...),
requirements=[...], extra_packages=["app", "examples"])`). Notes:

* The default `InMemorySaver` keeps sessions per-container; for real
  deployments swap in a durable saver (Cloud SQL / AlloyDB, see the
  comment in `set_up()`).
* On the platform, prefer `google_vertexai:...` models — they
  authenticate via the runtime's service account, no API key to manage.
* The adapter's local contract (pickling, set_up, query round-trip) is
  covered by `tests/test_agent_engine.py` with the fake LLM; validate a
  real deployment with one `remote.query(...)` call.

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
