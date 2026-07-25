# LangGraph Starter Template

A minimal, modern **LangGraph 1.x** agent project. It implements a small
two-stage chatbot — collect the user's name, then chat (with tool
calling) — purely as a vehicle for the architecture patterns it
demonstrates. Replace the agents with your own and keep the skeleton.

> **New to LangGraph?** Start with [ELI5.md](ELI5.md) — a plain-words
> intro to the concepts plus a step-by-step first-chatbot guide.

**Contents** — [Quick start](#starting-a-new-project-from-this-template) ·
[Patterns](PATTERNS.md) · [Optional features](#optional-features--how-to-add-or-remove) ·
[Model providers](#model-providers) · [Docker](#docker) ·
[AI coding tools](#working-with-ai-coding-tools) ·
[Deploy: Agent Engine](#deploying-to-google-agent-engine-gemini-enterprise-agent-platform) ·
[Serving over HTTP](#serving-over-http) · [Security](#security)

```
langgraph-template/
├── main.py                 # async CLI entry point (streaming chat loop)
├── pyproject.toml          # packaging, deps, ruff + pytest config
├── langgraph.json          # LangGraph Studio / platform config
├── Dockerfile              # container for the server example [removable]
├── compose.yaml            #   `docker compose up --build`
├── LICENSE                 # 0BSD — permissive, no attribution required
├── .env.example            # copy to .env and fill in
├── .github/workflows/ci.yml# lint + format + tests on 3.10/3.12/3.14
├── AGENTS.md               # entry point for AI coding tools -> CLAUDE.md
├── .claude/                # AI-tooling config: permissions, hooks, skills
├── evals/                  # real-model evals: pytest evals  [removable]
├── examples/               # runnable pattern demos, each  [removable]
│   ├── human_approval.py   #   bare interrupt() mechanism (no LLM)
│   ├── tool_approval.py    #   approval gate for dangerous tool calls
│   ├── fastapi_server.py   #   HTTP + SSE serving (FastAPI, [serve] extra)
│   ├── batch_pipeline.py   #   side-effecting batch pipeline (pattern 16)
│   ├── parallel_fanout.py  #   Send API map-reduce
│   ├── long_term_memory.py #   cross-thread memory (Store API, no LLM)
│   ├── time_travel.py      #   checkpoint history + forking (no LLM)
│   └── agent_engine_app.py #   Google Agent Engine adapter
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
│   └── test_<example>.py   # one per examples/ demo        [removable]
├── app/                    # THIS agent — tightly coupled, replace freely
│   ├── state.py            #   typed state schema + reducers
│   ├── graph.py            #   graph assembly, retries, Studio entry point
│   └── agents/
│       ├── greeter.py      #   onboarding stage (node + gate)
│       └── chat.py         #   main conversation stage (tools + trimming)
├── lib/                    # reusable LangGraph helpers — project-agnostic,
│   ├── llm.py              #   liftable into any project unchanged
│   ├── env.py              #   provider registry + startup checks
│   ├── log.py              #   logging config (the vendor seam)
│   ├── visualization.py    #   Mermaid export helpers
│   └── agent.py            #   BaseAgent: LLM, structured-output + image plumbing
└── tools/                  # tool registry — one tool per file  [removable]
    ├── __init__.py         #   aggregates TOOLS + how-to/security docstring
    └── get_current_time.py #   the example tool
```

**Layering.** `app/` holds only what is specific to *this* agent (its
state schema, its graph wiring, its concrete stages). Everything reusable
across LangGraph projects lives adjacent to it: `lib/` (model factory,
provider/env checks, logging, visualisation, the `BaseAgent` base class)
and `tools/` (one `@tool` per file). Copy `lib/` and `tools/` into another
project unchanged; rewrite `app/`.

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
   pytest, ruff, the LangGraph CLI for Studio, the SQLite saver, and
   fastapi so the server example's tests run.)

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
   pytest                          # 68 tests, no API key needed
   pytest evals                    # model-quality evals (REAL calls, costs money)
   ruff check . && ruff format .   # lint + format
   langgraph dev                   # open the graph in LangGraph Studio
   python examples/human_approval.py   # every examples/ demo is runnable —
                                       # the run command is in its docstring
   ```

5. **Make it yours:**

   * Rename `pyproject.toml`'s `name`, update `LICENSE`'s copyright line.
   * Replace the agents in `app/agents/`, add your tools in
     `tools/__init__.py`, extend `Profile`/`AppState` in `app/state.py`,
     register nodes in `app/graph.py`.
   * Remove the optional features you don't need — each one lists its
     removal steps where it lives (see "Optional features" below).

## Architecture patterns

The template demonstrates 18 patterns — one graph run per
turn, checkpointer sessions, async nodes, typed state, tool calling,
structured output, streaming, retries, logging, evals, and more. Each is
documented with the file that implements it in **[PATTERNS.md](PATTERNS.md)**.

You don't need them to get started: install, add a key, `python main.py`.
Read a pattern when you want to change the thing it describes.

## Optional features — how to add or remove

Each feature is self-contained and marked with a bracketed tag in code
comments. Removal never requires understanding the feature's internals.

| Feature | Lives in | Remove by |
|---|---|---|
| Tool calling `[tools]` | `tools/`, plus every `[tools]`-marked line in `app/agents/chat.py` and `app/graph.py` | steps listed in `tools/__init__.py` docstring |
| History trimming `[trim]` | `app/agents/chat.py` | delete the `max_messages=` argument in `respond()` |
| SQLite sessions `[sqlite]` | `main.py` `--db` blocks, `tests/test_persistence.py` | delete the marked blocks + test + `langgraph-checkpoint-sqlite` dep |
| interrupt() demo | `examples/human_approval.py`, `tests/test_human_approval.py` | delete both files |
| Tool-approval gate | `examples/tool_approval.py`, `tests/test_tool_approval.py` | delete both files |
| FastAPI server | `examples/fastapi_server.py`, `tests/test_fastapi_server.py`, `[serve]` extra + fastapi in `dev` | delete all four |
| Batch pipeline | `examples/batch_pipeline.py`, `tests/test_batch_pipeline.py` | delete both files |
| Parallel fan-out | `examples/parallel_fanout.py`, `tests/test_parallel_fanout.py` | delete both files |
| Long-term memory | `examples/long_term_memory.py`, `tests/test_long_term_memory.py` | delete both files |
| Time travel | `examples/time_travel.py`, `tests/test_time_travel.py` | delete both files |
| Agent Engine (GCP) | `examples/agent_engine_app.py`, `tests/test_agent_engine.py`, `[vertexai]` extra | delete all three |
| Evals | `evals/`, `.github/workflows/evals.yml` | delete both |
| Docker | `Dockerfile`, `.dockerignore`, `compose.yaml`, README "Docker" | delete all four |

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

`check_environment()` in `lib/env.py` validates every configured model
(the `MODEL_NAME` default plus any per-agent override variables you pass
via `extra_model_vars`) before the first run: provider package present,
API key set, and — where the provider row defines a `preflight` — extra
checks like pinging the local Ollama server. A bare model name (no
`provider:` prefix) is checked against the provider `init_chat_model`
will actually infer for it (`gpt-*` → OpenAI, `claude-*` → Anthropic,
…). Call it from any driver you write, not just the CLI; failures raise
`EnvironmentCheckError` with a fix-it message, and each driver picks its
reaction — `main.py` turns it into a clean exit, a server should log it
and refuse to start. To add another provider: one `Provider` row in
`lib/env.py`, one extra in `pyproject.toml`, one example line in
`.env.example`.

## Docker

The container runs the FastAPI server example (`examples/fastapi_server.py`)
— the deployable surface; `main.py` is the local-dev CLI.

```bash
docker build -t my-agent .
docker run --rm -p 8000:8000 --env-file .env my-agent
```

or, in one step:

```bash
docker compose up --build       # http://localhost:8000
```

Notes: secrets are passed at run time (`--env-file` / your orchestrator's
secret store) and `.dockerignore` excludes `.env`, so a stray `COPY . .`
can't bake one into an image; the image runs as a **non-root** user and
sets `LOG_FORMAT=json` so collectors parse the logs natively. Sessions use
`InMemorySaver` — they live as long as the container, so swap in a
Postgres/SQLite saver before running more than one replica. To containerise
a different entry point, change the `CMD`.

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
* The sync `query()` entry point runs `asyncio.run` per call, so it
  resets the model cache each time (a cached model's HTTP client stays
  bound to the loop that created it — see `reset_llm_cache`); the async
  path keeps the cache and is the one to prefer under load.
* The adapter's local contract (pickling, set_up, query round-trip) is
  covered by `tests/test_agent_engine.py` with the fake LLM; validate a
  real deployment with one `remote.query(...)` call.

## Serving over HTTP

The graph is transport-agnostic. `examples/fastapi_server.py` is a
working server — JSON endpoint plus SSE token streaming; install with
`pip install -e ".[serve]"`. The essence is a handler like this — note
the `thread_id` comes from the **authenticated** user, never from the
request body/path (see Security below):

```python
graph = build_graph(checkpointer=my_durable_checkpointer)


@app.post("/chat")
async def chat(text: str, user=Depends(current_user)) -> dict:
    # thread_id is derived server-side from who is logged in.
    config = {"configurable": {"thread_id": f"user:{user.id}"}}
    state = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config)
    return {"reply": state["messages"][-1].text}
```

For streaming responses, see the `/chat/stream` SSE endpoint in
`examples/fastapi_server.py` — it is `run_turn` from `main.py` yielding
SSE events instead of printing.

## Security

This is a starter template — review the trust model before shipping it.
Two issues live above the code and are easy to get wrong:

* **Session isolation is your job.** `thread_id` is the *only* thing
  separating one user's conversation from another's, and the graph does
  **no** per-user authorization. Always derive `thread_id` from an
  authenticated server-side identity; never accept a client-supplied
  one. A client that picks another user's `thread_id` reads and
  continues that user's conversation (an IDOR), and a shared/default id
  merges everyone into one session — which is why the Agent Engine
  adapter requires an explicit `thread_id` rather than defaulting.
* **Tools run under prompt injection.** The chat model is bound to the
  tools in `tools/__init__.py`, and a user can steer *which* tool it calls
  and *with what arguments* via a crafted message. Treat every tool
  argument as attacker-controlled: keep tools least-privilege, validate
  inputs, and gate irreversible actions behind human approval
  (`examples/tool_approval.py` is the wiring to copy). Full guidance is
  in `tools/__init__.py`.

Also worth a look before production: conversation history is stored
**unencrypted at rest** by the checkpointer (the SQLite `--db` file, or
your Postgres) — it is PII, so apply encryption/retention as your policy
requires; LLM-as-judge evals are themselves injectable, so don't trust
eval scores on adversarial input; and pin dependencies (hashes) for a
locked-down supply chain. Already handled by the template: no
conversation content in logs, `.env` reads denied to AI tools, the
Ollama URL is scheme-checked, `image_message` guards path/size, and the
secret-bearing CI workflow is manual-dispatch only (never PR-triggered).

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

