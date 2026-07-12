# Project: LangGraph Starter Template

Hello-world LangGraph 1.x agent (collect name → chat with tools) meant
to be copied as the skeleton for new projects. Keep it minimal: it
exists to demonstrate patterns, not accumulate features. README.md
documents each pattern — update it whenever the pattern it describes
changes.

## Commands

```
pip install -e ".[dev]"   # deps live in pyproject.toml (no requirements.txt)
python main.py            # async CLI chat loop (needs .env)
python main.py --db x.db  # durable sessions (SQLite)
python main.py --graph    # print graph as Mermaid source
pytest                    # fake-LLM tests, no API key needed
pytest evals              # model-quality evals — REAL calls, costs money
ruff check . && ruff format .   # CI enforces both
langgraph dev             # LangGraph Studio
```

## Architecture

- `app/graph.py` — graph wiring, RetryPolicy, two entry points (see
  gotcha below). Nodes/edges are registered here only.
- `app/state.py` — TypedDict state; `messages` uses the `add_messages`
  reducer (append), `profile` overwrites. Nodes return partial updates.
- `app/agents/` — one class per stage: **async** node methods return
  state updates, sync gate methods route conditional edges. Agents are
  shared across sessions — keep them stateless. Per-agent model
  overrides via `BaseAgent(model_env="MY_STAGE_MODEL")`; image input via
  `image_message()` / `query_image_structured()` in `base.py`.
- `app/log.py` — the only module that configures logging (enforced by
  an invariant test). `app/` modules just do
  `logging.getLogger(__name__)` and emit; drivers call
  `configure_logging()` (env: `LOG_LEVEL`, `LOG_FORMAT=text|json`).
  Levels: DEBUG diagnostics, INFO one line per lifecycle event,
  WARNING degraded, ERROR failed. Conversation content and profile
  values are PII — never log them at any level (enforced by
  `test_no_conversation_content_in_logs`); log events + metadata
  (durations, counts, `thread_id` via `extra=`).
- `app/llm.py` — the only place a model is constructed
  (`init_chat_model`, provider comes from `MODEL_NAME` env, explicit
  `model=` arg for per-agent overrides).
- `app/env.py` — provider registry (`PROVIDERS`, with optional
  `preflight` checks like the Ollama server ping) and
  `check_environment(extra_model_vars=...)`, which every driver should
  call at startup. Failures raise `EnvironmentCheckError` (never
  `sys.exit` — drivers decide; `main.py` translates it to a clean
  exit). Bare model names are checked against the provider
  `init_chat_model` infers, not assumed to be OpenAI. Providers are
  defined in three places that must stay in sync: `PROVIDERS` here, the
  extras in `pyproject.toml`, and the table in `.env.example`/README.
- `app/tools.py` — tool list for the chat ⇄ tools loop.
- One graph run per chat turn; incomplete onboarding ends the run and
  the next invoke re-enters from START, so nodes must be idempotent.

## Optional features

Tool calling `[tools]`, prompt trimming `[trim]`, SQLite sessions
`[sqlite]`, the `examples/` interrupt demo, and the Google Agent
Engine adapter (`examples/agent_engine_app.py` + `[vertexai]` extra;
config-only pickled `__init__`, graph built in `set_up()`, never call
`check_environment` there) are deliberately loosely coupled. Code blocks belonging to each are marked with the
bracketed tag, and removal steps live in the feature's home file
(`app/tools.py`, `app/agents/chat.py`, `main.py`, `examples/`). When
editing near a marked block, preserve the tag comments and keep the
feature removable; when asked to remove a feature, follow its
documented steps including deleting the matching tests.

## AI-tooling layer (ships with the template)

- `tests/test_template_invariants.py` encodes the architecture rules
  (provider sync, the `get_llm` seam, async nodes / sync gates). When
  one fails, fix the code to match the pattern — only change the test
  if the pattern itself is deliberately changing, with docs updated in
  the same commit.
- `.claude/settings.json` is committed team config (permissions +
  hooks); personal overrides belong in `.claude/settings.local.json`.
- Hooks: every Python file you write is ruff-formatted automatically
  (PostToolUse), and a Stop hook blocks finishing while `pytest` is
  red. Keep hook scripts pure Python and ASCII-output (Windows parity),
  and keep the suite fast and API-key-free or the Stop hook becomes
  unaffordable.
- Recipes for recurring work live in `.claude/skills/` (`add-stage`,
  `add-tool`, `add-provider`, `remove-feature`) — follow them rather
  than improvising, and update them when the recipe changes.
- Evals (`evals/`) call real models and cost money: never add them to
  pytest `testpaths`, the Stop hook, or CI's default matrix (the
  `Evals` workflow is manual-dispatch only). Run them when a prompt,
  model, or provider changes. The judge rubric in
  `evals/test_chat_quality.py` mirrors `_SYSTEM_PROMPT`'s promises —
  editing one means updating the other. Judge failures print their
  reasoning: read it before changing code; don't add retries to mask
  stochastic flakes.
- Comment markers are grep-able conventions — preserve and reuse them:
  `[tag]` marks an optional feature's lines; "enforced by tests/..."
  marks a contract with a matching invariant test; "Customisation
  knob" marks lines meant to be edited freely. Explanations live in
  exactly one place; everywhere else points to it — never restate.

## Security (see README "Security")

- `thread_id` is the only session isolation and the graph does no
  per-user authz — derive it from an authenticated identity, never a
  client value; the Agent Engine adapter requires it (no default) on
  purpose. Don't reintroduce a default/shared thread_id.
- Tools run under prompt injection: the model picks tool + arguments
  from a user-steerable conversation. New tools must treat arguments as
  attacker-controlled (validate paths/URLs/ids, least privilege, gate
  irreversible actions). Keep the SECURITY note in `app/tools.py`.
- Conversation history is PII at rest (checkpointer stores it
  unencrypted) and must never be logged (enforced by a test).

## Watch out for

- Two graph entry points: module-level `graph` (no checkpointer — used
  by langgraph.json / Studio, which inject persistence) vs
  `build_graph(checkpointer=...)` for CLI/tests/servers. Don't add a
  default checkpointer back to `build_graph`.
- `graph = build_graph()` runs at import time — everything it constructs
  must be side-effect free until first use (no file/network I/O or
  env-var snapshots in constructors; make such dependencies lazy).
- RetryPolicy does NOT retry ValueError-family errors — including
  `OutputParserException` (malformed structured output), which
  subclasses ValueError. Only connection errors, HTTP 5xx, and unknown
  exceptions retry by default.
- Nodes are async — call the graph with `ainvoke`/`astream`. Sync
  `invoke` raises. Tests wrap calls in `run()` from `tests/fakes.py`.
- `STREAMING_NODES` in `main.py` whitelists which nodes stream tokens.
  Any new LLM-calling node is silent until added there; never add nodes
  that emit structured-output JSON or raw tool traffic.
- Tests monkeypatch `app.agents.base.get_llm` — new agents must fetch
  the model via `BaseAgent.llm`, not construct one directly. The fake's
  structured results are keyed by schema class
  (`fake.structured_results[MySchema] = ...`); an un-queued schema
  makes the call fail loudly, which is how "node did not run" is
  asserted. Never let tests depend on installed packages or live
  servers — stub `find_spec` and preflights (see test_environment.py).
- `get_llm` re-reads the env on every call and caches instances per
  resolved (model, temperature) pair — `MODEL_NAME` and per-agent
  override variables have identical semantics. A cached instance's
  async HTTP client stays bound to the event loop that first used it,
  so any code path doing consecutive `asyncio.run()` calls in one
  process must call `reset_llm_cache()` between them (the evals and
  the Agent Engine adapter's sync `query()` both do).
- Trimming affects only what is *sent* to the model
  (`MAX_HISTORY_MESSAGES` in `chat.py`); state keeps the full history.
- Keep dependency pins to major 1.x ranges (verified against langgraph
  1.2.9). CI runs ruff check + format + pytest on 3.10/3.12/3.14 —
  run all three locally before pushing.
- License is 0BSD — no attribution required; don't add license headers
  to source files.
