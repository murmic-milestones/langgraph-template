# LangGraph, Explained Like You're 5 🧒

You're about to build an AI agent. Good news: you don't need to know
much. This page explains the whole idea in plain words, using the real
code in this folder. Read it top to bottom — it's short on purpose.

---

## Part 1: The big idea

### What's an "agent"?

A chatbot that can **think in steps** and **do things** (not just talk).
LangGraph is the Lego kit for building one.

### The graph = a board game path 🎲

Your agent is a little flowchart. Every user message takes **one trip**
across the board, from START to END:

```
START → collect_name → chat → END
```

Ours does exactly two things: learn your name, then chat with you.
That's the whole game. See it drawn for real:

```bash
python main.py --graph     # paste the output into https://mermaid.live
```

### Nodes = workers 👷

Each square on the board is a **node**: a function that does one job.
Our two workers live in [app/agents/](app/agents/):

- `greeter.py` — "Do we know your name yet? No? I'll ask."
- `chat.py` — "I write the actual reply."

### State = the backpack 🎒

Every worker gets handed the same backpack, looks inside, does its job,
and drops something new in. The backpack is defined in
[app/state.py](app/state.py) and holds just two things:

- `messages` — the conversation so far
- `profile` — facts we've learned (like your name)

One magic rule: workers put in only their **new** stuff (the last reply,
one new fact). LangGraph handles adding it to the pile.

### Edges = the arrows, gates = the bouncers 🚦

Arrows say what happens next. Some arrows have a **gate** — a tiny
question that picks the path:

```python
# app/graph.py — real code:
builder.add_conditional_edges(
    "collect_name",
    greeter.is_name_set,          # the question: "got a name yet?"
    {True: "chat", False: END},   # yes → chat, no → stop and wait
)
```

If we don't know your name, the trip ends after asking for it. Your
answer starts the *next* trip — and this time the gate says go.

### The checkpointer = save slots 💾

Between trips, the backpack is saved under a `thread_id` — like save
slots in a video game. Alice's slot never touches Bob's slot. That's
the entire secret of how the bot "remembers" you.

### Tools = giving the robot hands 🤖✋

The chat worker can use **tools** — plain Python functions it's allowed
to call. Ask "what time is it?" and the model says *"run
`get_current_time` for me, please"*, the graph runs it, and the answer
goes back to the model:

```
chat → tools → chat → END
```

All tools live in [app/tools.py](app/tools.py). One file, one list.

That's it. Graph, nodes, state, gates, save slots, tools. Everything
else is detail.

---

## Part 2: Build your chatbot, step by step

### Step 0 — What you need

Python 3.10+ and an OpenAI API key (or another provider — see the
README's provider table).

### Step 1 — Get the code and install

```bash
# copy this folder to a new one (don't build inside the template!)
cd my-first-agent
git init
python -m venv .venv
.venv\Scripts\activate            # Windows; Mac/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

### Step 2 — Add your key

```bash
copy .env.example .env            # Mac/Linux: cp .env.example .env
```

Open `.env` and paste your key into `OPENAI_API_KEY=`. Never share or
commit this file (it's already gitignored).

### Step 3 — Talk to it! 🎉

```bash
python main.py
```

It asks your name, remembers it, and chats. Ask "what time is it?" to
watch it use a tool. Congratulations — you're running an agent.

### Step 4 — Prove it works (do this every time you change code)

```bash
pytest
```

34 tests, ~1 second, **zero API cost** — the tests swap the real AI for
a fake one (`tests/fakes.py`), so they check *your wiring*, not
OpenAI's servers. Green = safe to continue. This is the habit that
separates toys from real projects: **change code → run pytest.**

### Step 5 — Make it yours (first tiny change)

Open [app/agents/chat.py](app/agents/chat.py) and edit
`_SYSTEM_PROMPT` — make the bot a pirate, a chef, whatever:

```python
_SYSTEM_PROMPT = """\
You are a cheerful pirate assistant.
The user's name is {name}; call them 'Cap'n {name}'.
Keep replies short and salty.
"""
```

Run `python main.py` to enjoy it, then `pytest` to confirm nothing
broke.

### Step 6 — Add your first tool

Open [app/tools.py](app/tools.py) and add:

```python
import random

@tool
def roll_dice(sides: int = 6) -> int:
    """Roll one die with the given number of sides."""
    return random.randint(1, sides)

TOOLS = [get_current_time, roll_dice]   # <- add it to the list
```

That's the *entire* job — the docstring is how the model knows what the
tool does, so write it clearly. Run the bot and ask it to roll a d20.
Then (say it with me) run `pytest`.

### Step 7 — Add a test for your tool

Copy the shape of `test_tool_calling_loop` in
[tests/test_graph.py](tests/test_graph.py): queue a fake reply that
asks for `roll_dice`, run one turn, check a tool message appeared. Ten
lines, and your feature is protected forever.

### Step 8 — Keep the robot-helper notes fresh 📝

This project has a [CLAUDE.md](CLAUDE.md) file — instructions for AI
coding assistants (Claude Code and friends). It lists the commands, the
architecture, and the traps. **When you change how something works,
update CLAUDE.md too** — future-you, teammates, and AI assistants all
read it, and stale notes are worse than none. Same goes for the README
if you change a pattern it describes.

### Step 9 — Level up (when you're ready)

- **New onboarding step?** Copy `greeter.py` as a recipe: one node
  method that collects a fact, one gate method that checks it, two
  lines of wiring in `graph.py`.
- **See inside the machine:** `langgraph dev` opens LangGraph Studio —
  step through your graph node by node.
- **Everything else** (streaming, providers, deployment, removing
  features you don't need) is in the [README](README.md).

---

## Cheat sheet 🗺️

| Concept | Plain words | Where |
|---|---|---|
| Graph | The board game path | `app/graph.py` |
| Node | A worker doing one job | `app/agents/` |
| State | The shared backpack | `app/state.py` |
| Gate | A bouncer picking the path | `is_name_set` in `greeter.py` |
| Checkpointer | Save slots per conversation | `build_graph(checkpointer=...)` |
| Tools | The robot's hands | `app/tools.py` |
| Tests | Your safety net (free + fast) | `tests/` — run `pytest` |
| CLAUDE.md | Notes for AI helpers | keep it updated! |

**The golden loop:** change one small thing → `python main.py` to feel
it → `pytest` to prove it → update CLAUDE.md/README if behaviour
changed → repeat. Welcome to agent building. 🚀
