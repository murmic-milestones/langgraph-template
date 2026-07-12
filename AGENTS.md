# Instructions for AI coding agents

**Read [CLAUDE.md](CLAUDE.md) first** — it is the single source of truth
for commands, architecture, and this project's rules, regardless of
which AI tool you are. (It is named for Claude Code but written for any
agent; do not duplicate its content here.)

The short version:

- **Verify with** `pytest` (fast, no API key) and `ruff check . &&
  ruff format .` after every change. The suite includes *invariant
  tests* (`tests/test_template_invariants.py`) that enforce the
  architecture — if one fails, fix your code to match the pattern, don't
  edit the test.
- **Recipes** for the common workflows (add an onboarding stage, add a
  tool, add a model provider, remove an optional feature) live in
  `.claude/skills/*/SKILL.md` — follow them step by step even if your
  tool doesn't auto-load skills.
- **Keep docs in sync**: behaviour changes must update CLAUDE.md and the
  README section describing the pattern, in the same change.
- **Respect the comment markers**: `[tag]` = optional-feature lines,
  `enforced by tests/...` = a contract with an invariant test,
  `Customisation knob` = edit freely. Preserve and extend them.
- **Never read or write `.env`**; use `.env.example` for placeholders.
