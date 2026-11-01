# Repository Instructions

This repository holds `pamin-runtime`, the open-source compute crate shared by Pamin modules that run models locally: instruction-set kernel dispatch, inference devices over ONNX Runtime, per-call thread budgets and the prepared-model cache. It is published to crates.io and used by PaminMemory and other modules.

## Source of truth

- Specification, decisions and plans live in the private InternalDocs repository under `pamin-runtime/` (start with its `README.md`), plus the shared extension architecture in `pamin-platform/decisions/`. InternalDocs is a sibling clone at `$PAMIN_LABS_DEV_INTERNAL_DOCS` (default `../InternalDocs`); pull it before reading. Read the relevant documents before changing behaviour.
- When a design or decision changes, update InternalDocs in a PR there; cite it from here with SHA permalinks.
- This repository is public. Keep private planning, pricing, strategy and anything about paid or closed modules out of it; that lives in InternalDocs.

## Rules

- Everything committed is English: code, comments, logs, commit messages, PR titles and bodies, documents.
- Conventional Commits with a required scope for commits and PR titles (`fix(engine): ...`), one logical change per commit, each building and passing on its own.
- Coordinate through pamin-labs-dev before editing. With the plugin, its hooks and the `protocol` skill guide you. Without it, declare an intent with the `pamin-labs-dev` CLI first (`pamin-labs-dev intent declare ...`), work on its branch `<type>/i-<number>-<slug>` (no person or tool names), and put `Intent: I-<number>` in the PR description; CI rejects PRs without a live intent for their branch. Intents of the products this one uses appear in the session briefing; check them before changing code that depends on them.
- Keep PRs focused on one concern and reviewable in one sitting; split large work into a stack of PRs, each mergeable on its own. PR descriptions have a summary, the changes, how it was verified, and any risk or follow-up.
- Commits and PRs carry no agent attribution: no `Co-Authored-By` trailers, no session URLs, no "Generated with" footers.
- The commit author is always the engineer's own git identity; never invent one. If it is missing, stop and ask, then save it as `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL` in the environment settings.
- Never squash-merge; keep every commit.
- Never write tokens, keys or credentials to files, commits or logs.
- Prefer `rtk <command>` for shell commands when `rtk` is available.

## Code

The bar is high performance, low resource use, fast builds, and clean, DRY, decoupled code that is easy to maintain and extend. The `pamin-engineering` skill says how to meet it; follow it for every change.

- One implementation per concern: search for an existing one before adding logic, and extend it through its interface.
- Do not reinvent what exists: the standard library, then the platform, then a maintained, widely used dependency. This crate is a library: require dependencies with caret minimums and let lock files pin exact versions; pin exactly only with a stated reason, as for `ort` (InternalDocs `pamin-platform/decisions/2026-11-01-library-dependency-requirements.md`).
- No speculative abstractions, options or layers; add them when a second real use appears.
- Measure before and after any change that claims a speed-up or a smaller footprint, and put the numbers in the PR.
- Nothing here knows about MCP, HTTP, tenants or any module; it changes only when hardware or inference back ends change.
- Every kernel path is tested on its own hardware (x86-64, Linux aarch64, Apple silicon); AVX-512 stays out until a measurement beats AVX2.

## Gates before every push

`cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo deny check`.

## Skills

Longer guidance lives in `.agents/skills/`, one directory per skill; `.claude/skills/` links to the same directories.

| Skill | Use it when |
| --- | --- |
| `pamin-engineering` | Writing or changing code, adding a dependency, claiming a speed-up, reviewing |
