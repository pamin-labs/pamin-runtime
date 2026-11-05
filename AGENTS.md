# Repository Instructions

This repository holds `pamin-runtime`, the open-source compute crate shared by Pamin modules that run models locally: instruction-set kernel dispatch, inference devices over ONNX Runtime, per-call thread budgets and the prepared-model cache. It is published to crates.io and used by pamin-memory and other modules.

## Source of truth

- Specification, decisions and plans live in the private InternalDocs repository under `pamin-runtime/` (start with its `README.md`), plus the shared extension architecture in `pamin-platform/decisions/`. InternalDocs is a sibling clone at `$PAMIN_LABS_DEV_INTERNAL_DOCS` (default `../InternalDocs`); pull it before reading. Read the relevant documents before changing behaviour.
- When a design or decision changes, update InternalDocs in a PR there; cite it from here with SHA permalinks.
- This repository is public. Keep private planning, pricing, strategy and anything about paid or closed modules out of it; that lives in InternalDocs.

## Rules

- Everything committed is English: code, comments, logs, commit messages, PR titles and bodies, documents.
- Conventional Commits with a required scope for commits and PR titles (`fix(engine): ...`), one logical change per commit, each building and passing on its own. A commit's first line and a PR title are at most 72 characters, scope included: the pamin-labs-dev hook refuses a longer commit and CI fails a longer title, and a pushed commit is not rewritten without the owner, so count before committing.
- Coordinate through pamin-labs-dev before editing. With the plugin, its hooks and the `protocol` skill guide you. Without it, declare an intent with the `pamin-labs-dev` CLI first (`pamin-labs-dev intent declare ...`), work on its branch `<type>/i-<number>-<slug>` (no person or tool names), and put `Intent: I-<number>` in the PR description; CI rejects PRs without a live intent for their branch. Intents of the products this one uses appear in the session briefing; check them before changing code that depends on them.
- Keep PRs focused on one concern and reviewable in one sitting; split large work into a stack of PRs, each mergeable on its own. PR descriptions have a summary, the changes, what it reused, how it was verified, and any risk or follow-up.
- Commits and PRs carry no agent attribution: no `Co-Authored-By` trailers, no session URLs, no "Generated with" footers. Some tools append such a footer when they create or edit a PR, so read the description back after every such call and remove it.
- The commit author is always the engineer's own git identity; never invent one. If it is missing, stop and ask, then save it as `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL` in the environment settings.
- Never squash-merge; keep every commit.
- **A PR merges only when its final commit is proven** (owner, 2026-11-05): GitHub reports it clean, every check has passed (a failed or pending check blocks it, however small the change), its tests pass, and, when it changes a service or binary that is deployed or shipped, an end-to-end check has passed on that same final commit: a local acceptance run or staging, never a run from before the last review fix.
- Work found and not done (a review's follow-ups, an audit's findings, a flaky test) goes into the maintainers' roadmap in InternalDocs in the same session, and a decision the owner makes goes into a decision record there before or with the change, so nothing lives only in a conversation.
- Never write tokens, keys or credentials to files, commits or logs.
- Prefer `rtk <command>` for shell commands when `rtk` is available.

## Code

The bar is high performance, low resource use, fast builds, and clean, DRY, decoupled code that is easy to maintain and extend. The `pamin-engineering` skill says how to meet it; follow it for every change.

- One implementation per concern: search for an existing one before adding logic, and extend it through its interface.
- Do not reinvent what exists: the repository's own implementation of the concern, then the official SDK of a service, then the standard library, then the platform, then a dependency already in the tree, then a maintained, widely used dependency; write your own only when none of them meets the need. A PR's "Reused" section names what it searched for, what it reused, and why anything was written by hand; hand-written code where one of those already covers the need is a defect, and review sends it back. This crate is a library: require dependencies with caret minimums and let lock files pin exact versions; pin exactly only with a stated reason, as for `ort` (InternalDocs `pamin-platform/decisions/2026-11-01-library-dependency-requirements.md`).
- No speculative abstractions, options or layers; add them when a second real use appears.
- **Model external data with types.** Parse external JSON (the plan cache's files, model metadata) into typed structs and enums that ignore unknown fields, and serialize requests from typed structs. No hand-built JSON strings, no chains of untyped lookups, no searching payloads for substrings. A value stays opaque only when a protocol requires it to be echoed back unchanged, and the code says so.
- **Tests meet the same bar as code.** A test builds requests from typed values, decodes an answer into the API's own types and compares JSON by decoding it, never as strings, and names a value owned elsewhere from its owner, like any code.
- **No deferral inside a PR's own code.** A written rule that code the PR adds or touches breaks is fixed in that PR, in its own commit; a follow-up is only work outside the PR's concern.
- **Comments claim only what the code guarantees.** A guarantee that waits on later work is named as not yet given.
- Before adding a dependency, check its maintenance, its licence and what it adds to build time and binary size, and record that in the PR.
- Measure before and after any change that claims a speed-up or a smaller footprint, and put the numbers in the PR.
- **No hard-coded value in more than one place.** Every port, address, URL, id, version, limit, price, file name and default is defined once, in the crate or file that owns it, and everything else reads it from there or is checked against it by a test or a generator. A literal that names a value owned elsewhere is a bug, in code, config, scripts, tests and docs alike.
- **No hidden defaults.** The crate deploys nothing, so it reads few settings; each environment variable it reads and each default it falls back to (a device selection, a thread budget, a cache directory) is named once, as a constant in the module that owns it, and documented there.
- **Built files are never committed:** `target/`, binaries, benchmark outputs and generated caches are git-ignored; only generated files a gate checks are committed.
- **Extensibility check before every PR:** adding one more of something the code already has (a kernel, a device, an execution provider, a prepared-model rewrite) touches its one definition plus its tests and docs. If the change needed more places, fold them into one source in the same PR.
- Nothing here knows about MCP, HTTP, tenants or any module; it changes only when hardware or inference back ends change.
- Every kernel path is tested on its own hardware (x86-64, Linux aarch64, Apple silicon); AVX-512 stays out until a measurement beats AVX2.

## Review before every PR

Before a PR opens, go through its whole diff, tests included, against each rule of every `AGENTS.md` the change touches and the `pamin-engineering` skill, and fix what misses in new commits; then run the `pamin-labs-dev:peer-review` skill and put its verdict and findings under the PR's verification. Work an agent delegated is reviewed the same way: the agent that opens the PR reads its diff and tests itself and never accepts them on the delegate's report alone.

## Gates before every push

`cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo deny check`.

## Skills

Longer guidance lives in `.agents/skills/`, one directory per skill; `.claude/skills/` links to the same directories.

| Skill | Use it when |
| --- | --- |
| `pamin-engineering` | Writing or changing code, adding a dependency, claiming a speed-up, reviewing |
