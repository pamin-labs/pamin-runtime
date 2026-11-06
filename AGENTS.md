# Repository Instructions

This repository holds `pamin-runtime`, the open-source compute crate shared by Pamin modules that run models locally: instruction-set kernel dispatch, inference devices over ONNX Runtime, per-call thread budgets and the prepared-model cache. It is published to crates.io and used by pamin-memory and other modules.

## Source of truth

- Specification, decisions and plans live in the private InternalDocs repository under `pamin-runtime/` (start with its `README.md`), plus the shared extension architecture in `pamin-platform/decisions/`. InternalDocs is a sibling clone at `$PAMIN_LABS_DEV_INTERNAL_DOCS` (default `../InternalDocs`); pull it before reading. Read the relevant documents before changing behaviour.
- When a design or decision changes, update InternalDocs in a PR there; cite it from here with SHA permalinks.
- This repository is public. Keep private planning, pricing, strategy and anything about paid or closed modules out of it; that lives in InternalDocs.

## Rules

You must follow `.agents/skills/pamin-engineering/references/core.md`: @.agents/skills/pamin-engineering/references/core.md
You must follow `.agents/skills/pamin-engineering/references/rust.md`: @.agents/skills/pamin-engineering/references/rust.md
You must follow `.agents/skills/pamin-engineering/references/performance.md`: @.agents/skills/pamin-engineering/references/performance.md

- This crate is a library, so `core.md`'s rule for published libraries applies; it pins `ort` exactly, for the reason in InternalDocs `pamin-platform/decisions/2026-11-01-library-dependency-requirements.md`.
- **No hidden defaults.** The crate deploys nothing, so it reads few settings; each environment variable it reads and each default it falls back to (a device selection, a thread budget, a cache directory) is named once, as a constant in the module that owns it, and documented there.
- External JSON here is the plan cache's files and model metadata; `core.md`'s "Model external data with types" binds them.
- Extending the crate means one more kernel, device, execution provider or prepared-model rewrite: `core.md`'s extensibility check applies to each.
- Nothing here knows about MCP, HTTP, tenants or any module; it changes only when hardware or inference back ends change.
- Every kernel path is tested on its own hardware (x86-64, Linux aarch64, Apple silicon); AVX-512 stays out until a measurement beats AVX2.

## How the shared rules apply here

- The coordination tool is pamin-labs-dev: its `protocol` skill, its `pamin-labs-dev:commit` skill for commits and PRs, and `pamin-labs-dev:peer-review` for the review before every PR, which also checks the change against the `pamin-engineering` skill.
- The roadmap and decision records are in InternalDocs; a published library's caret minimums are InternalDocs `pamin-platform/decisions/2026-11-01-library-dependency-requirements.md`.
- Prefer `rtk <command>` for shell commands when `rtk` is available.
- A module served by the Pamin platform is metered by the platform, never by its own code.

## Layout

The crate is one project in one language, so this file and the references it imports hold all its rules. **A directory says what its code is:** this list names every module and folder by its path, with what it owns, and a new one is described here in the PR that adds it, never placed beside them without a line; no test reads this list yet, so review checks it.

- `src/lib.rs`: the crate's documentation, its modules and its public re-exports.
- `src/error.rs`: the crate's one error type and `Result`.
- `src/kernels.rs`: vector math dispatched by instruction set at run time (the exact-rescoring dot product); `src/kernels/f16.rs`: IEEE 754 half-precision encoding and decoding.
- `src/onnx.rs`: small ONNX wire helpers the graph rewrites share, leaving unknown fields unchanged.
- `src/activity.rs`: what model loads in this process fetched and prepared, as counters.
- `src/device.rs`: which ONNX Runtime execution providers a machine can run, and what a session was actually given.
- `src/winml.rs`: Windows ML's certified vendor catalog, on Windows only.
- `src/budget.rs`: how much of the machine one forward pass may use, and the session options every model is built with.
- `src/session.rs`: ONNX Runtime sessions over a model's file, and the Core ML compiled cache for static shapes.
- `src/resident.rs`: handing a model's mapped weights back to the page cache while it is idle.
- `src/plan.rs`: which device a model runs on, chosen by measuring it and kept per machine.
- `src/prepared/`: a prepared, memory-mapped copy of a model: `mod.rs` builds and loads it, `key.rs` keys it to the runtime and CPU that wrote it, and `attention.rs`, `word_embeddings.rs` and `native.rs` are its graph rewrites.
- `build.rs`: carries the Windows ML catalog with an installed executable.
- `tests/`: integration tests through the public API, with ONNX fixtures in `tests/fixtures/`.

## Layers here

pamin-runtime is a compute library: it has no entry point, wiring or transport (its caller builds and serves), and no database, so the rules about those have no subject here. Its modules under "Layout" above still layer one way by their `use` paths:

- **Domain:** `prepared` (the prepared copy and its rewrites) and `plan` (choosing a device by measuring).
- **Integrations:** `session`, `budget`, `device`, `winml` and `resident`, the modules that configure ONNX Runtime, the platforms' accelerator catalogs and the page cache.
- **Values:** `error`, `kernels`, `onnx` and `activity`, which import no other module of the crate but `error`.

`prepared` and `plan` → `session` → `budget` and `resident` → `device` → `winml` → values. Two edges point back up today: `budget` reads `prepared::LEVEL` and `session` calls `prepared::source_digest`; new code adds none, and moving them down is its own change. The test over the modules' `use` paths that fails when a layer points back up is not yet written; on the roadmap.

## How this repository meets the errors, data, settings and calls rules

pamin-runtime is a compute library with no database, operator API, deployment or service of its own. The rules above apply to it for errors, model calls, logs and deadlines; the rest has no subject here.

- **Errors:** one `Error` enum in `src/error.rs`, whose variants keep apart what a caller treats differently (an incompatible provider, a numerical mismatch, a runtime refusal, a plugin conflict, I/O); turning them into a client's codes is the caller's.
- **Model calls:** every ONNX Runtime session is built in `src/session.rs` with the options of `src/budget.rs`, and the device it runs on is chosen in `src/plan.rs`, so local inference has one owner. Pricing and metering are the caller's.
- **Deadlines:** the crate calls no service on a request's path; a forward pass's deadline is its caller's.
- **Settings:** the few environment variables it reads, such as the CUDA device selection (`device::CUDA_SELECTION`), are named once, as constants of the module that reads them.
- **Logs:** `tracing` with paths, providers, counts and errors, never a model's inputs or outputs.

## Gates before every push

`cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo deny check`.

## Skills

Longer guidance lives in `.agents/skills/`, one directory per skill; `.claude/skills/` links to the same directories.

| Skill | Use it when |
| --- | --- |
| `pamin-engineering` | Writing or changing code, adding a dependency, claiming a speed-up, reviewing |
