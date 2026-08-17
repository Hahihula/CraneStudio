# CraneStudio — Implementation Plan

> **Read this first, and read it fully, before writing any code.**
> This document is written for implementing agents (Sonnet-class) who have not
> seen the Crane codebase. Section 2 contains verified facts about Crane that
> contradict common assumptions — do not skip it. Where this document says
> "verified", the claim was checked against the Crane source at the paths given.

---

## 1. Goal and user story

CraneStudio is an LM Studio-style local model studio, in Rust, backed by the
[Crane](https://github.com/lucasjinreal/Crane) pure-Rust inference stack
(Candle-based, no llama.cpp, no Python at runtime).

**The mission:** make Crane usable by people who do not know Rust and will never
run `cargo build`. They download one prebuilt binary from a GitHub release, run
it, and get a working local OpenAI-compatible server that their coding agent can
talk to.

**The v1 user story, end to end:**

1. User downloads `cranestudio` for their platform and runs it. No install, no
   Python, no toolchain.
2. CraneStudio measures the machine: GPU, VRAM, free VRAM, driver, RAM, disk.
3. It shows a catalog of models that *this machine can actually run*, annotated
   with what will fit.
4. User picks one. It is downloaded (with resume + progress) or located on the
   local filesystem.
5. A wizard walks through quantization, KV-cache compression, and context
   length, showing a live VRAM budget bar that updates as knobs move.
6. The server starts. If it OOMs, CraneStudio explains which specific knob to
   change and by how much.
7. The screen prints exact, copy-pasteable instructions for connecting
   **opencode** (and in v1.1, **Claude Code**).
8. The configuration can be named and saved, and reappears as the default next
   time.

**Later (post-v1):** a GUI reusing the same daemon, and per-model "apps" —
dedicated pages for specialised models like VoxCPM2 where the user works with
the model directly inside the studio.

---

## 2. Verified facts about Crane

These were checked against the source. Several contradict reasonable
assumptions. **Trust this section over your intuition.**

### 2.1 Workspace layout

| Crate | Path | Role |
|---|---|---|
| `crane-core` | `crane-core/` | Model impls, tokenizer, fused CUDA/HIP kernels, generation |
| `crane` | `crane/` | High-level SDK (chat/vision/audio/llm) |
| `crane-serve` | `crane-serve/` | axum OpenAI + SGLang server, continuous-batching engine |
| `crane-examples` | `example/` | Demo binaries |

Server entry point is `crane-serve/src/main.rs`, which only calls
`crane_serve::cli_main`. The library target is `crane_serve`.

### 2.2 crane-serve serves exactly ONE model per process

`crane_serve::Args` (`crane-serve/src/lib.rs:31-88`) takes a single
`--model-path`. `AppState` holds one `Option<EngineHandle>` plus optional
per-modality channels. **There is no load endpoint, no unload endpoint, and no
hot swap.** Switching models means starting a different process.

This is why CraneStudio is a supervisor. It is not a nice-to-have.

### 2.3 Tool calling — MERGED to upstream main (PR #102)

**Status: landed 2026-08-17.** PR #102 merged as `2078b18`; upstream `main` is
`4242e9c`. `crane-serve/src/tools.rs` and `crane-serve/tests/tool_calling.rs`
are on `main`. **There is no longer any upstream blocker for CraneStudio v1.**

Verified on `upstream/main` at `crane-serve/src/openai_api.rs`:

`ChatCompletionRequest` now carries:

```rust
/// Function/tool specs, passed to the chat template verbatim — the
/// template owns the prompt syntax (`tool | tojson` for the Qwen family).
pub tools: Option<Vec<Tool>>,
pub tool_choice: Option<serde_json::Value>,
```

`Tool` is `{ kind: String (default "function"), function: FunctionDef }`, passed
through to the template untouched — so an unusual `parameters` schema needs no
special support.

`ChatMessage` gained `tool_calls: Option<Vec<ToolCall>>`, `tool_call_id:
Option<String>`, and a `name` field for the tool name on `role: "tool"` messages
(a fallback for clients predating `tool_call_id`). There is a
`ChatMessage::tool_calls(...)` constructor, and the streaming delta type carries
`tool_calls` too — so streaming tool calls work.

**Known limitation — `tool_choice` is advisory.** From the field's own doc
comment:

> Accepted for OpenAI compatibility. `"none"` suppresses the tool block;
> anything else is advisory, since forcing a specific call would require
> constrained decoding the engine does not implement.

So `"none"` is honoured, but `"required"` and
`{"type":"function","function":{"name":"..."}}` are **accepted and ignored** —
the model is free to answer in prose instead of calling the named tool.

**This matters for CraneStudio.** Agent clients do sometimes force a tool
choice, and a silently-ignored constraint produces a confusing failure (the
agent waits for a tool call that never comes). Two consequences:

1. During M0's end-to-end opencode test, specifically exercise a forced
   `tool_choice` path if the client uses one, and note the behaviour.
2. If it proves to be a real problem in practice, constrained decoding is
   **upstream work** (§11.2 territory), not something to paper over in the
   gateway.

The older standalone demo, `example/src/ornith_tools.rs`, still exists and
remains a useful reference for how `apply_chat_template_with_tools` behaves.

### 2.4 There is no Anthropic surface

Route table (`crane-serve/src/lib.rs:1229-1263`):

```
GET  /health
GET  /v1/stats
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/audio/speech
GET  /v1/audio/duplex          (websocket)
POST /v1/tokenize   POST /tokenize
POST /v1/detokenize POST /detokenize
POST /generate                 (SGLang)
GET  /model_info /server_info /health_generate   (SGLang)
POST /abort_request            (SGLang)
```

No `/v1/messages`. Claude Code speaks the Anthropic Messages API with its own
SSE event grammar. The translator is CraneStudio's job, scheduled for v1.1.

### 2.5 GPU support is compile-time, not runtime

Cargo features flow `crane-core` → `crane` → `crane-serve`:
`cuda`, `cudnn`, `mkl`, `metal`, `accelerate`, `onnx`, `rocm`.

- **CUDA:** `crane-core/build.rs` compiles `crane-core/kernels/cuda/*.cu` via
  `bindgen_cuda` at build time. Needs `nvcc`. Crane's CI pins
  `CUDA_COMPUTE_CAP=80` because the runner has no GPU; PTX JITs forward to newer
  cards.
- **Metal:** macOS. Auto-enabled for `aarch64-apple-darwin` in `crane-core`'s
  target-specific deps.
- **ROCm:** `crane-core`'s `rocm` feature is literally `rocm = []` with the
  comment *"candle 0.11 on crates.io has no `rocm` feature, so this can't
  forward to `candle-core/rocm` yet"* — yet `crane-serve/src/engine/memory.rs`
  calls `candle_core::rocm_backend::rocm_rs::hip::memory_info()`. A ROCm build
  therefore requires a patched candle. Additionally, per the 2026.08.01 README
  entry, the ROCm build **compiles kernels with `hipcc` on the user's machine at
  first use** and caches the code object. **ROCm is not ready. Do not ship it in
  v1.** Design so it can be added later (§13).

**Consequence:** "one prebuilt binary" is really a release matrix. See §12.

### 2.6 Device selection is hardcoded to GPU 0

`crane-serve/src/lib.rs:481` calls `Device::cuda_if_available(0)`. There is no
`--device` / `--gpu-id` flag, and no tensor parallelism.

**Multi-GPU is an upstream Crane project, explicitly out of scope for this
repository.** It will be added to Crane separately once multi-GPU hardware is
available for development. CraneStudio must not attempt to work around its
absence.

What CraneStudio *does* do, because it spawns the child: set
`CUDA_VISIBLE_DEVICES` so a multi-GPU user can at least choose *which single*
GPU a model runs on, and run different models on different GPUs concurrently.
That is a legitimate use of an existing mechanism, not a workaround. Keep the
`device: usize` field in the profile schema (§5) so the UI is already in place
when real multi-GPU lands.

### 2.7 Key knobs are environment variables with no CLI flag

Every one of these must be set on the **child process environment**, not passed
as an argument. This is a second, independent reason the architecture must spawn
processes rather than link a library.

| Env var | Effect | Scope |
|---|---|---|
| `CRANE_KV_QUANT` | `int8` → ~2× smaller KV, `int4` → ~4× smaller | **Qwen 3.5 family ONLY** (`crane-core/src/models/qwen3_5/kv_cache.rs:67`) |
| `CRANE_ISQ` | In-situ quantization level (`q4k`, `q8_0`, …) | Qwen 3.5 family; `--quant` overrides |
| `CRANE_PREFILL_CHUNK` | Prefill chunk size — lowers peak VRAM during prefill | General |
| `CRANE_PREFILL_TOKENS` | Prefill token budget | General |
| `CRANE_PROF` | Per-forward-pass profiler | Debugging |
| `CUDA_VISIBLE_DEVICES` | GPU selection (standard NVIDIA var) | CUDA builds |

`CRANE_ISQ` **panics** on an invalid value
(`crane-core/src/models/qwen3_5/model.rs:601-609`) rather than falling back —
CraneStudio must validate before spawning.

Other `CRANE_*` vars exist for TTS/ASR/OCR/MiniCPM paths; out of v1 scope.

### 2.8 KV-cache compression is Qwen 3.5-family only

`CRANE_KV_QUANT` is read exclusively in
`crane-core/src/models/qwen3_5/kv_cache.rs`. **Qwen 3, Qwen 2.5, Gemma 4, and
Hunyuan Dense have no KV quantization at all.** The wizard's available knobs are
per-architecture. Do not build a UI that implies otherwise.

### 2.9 In-situ quantization is a fallback, not the selling point

From Crane's own `AGENTS.md` §"Known gaps":

> **In-situ quantization is not competitive with GGUF at 27B scale.** Measured
> for Qwen 3.8-27B: `--quant q4k` lands at ~15.8 GiB versus 15.40 GiB for the
> Q4_K_M GGUF, i.e. slightly *larger* and cruder.

Two causes, both documented upstream: ISQ leaves `embed_tokens` dense (2.37 GiB
bf16 vs 0.67 GiB as Q4_K), and it applies **one dtype to every tensor**, whereas
Q4_K_M deliberately spends 4.14 GiB putting Q6_K on the 65
quantization-sensitive tensors (`ffn_down` ×64 + `output`).

Also: `--quant` is **rejected for `Qwen3_5VL`** (`model_factory.rs:445` requires
`ModelType::Qwen3_5` exactly). Since a 27B safetensors directory has a
`vision_config` and auto-detects as VL, ISQ requires `--text-only` or an
explicit `--model-type qwen3_5`.

**Design rule for the wizard: prefer a published GGUF whenever one exists.
Offer ISQ only when no GGUF is available, and label it as such.**

### 2.10 There is no VRAM predictor to reuse

`crane-serve/src/engine/memory.rs` only:
- parses a limit string (`"8G"`, `"8GiB"`, `"0.9"` fraction, raw bytes) —
  `MemoryConfig::parse_memory_limit`
- queries *live* usage — `query_gpu_memory_usage(device) -> (used, total)`

`query_gpu_memory_usage` is implemented for **CUDA** (via
`cudarc::driver::result::mem_get_info`) and **ROCm** (via `hip::memory_info`).
It returns `(0, 0)` for **Metal and CPU**.

**Consequence:** the "will this fit?" prediction is entirely new code in
CraneStudio (§7), and Apple Silicon needs its own VRAM-detection path (§6).

### 2.11 Peak VRAM occurs at prefill, not at load

A model that loads fine can OOM later on the first long prompt. Any estimate
that counts only weights + KV will under-predict. `CRANE_PREFILL_CHUNK` is the
knob that lowers this peak. See also §2.11b — KV is not pre-allocated, so peak
memory is reached gradually as context fills.

### 2.11b KV cache grows dynamically, and over-budget means thrashing, not OOM

**This is decisive for the 256k context goal (§7.0). Read it carefully.**

Verified in `crane-core/src/models/qwen3_5/kv_cache.rs` and
`crane-serve/src/engine/mod.rs`:

- **KV is not pre-allocated to `--max-seq-len`.** `grow_append` allocates on
  demand and grows with a fixed headroom of `ROOM = 256` positions
  (`kv_cache.rs:28`). So `--max-seq-len 262144` does **not** reserve 16 GiB at
  startup; memory grows with the tokens actually in flight.
- **Over-budget triggers preemption, not a crash.** `evict_if_needed`
  (`engine/mod.rs:430`) drops a running sequence's KV cache and moves it back to
  the waiting queue, to be **re-prefilled later**. There is a 5-step
  `eviction_cooldown` to avoid thrash-looping, and `effective_max_running` is
  clamped post-eviction.
- The `total_kv_swaps` counter in `/v1/stats` (§2.14) **exposes this
  directly.** A rising `kv_swaps` means the configuration is over budget and the
  server is paying re-prefill costs. CraneStudio can detect and report a bad
  configuration from telemetry the server already emits, without waiting for a
  crash.

**The concurrency trap.** `ModelBackend::supports_kv_swap()` defaults to `false`
(`engine/backend.rs:67`). Only **`HunyuanBackend`** (line 286) and
**`Qwen3Backend`** (line 663) override it to `true`. `Qwen3_5Backend`
(`backend.rs:500-599`) does **not**.

`engine/mod.rs:130-137` then does:

```rust
let effective_max = if model.supports_kv_swap() { max_concurrent } else { 1.min(max_concurrent) };
```

So for **Qwen 3.5 / 3.6 / 3.8 — the flagship family — `--max-concurrent` is
silently forced to 1**, with only an `info!` log to say so. Gemma 4, Qwen 2.5,
and MiniCPM5 are likewise capped at 1.

**Consequences for CraneStudio:**
1. The wizard must **not** offer a concurrency knob for families capped at 1;
   show "single sequence (this model does not support KV swap)" instead.
   Catalog entries carry a `supports.kv_swap` flag.
2. The estimator's concurrency multiplier is **1** for those families — which is
   exactly what makes a 256k target reachable, since the whole KV budget goes to
   one sequence.
3. `--gpu-memory-limit` is the real safety mechanism, not `--max-seq-len`. Set it
   from the measured usable VRAM (§6) so the engine evicts instead of OOMing.
4. Watch `total_kv_swaps` and surface "this configuration is thrashing — reduce
   context or deepen KV quantization" in the dashboard.

### 2.12 Supported model families (v1 relevant subset)

From `crane-serve/src/engine/model_factory.rs` and `AGENTS.md`:

| Family | `--model-type` aliases | Formats | Notes |
|---|---|---|---|
| Qwen 3.5 / 3.6 / 3.8 | `qwen3_5`, `qwen35`, `qwen3_6`, `qwen3_8`, … | safetensors + GGUF | Hybrid GDN + softmax attention. Supports ISQ and KV quant. 3.6/3.8 declare `model_type: "qwen3_5"` — same architecture scaled up |
| Qwen 3.5/3.6/3.8 VL | `qwen3_5_vl`, `qwen35_vl`, … | safetensors only | No GGUF/mmproj vision loader. Rejects `--quant` |
| Qwen 3 | `qwen3` | safetensors + GGUF | Dense + MoE |
| Qwen 2.5 | `qwen25` | safetensors | |
| Hunyuan Dense | `hunyuan` | safetensors + GGUF | |
| Gemma 4 | `gemma4` | safetensors | Text + vision, no audio. **Gated repo on HF** |

**Not supported today: Llama, Mistral, DeepSeek, GLM, Phi.** A generic "browse
HuggingFace" UI would currently offer many models that fail to load. This is why
§8 specifies a curated catalog with filtered search.

**This list is actively growing** — expanding Crane's model coverage is ongoing
upstream work. Therefore:

- **Never hardcode the supported-architecture list in CraneStudio.** Derive it
  from Crane's own alias table in `model_factory.rs`. Ideally, get a
  `pub fn supported_model_types() -> &'static [(&str, &str)]` added upstream and
  call it, so CraneStudio's coverage grows automatically with a dependency bump.
  Until that exists, mirror the table in one clearly-marked constant with a test
  that fails when it drifts from Crane's.
- Adding a model to the catalog must be a **single JSON/RON entry with no code
  change**. If it requires code, the catalog schema is wrong.
- A dependency bump that adds a Crane model family should require only a catalog
  update to expose it in CraneStudio.

`--model-type auto` auto-detects from `config.json` `model_type` /
`architectures`, then falls back to path-name heuristics
(`model_factory.rs:200-380`). Prefer passing an explicit `--model-type` from the
catalog; keep `auto` for user-supplied local paths.

### 2.13 crane-serve CLI surface (the full contract CraneStudio drives)

```
-m, --model-path <PATH>          required; directory or .gguf file
    --model-type <TYPE>          default "auto"
    --model-name <NAME>          name reported by /v1/models
    --host <HOST>                default "0.0.0.0"   ← see §10, security
-p, --port <PORT>                default 8080
    --cpu                        force CPU
-c, --max-concurrent <N>         default 16
    --decode-tokens-per-seq <N>  default 16
    --format <auto|safetensors|gguf>
    --quant <q4k|q8_0|...>       ISQ; qwen3_5 only; overrides CRANE_ISQ
    --dtype <f16|bf16|f32>       defaults: BF16 CUDA, F16 ROCm, F32 CPU,
                                 F32 Metal except qwen3_5 → F16
    --max-seq-len <N>            0 = unlimited
    --gpu-memory-limit <S>       "0.9" fraction | "8G"/"8GiB"/"5120M" | bytes.
                                 LLM engine mode only
    --llm-gguf <PATH>            MiniCPM-o duplex only (out of v1 scope)
    --text-only                  load a VL checkpoint as text-only; also
                                 unlocks --quant on VL checkpoints
```

### 2.14 Useful runtime signals

`GET /v1/stats` returns `StatsSnapshot`
(`crane-serve/src/engine/stats.rs:81-93`):

```
total_requests, completed_requests, cancelled_requests, failed_requests,
total_prompt_tokens, total_completion_tokens, active_sequences,
waiting_sequences, total_kv_swaps, avg_decode_tokens_per_sec,
avg_prefill_tokens_per_sec
```

Poll this for the TUI dashboard. `GET /health` returns `{"status":"ok"}` — use
it as the child readiness probe.

### 2.15 The model downloader is a Python script

`data/crane-model-download` is a `#!/usr/bin/env -S uv run --script` file
depending on `huggingface_hub`. **A precompiled binary cannot rely on this.**

**Hard rule: there is no Python anywhere in this repository.** Not in the
runtime, not in build scripts, not in CI helpers, not in tooling. The
downloader is reimplemented natively in Rust (§9). If you find yourself reaching
for a Python script, write it in Rust or as a `cargo xtask`.

Note `hf-hub` 0.5 is already a `crane-core` dependency, but it lacks the
resume/progress/concurrency behaviour this UX needs — evaluate it, and expect to
write the downloader directly against `reqwest`.

### 2.16 VoxCPM2 and MiniCPM ARE on upstream main — Crane's AGENTS.md is stale

Crane's `AGENTS.md` has a section titled "Pre-merge work on `minicpm-support`
(not on `main`)". **That section is out of date.** Verified with
`git ls-tree upstream/main`: `crane-core/src/models/` on `lucasjinreal/Crane`
main contains `minicpm5`, `minicpm_v`, `minicpmo`, and `voxcpm2`, and
`model_factory.rs` has `ModelType` variants and aliases for all of them
(`"voxcpm2" | "voxcpm-2" | "voxcpm_2" | "voxcpm"`, `"minicpm5"`, `"minicpmv46"`,
`"minicpmo"`).

Good news for the post-v1 "specialised model apps" phase — VoxCPM2 needs no
branch merge. Note `/v1/audio/duplex` is still guarded by an exclusivity mutex:
one live session at a time.

**General lesson: `AGENTS.md` is a useful map but it is not authoritative. Check
the tree.**

### 2.17 Crane repo conventions worth adopting

- Files are kept under a **400-line cap**; oversized modules get split.
- **`cargo fmt` is now gating in Crane's CI** as of PR #103 (`9121de7 ci:
  Enforce cargo fmt`, 2026-08-17), and the tree was reformatted in `2d6ea3c`.
  Match this — any upstream PR from this project must be `cargo fmt` clean.
- `cargo clippy -- -W clippy::pedantic -D warnings` is still `continue-on-error`
  in Crane's CI; **make it gating in CraneStudio's**.
- `edition = "2024"`, let-chains in use.

---

## 3. Architecture

### 3.1 Decision: one binary, self-spawning, daemon + gateway

```
                    ┌─────────────────────────────────────────┐
   user's terminal  │  cranestudio  (TUI client)              │
                    └───────────────┬─────────────────────────┘
                                    │ unix socket / loopback HTTP
                                    │ (control API, JSON)
                    ┌───────────────▼─────────────────────────┐
                    │  cranestudio --daemon                   │
                    │                                         │
                    │  ┌───────────┐  ┌────────────────────┐  │
   agent clients ──▶│  │ gateway   │  │ supervisor         │  │
   (opencode,       │  │ :1234     │  │ spawn/kill/health  │  │
    Claude Code)    │  │ /v1/*     │  │ log capture        │  │
                    │  └─────┬─────┘  └────────┬───────────┘  │
                    │        │  routes by      │              │
                    │        │  "model" field  │              │
                    └────────┼─────────────────┼──────────────┘
                             │                 │ spawns
                    ┌────────▼─────────────────▼──────────────┐
                    │ cranestudio __serve -m ... -p 41xxx     │  ← re-exec of
                    │   (== crane_serve::run)                 │    the SAME binary
                    ├─────────────────────────────────────────┤
                    │ cranestudio __serve -m ... -p 41xxx     │
                    └─────────────────────────────────────────┘
```

**The single-binary trick:** `cranestudio` links `crane-serve` as a library. When
invoked as `cranestudio __serve <crane-serve args...>`, it parses
`crane_serve::Args` and calls `crane_serve::run` — i.e. it *is*
crane-serve. The daemon spawns children by re-executing
`std::env::current_exe()` with the `__serve` subcommand.

This gets every benefit at once:
- **One file to download.** No "also grab the crane binary" step.
- **Per-child environment.** `CRANE_KV_QUANT`, `CRANE_ISQ`,
  `CUDA_VISIBLE_DEVICES` (§2.7) can be set per model.
- **Crash isolation.** A CUDA OOM kills one child; the daemon survives, captures
  the child's stderr, and shows advice (§7.4).
- **Model switching** despite §2.2.
- **Build features propagate naturally.** A `--features cuda` build of
  CraneStudio produces a CUDA-capable `__serve` worker.

The `__serve` subcommand is hidden from `--help` (clap `hide = true`). It is an
internal implementation detail, not a supported CLI.

### 3.1a Daemon lifetime — no orphans, ever

The daemon can outlive the TUI, but **only by explicit user consent**. Nothing
is ever left running behind the user's back — an orphaned process holding 18 GiB
of VRAM is the worst failure mode this product can have.

**On TUI exit with models running**, prompt:

```
  ┌─ Quit CraneStudio ────────────────────────────────┐
  │  1 model is still running (Qwen 3.5 9B, 18.4 GiB) │
  │                                                    │
  │  [K] Keep serving in background                    │
  │  [S] Stop everything and quit          (default)   │
  │  [C] Cancel                                        │
  └────────────────────────────────────────────────────┘
```

**When user interaction is impossible, always kill.** That covers: `SIGHUP`
(terminal window closed), `SIGTERM`, `SIGINT` twice, stdin not a TTY, and the
TUI process dying abnormally. Default-to-stop is the rule; keeping alive requires
a live human saying so.

**Mechanism — the detach lease.** Do not rely on catching signals alone; a
`SIGKILL`'d TUI catches nothing.

1. The daemon tracks connected control clients.
2. It holds a boolean `detached` flag, initially `false`.
3. Choosing "keep serving" sends an explicit `detach` command, setting it `true`.
4. **When the last control client disconnects and `detached == false`, the
   daemon shuts down all children and exits.** Give a short grace period (~5 s)
   so a TUI restart or a transient reconnect does not kill a session.
5. `detached` resets to `false` when a new interactive client attaches, so the
   next quit asks again.

This makes correct behaviour the consequence of the client vanishing, not of
successfully running cleanup code. Test it by `kill -9`-ing the TUI and
asserting the children are gone within the grace period.

**Additional safety nets:**
- Children are spawned in the daemon's process group so a group signal reaches
  them.
- On Linux, set `PR_SET_PDEATHSIG` on children so they die with the daemon even
  if it is `SIGKILL`'d.
- On startup, the daemon scans for stale `__serve` processes from a previous
  crashed run (recorded in a pidfile) and reaps them, reporting what it cleaned.

`cranestudio stop` shuts down a detached daemon. `cranestudio status` reports
what is running and whether it is detached — a user must always be able to find
out what is holding their GPU.

The daemon remains a separate process (rather than living inside the TUI)
because that is what makes the future GUI cheap: the GUI is just another client
of the same control API, with the same lease semantics.

### 3.2 The gateway is the key UX idea

The gateway is a **single stable port** (default `:1234`, loopback only) that:

- aggregates `/v1/models` across every configured model, whether or not its
  child is currently running;
- routes each request to the correct child by the request's `model` field;
- **starts a child on demand** when a request names a model that is not running;
- evicts by LRU when VRAM is insufficient for a new child;
- streams SSE straight through;
- in v1.1, hosts `/v1/messages` (Anthropic translation).

**Why this matters:** the connect instructions become a constant string that
never changes when the user switches models. That is most of the UX win over
raw crane-serve, where each model is a different port.

### 3.3 Crate layout

```
CraneStudio/
├── Cargo.toml                    # workspace
├── rust-toolchain.toml           # pin stable; edition 2024
├── PLAN.md                       # this file
├── catalog/models.ron           # curated catalog, versioned in-repo
├── crates/
│   ├── studio-core/              # lib — no I/O side effects beyond fs/http
│   │   ├── hardware/             # GPU/CPU/RAM/disk probing (§6)
│   │   ├── catalog/              # catalog schema, load, HF search filter (§8)
│   │   ├── estimator/            # VRAM math + measurement DB (§7)
│   │   ├── download/             # HF downloader (§9)
│   │   ├── profile/              # named configs (§5)
│   │   └── launch/               # LaunchSpec → argv + envp for a child
│   ├── studio-supervisor/        # lib — child lifecycle, health, log ring,
│   │                             #       exit classification (§7.4)
│   ├── studio-gateway/           # lib — axum: control API + /v1/* multiplex
│   ├── studio-tui/               # lib — ratatui screens (§4)
│   └── cranestudio/              # bin — dispatch: tui | daemon | __serve | doctor
└── .github/workflows/release.yml
```

Dependency direction is strictly downward: `cranestudio` → {`studio-tui`,
`studio-gateway`, `studio-supervisor`, `studio-core`, `crane-serve`};
`studio-gateway` → {`studio-supervisor`, `studio-core`}; `studio-supervisor` →
`studio-core`. **`studio-core` depends on nothing else in the workspace.**

`studio-core`, `studio-supervisor`, and `studio-gateway` must contain **zero
terminal-rendering code**. All of it lives in `studio-tui`. This is what makes
the future GUI a drop-in replacement rather than a rewrite.

### 3.4 Depending on Crane

Pin to an **exact commit on upstream** `lucasjinreal/Crane`. Verified good as of
2026-08-17:

```toml
[workspace.dependencies]
# lucasjinreal/Crane main @ 2026-08-17. Includes:
#   PR  #99 — Qwen 3.5 GGUF long-prompt quant fix
#   PR #100 — Qwen 3.6 / 3.8 (27B) support
#   PR #102 — OpenAI tool calling in crane-serve
crane-serve = { git = "https://github.com/lucasjinreal/Crane", rev = "4242e9c3d85c030341aac562b07c650a0fe3e5f6" }
crane-core  = { git = "https://github.com/lucasjinreal/Crane", rev = "4242e9c3d85c030341aac562b07c650a0fe3e5f6" }
```

**No `[patch]` section is needed for capability reasons — every capability v1
requires is on upstream `main`.** However, as of 2026-08-17 the pinned rev
**does not compile**: `crane-core/src/models/muscriptor/conditioner.rs` mixes
`&mut Tensor` and `&Tensor` in one array literal passed to `Tensor::cat`
(`*mel = Tensor::cat(&[mel, &pad_mel], 0)?;` and the equivalent for `mask`),
which candle-core 0.11's array-literal coercion rejects. Verified: this is on
`origin/main` itself (our pin *is* `origin/main`), landed in `245b6bb1 "added
muscriptor support"` (2026-08-15, author hahihula), and blocks the whole
crate — muscriptor is an unconditional module, not feature-gated. Confirmed
unrelated to the model families CraneStudio drives.

Fix is a one-line reborrow (`&*mel`, `&*mask`) in both spots; verified with
`cargo check -p crane-core --features cuda` and `cargo fmt --check`. Prepared
as a patch at `/home/hahihula/muscriptor-cat-fix.patch` and applied on a local
branch (`fix/muscriptor-cat-mut-borrow`, commit `083f2c1`) at
`/home/hahihula/mywork/crane-local-patched`, pending push to
`lucasjinreal/Crane` by hahihula (who has direct write access, per the
`245b6bb1` commit). Until that lands, CraneStudio's workspace `Cargo.toml`
carries a temporary `[patch]` pointing at that local clone — remove it and
re-pin `rev` to the commit that lands upstream once pushed; per §11.0, treat
that removal as an M10 release gate.

Bump this pin deliberately, never automatically, and re-run the estimator's
calibration tests (§7.3) after every bump — a change in Crane's memory behaviour
invalidates measurements silently otherwise.

**Sequencing — resolved.** All three prerequisites landed upstream within two
days, verified by `git fetch upstream`:

| PR | What | Landed |
|---|---|---|
| #99 | Qwen 3.5 GGUF long-prompt quant fix | 2026-08-16 |
| #100 | Qwen 3.6 / 3.8 (27B) support | 2026-08-16 |
| #102 | OpenAI tool calling in crane-serve (§2.3) | 2026-08-17, as `2078b18` |

**If you ever need to base on an unmerged PR again, one cargo gotcha:** do not
write `rev = "<pr-head-sha>"` against the upstream URL. A PR head lives on
`refs/pull/N/head`, and cargo's fetch refspec covers branches and tags, not
GitHub PR refs — it will not resolve. Point a `[patch]` at the PR's **source
repository and branch**, where the commit is a real branch head. And treat
removing any such `[patch]` as a release gate (§14, M10).

### 3.5 Rename before writing code

`Cargo.toml` currently says `name = "CraneStudio"`, which produces a binary
literally named `CraneStudio`. Fix this first:

- workspace root: virtual manifest, no `[package]`
- binary crate: `name = "cranestudio"`, `[[bin]] name = "cranestudio"`

---

## 4. TUI design

Built with `ratatui` + `crossterm`. Screens:

### 4.1 Home / dashboard
Hardware summary line, list of running models with live `t/s` and VRAM from
`/v1/stats`, saved profiles, and a persistent footer showing the gateway URL and
connection status.

### 4.2 Hardware report (`doctor`)
Everything from §6, plus explicit warnings: no GPU detected, driver too old,
insufficient free VRAM because something else is using the card, low disk.
Reachable as a subcommand too: `cranestudio doctor` prints and exits — this is
the first thing to ask a user for in a bug report.

### 4.3 Model browser
Two tabs: **Catalog** (default) and **Search HuggingFace**. Each row shows name,
parameter count, format, download size, and a fit verdict:

```
  ● Fits comfortably     estimated peak ≤ 85% of usable VRAM
  ◐ Tight                85–100%; will work but leaves no headroom
  ○ Needs smaller quant  a different variant of this model does fit
  ✕ Won't fit            no variant fits, even at minimum context
```

Never hide non-fitting models — show them greyed with the reason. Users need to
understand *why* something is out of reach.

### 4.4 The launch wizard

The core screen. **It leads with an answer, not with knobs.** The solver (§7.2)
runs first and proposes a configuration that reaches 256k; the user confirms or
opens the knobs to override.

```
 ┌─ Qwen 3.5 9B ───────────────────────────────────── RTX 3090, 24 GiB ─┐
 │                                                                       │
 │  ✓ Recommended — reaches the full 256k context                        │
 │                                                                       │
 │      Weights     Q4_K_M (GGUF, 5.4 GiB)                               │
 │      KV cache    int4  (4× smaller — required to reach 256k)          │
 │      Context     262144                                               │
 │      Sequences   1  (this model does not support KV swap)             │
 │                                                                       │
 │  ██████████████████████████░░░░░░░░░░░░░░  13.6 / 20.4 GiB at 256k   │
 │  weights 5.4 │ kv 4.0 │ prefill 2.8 │ ctx 1.4                        │
 │                                                                       │
 │  ⓘ Predicted (no measurement for this combination yet).               │
 │                                                                       │
 │  [Enter] Start   [A] Alternatives (2)   [E] Edit manually             │
 └───────────────────────────────────────────────────────────────────────┘
```

`[A] Alternatives` lists the other ranked solutions — e.g. "Q6_K + int4, 256k,
tighter" or "Q4_K_M + int8, 128k, better KV fidelity" — each with its trade-off
in one line.

When the solver returns `Short`, lead with the shortfall instead:

```
 │  ⚠ Best achievable context: 98304 (target 262144)                     │
 │    Limited by: weights 15.4 GiB of 20.4 GiB usable                    │
 │    → A Q4_K_M variant of this model would reach 256k.  [switch]       │
```

Rules:
- **Only show knobs the selected architecture supports.** KV quant is hidden
  entirely for Qwen 3 / Qwen 2.5 / Gemma 4 / Hunyuan (§2.8). Concurrency is
  hidden and shown as fixed for families capped at 1 (§2.11b).
- **Only offer ISQ when no GGUF exists** for the model, and label it
  "in-situ quantization — larger and cruder than a published GGUF" (§2.9).
- Context is capped at the model's native maximum (`max_position_embeddings`
  from `config.json`, or the GGUF header) — note many models are below 256k
  natively, in which case their native max *is* the target and that is fine.
- Manual editing may not go below the 32k floor (§10.3).
- Always show whether the number is **predicted** or **measured** (§7.3). Users
  trust a measured number and deserve to know when they aren't getting one.

### 4.5 Server running / connect
Green banner, gateway URL, live token rate, and copy-paste blocks per client
(§10). Log tail pane, toggleable.

### 4.6 Chat playground
Minimal streaming chat against the gateway. This is the "it works!" moment —
it lets the user feel the tokens/sec before wiring up an external client.

Keep it deliberately small: user/assistant turns, streaming text, a stop key, a
token-rate readout, and a "clear" action. Render `reasoning_content` (§2.14,
Crane separates the `<think>` scratchpad out of `content`) in a dimmed collapsed
block. **Do not** build a markdown renderer, syntax highlighting, or
conversation persistence in v1.

---

## 5. Configuration and profiles

Location via the `directories` crate:
- Linux: `~/.config/cranestudio/`, `~/.local/share/cranestudio/`
- macOS: `~/Library/Application Support/cranestudio/`

**Serialization format is RON** (`ron` crate) for everything CraneStudio writes
and reads — config, profiles, measurements, and the catalog. Not TOML. RON maps
cleanly onto Rust enums and `Option`, which this domain is full of
(`KvQuant::Int8 | Int4 | None`, `Format::Gguf | Safetensors`), and it avoids
TOML's awkwardness with nullable nested structures. One format, no exceptions,
so implementers never have to ask.

```
config.ron                  # global settings
profiles/<name>.ron         # named launch configurations
measurements.ron            # the measurement DB (§7.3)
catalog-cache.ron           # last fetched remote catalog
logs/<model>-<ts>.log       # captured child stderr, rotated
```

**Profile schema:**

```ron
(
    name: "qwen35-9b-coding",
    created: "2026-08-16T13:00:00Z",
    last_used: "2026-08-16T14:22:00Z",

    model: (
        catalog_id: Some("qwen3.5-9b-q4km"),  // None for a user-supplied path
        path: "/home/u/models/Qwen3.5-9B-Q4_K_M.gguf",
        revision: Some("a1b2c3d..."),          // HF commit sha, never a tag
        model_type: "qwen3_5",
        format: Gguf,
    ),

    runtime: (
        quant: None,                    // ISQ level; None when using a GGUF
        dtype: Bf16,
        kv_quant: Some(Int4),           // None when unsupported by the family
        max_seq_len: 262144,            // §7.0 — 256k is the default target
        max_concurrent: 1,              // forced to 1 for qwen3_5 (§2.11b)
        gpu_memory_limit: "0.92",
        device: 0,                      // → CUDA_VISIBLE_DEVICES
        text_only: false,
        prefill_chunk: None,            // → CRANE_PREFILL_CHUNK
    ),

    measured: Some((                    // written after a successful run
        peak_vram_bytes: 18454302720,
        decode_tokens_sec: 62.4,
        max_verified_depth: 131072,     // deepest context actually reached
        kv_swaps_observed: 0,           // >0 means it thrashed (§2.11b)
        verified_at: "2026-08-16T14:25:00Z",
    )),
)
```

Pin `revision` to a **commit sha**, never a floating tag — otherwise a profile
silently changes meaning when the upstream repo is updated.

`config.ron` holds: gateway host/port, default models directory, HF token, LAN
exposure opt-in, VRAM headroom fraction, target context (default 262144), and
`telemetry: Unasked` (§7.3).

---

## 6. Hardware probing

`studio-core::hardware` returns:

```rust
pub struct HardwareReport {
    pub gpus: Vec<GpuInfo>,
    pub cpu: CpuInfo,       // model, physical/logical cores
    pub ram_total: u64,
    pub ram_available: u64,
    pub disk: Vec<DiskInfo>,  // for the models dir specifically
    pub backend: Backend,     // Cuda | Metal | Cpu — what THIS binary was built with
}

pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub vram_total: u64,
    pub vram_free: u64,
    pub compute_capability: Option<(u32, u32)>,
    pub driver_version: Option<String>,
    pub unified_memory: bool,   // true on Apple Silicon
}
```

**Per-platform implementation:**

- **CUDA:** prefer NVML (`nvml-wrapper`, already a `crane` dependency behind the
  `cuda` feature) for name/VRAM/driver/compute-cap. Fall back to parsing
  `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` exactly as
  `crane/src/utils.rs` does. Fall back again to
  `cudarc::driver::result::mem_get_info` for totals only.
- **Metal:** §2.10 — Crane returns `(0,0)` here. Implement independently. Read
  the recommended working set via `metal::Device::recommended_max_working_set_size`
  (the `metal` crate), and total RAM via `sysctl hw.memsize`. On unified memory,
  `vram_total` is the recommended working set, not physical RAM; set
  `unified_memory: true` and warn that the OS and other apps share the pool.
- **CPU / RAM / disk:** `sysinfo` crate, all platforms.

**Important:** report **free** VRAM, not just total, and base fit verdicts on
free. On a desktop the compositor and browser routinely hold 2-4 GiB (the
development machine for this project idles at 4.1 GiB used of 24 GiB). Budgeting
against total VRAM will produce confident predictions that OOM immediately.

Define:

```
usable_vram = (vram_free - safety_reserve) * headroom_fraction
safety_reserve  = 512 MiB     # driver/context growth
headroom_fraction = 0.95      # configurable in config.ron
```

---

## 7. The VRAM estimator

This is the technically hardest and most valuable component. It has three parts:
prediction, measurement, and OOM diagnosis. **Build it in that order, but treat
measurement as the thing that makes it trustworthy.**

### 7.0 The target is 256k context — this inverts the estimator's job

**Design premise: 262144 (256k) context is the default and the goal, because
that is the size at which a local model is genuinely useful for coding agents.**
Anything below 32768 is unusable and must be refused, not warned about (§10.3).

This changes what the estimator is *for*. It is not "let the user drag a context
slider and tell them if it fits". It is:

> **Given this hardware and this model, find a configuration that reaches 256k.
> If none exists, say what the user must change.**

The wizard is therefore a **solver**, not a form (§4.4).

**What makes this tractable** (all verified, §2.11b):

1. KV is **not** pre-allocated to `--max-seq-len`, so setting 262144 costs
   nothing at startup. Memory grows with real usage.
2. `max_concurrent` is **forced to 1** for Qwen 3.5/3.6/3.8, Gemma 4, Qwen 2.5,
   and MiniCPM5 — so the entire KV budget serves a single sequence. For a
   single-user coding agent this is exactly right, and it is a 16× saving versus
   the naive reading of the `--max-concurrent 16` default.
3. Going over budget causes **eviction and re-prefill, not a crash**. A slightly
   optimistic configuration degrades in speed rather than dying — and
   `total_kv_swaps` in `/v1/stats` reports it precisely.
4. `CRANE_KV_QUANT=int4` gives a **4× reduction** on the Qwen 3.5 family, which
   is the difference between 256k being impossible and being routine.

**Worked example — why int4 is effectively mandatory at 256k.** Crane's
`AGENTS.md` gives Qwen 3.8-27B at ~64 KB/token (16 full-attention layers × 4 KV
heads × 256 × 2 × 2 B):

| KV dtype | Bytes/token | KV at 256k |
|---|---|---|
| f16 (default) | 64 KB | **16.0 GiB** |
| int8 | 32 KB | 8.0 GiB |
| int4 | 16 KB | **4.0 GiB** |

On a 24 GiB card with a 15.4 GiB Q4_K_M checkpoint, f16 KV at 256k is flatly
impossible while int4 leaves room. **So the solver's default preference order for
reaching the target is: int4 KV first, then a smaller weight quant, then reduce
context — in that order**, because KV quantization costs far less quality per
gigabyte saved than dropping the weights a level.

**Consequence for the catalog (§8):** every entry needs a
`max_context_achievable` per backend, computed by the solver and confirmed by
measurement. A model that cannot reach 32k on common hardware does not belong in
the catalog.

### 7.1 Prediction

```
predicted_peak =
      weights
    + kv_cache(ctx, concurrency)
    + recurrent_state          (hybrid architectures only)
    + vision_tower             (VL models only)
    + activation_working_set(prefill_chunk)
    + runtime_overhead
```

**Weights.**
- GGUF: the file size on disk is a good proxy. Per the README's 2026.08.16
  entry, embeddings now stay quantized and dequantize only gathered rows, so
  there is no longer a large embedding expansion at load.
- safetensors: sum of `.safetensors` file sizes, scaled by the dtype ratio
  (bf16→f16 = 1.0; anything→f32 = 2.0 from a 16-bit checkpoint).
- ISQ: linears compress to roughly the target bits/weight, but **`embed_tokens`
  stays dense** (§2.9). Estimate as
  `(non_embedding_params × bits/8) + (embedding_params × dtype_bytes)`.
  Read parameter counts from `config.json`.

**KV cache — per-token bytes.** For a standard dense decoder:

```
kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_dtype_bytes
head_dim = config.head_dim, else hidden_size / num_attention_heads
kv_dtype_bytes = 2 (f16/bf16) | 1 (CRANE_KV_QUANT=int8) | 0.5 (int4)
```

**For Qwen 3.5 / 3.6 / 3.8 this formula is wrong if applied naively.** The
architecture is hybrid: Gated Delta Net layers carry a **fixed-size recurrent
state independent of context length**, and only the full-attention layers
contribute per-token KV. Crane's `AGENTS.md` gives the ground truth for the 27B:

> KV cache is ~64 KB/token on the 27B (16 full-attn layers × 4 KV heads × 256 ×
> 2 × 2 B) — 16 GiB at the native 262k context, so `CRANE_KV_QUANT` stops being
> optional well before then.

So: `n_layers` in the formula must be **the count of full-attention layers**, not
total layers. **Read `crane-core/src/models/qwen3_5/modeling.rs` and
`kv_cache.rs` to determine how the layer pattern is derived from config** —
whether via a `layer_types` list, a `full_attention_interval`, or a hardcoded
stride. Do not guess. Write a unit test that reproduces the 64 KB/token figure
for Qwen 3.8-27B from its real `config.json`; that test is your correctness
anchor.

The GDN recurrent state is `n_layers_gdn × n_v_heads × head_k_dim × head_v_dim ×
dtype_bytes`, allocated per active sequence and **independent of context length**
— so it scales with concurrency, not with `max_seq_len`.

**Concurrency.** crane-serve's engine pre-allocates KV. Assume the worst case
that `max_concurrent` sequences each reach `max_seq_len` unless you verify
otherwise by reading `crane-serve/src/engine/{mod,sequence}.rs`. If the engine
uses a shared token budget rather than per-sequence allocation, model it that
way instead — **check before assuming**, since this is a 16× difference at the
default `--max-concurrent 16`.

**Vision tower.** Qwen 3.5-VL's ViT is ~600M params (per `crane-serve/src/lib.rs`
`--text-only` doc comment). At bf16 that is ~1.2 GiB, and it is never quantized
(§2.9 — VL rejects `--quant`).

**Activation working set.** Scales with `prefill_chunk × hidden_size ×
n_layers`. Start with a conservative empirical constant and let measurement
correct it (§7.3). This term is why §2.11 matters.

**Runtime overhead.** CUDA context is roughly 300–600 MiB per process. Treat as
a per-backend constant, refined by measurement.

### 7.2 The solver

Two directions are needed. The second is the one the wizard actually uses.

**(a) Max context for a fixed configuration.** Everything except
`kv_cache(ctx)` is constant with respect to `ctx`, so solve directly:

```
max_ctx = (usable_vram - fixed_terms) / kv_bytes_per_token / effective_concurrency
```

Round down to a friendly value (8192 / 16384 / 32768 / 65536 / 131072 / 262144)
and clamp to the model's native maximum.

**(b) Reach the target — the primary entry point.**

```rust
/// Search the configuration space for setups reaching `target_ctx`
/// (default 262144), best first.
pub fn solve(
    model: &ModelVariant,
    hw: &HardwareReport,
    target_ctx: usize,
) -> SolveResult;

pub enum SolveResult {
    /// One or more configurations reach the target. Ranked best-first:
    /// prefer higher weight quality, then shallower KV quantization.
    Reaches(Vec<Config>),
    /// Target unreachable; `best` is the deepest achievable context.
    /// `blockers` explains what is consuming the budget.
    Short { best: Config, achieved_ctx: usize, blockers: Vec<Blocker> },
    /// Cannot even reach the 32k floor — this model does not belong on
    /// this machine. Carries concrete alternatives.
    Unusable { achieved_ctx: usize, suggestions: Vec<Suggestion> },
}
```

The search space is small enough to enumerate exhaustively — no cleverness
needed:

```
weight variants  × kv_quant ∈ {none, int8, int4}   (int8/int4 only where §2.8 allows)
                 × concurrency ∈ {effective_max}   (usually forced to 1, §2.11b)
```

That is typically under 20 candidates. Score each on `(reaches_target,
weight_quality, kv_quality, headroom)` and rank.

**Ranking rule, stated explicitly** (§7.0): when trading off to reach the
target, spend KV quantization before weight quantization. `int4` KV on a Q6_K
checkpoint beats `f16` KV on a Q3_K one at equal memory.

**`Unusable` must be actionable, not a shrug.** Suggestions in priority order:
a smaller variant of the same model family; a different model of similar
capability that does fit; or the honest statement that this machine needs
`N` GiB more VRAM for this model. Never leave the user at a dead end.

### 7.3 Measurement — the part that makes it trustworthy

**Every launch is a data point.** The supervisor:

1. records the child's peak VRAM (poll `nvidia-smi`/NVML for the child's PID, or
   sample total-used delta against the pre-spawn baseline at ~1 Hz);
2. on clean shutdown or after a successful warmup + first real request, writes to
   `measurements.ron`:

```ron
(
    schema_version: 1,
    key: "qwen3.5-9b|Q4_K_M|kv:int4|ctx:262144|conc:1|cuda|sm86",
    predicted_bytes: 13743895347,
    measured_peak_bytes: 14603811840,
    max_depth_reached: 131072,   // deepest context actually exercised
    kv_swaps: 0,                 // >0 means it thrashed (§2.11b)
    decode_tokens_sec: 62.4,
    outcome: Ok,                 // Ok | Oom | Thrashed | Failed
    at: "2026-08-16T14:25:00Z",
)
```

Note `max_depth_reached`: a run that never exceeded 8k tokens has **not**
verified that the configuration works at 256k. Record the depth actually
exercised and only claim what was tested — a "measured" label on an untested
depth is worse than an honest prediction.

3. On a subsequent launch with a matching key, **the wizard shows the measured
   number instead of the prediction**, labelled as measured.
4. Failed launches are recorded too, with `"outcome": "oom"` — those bound the
   search from above and directly feed §7.4.

Additionally, maintain a global per-backend correction factor (measured ÷
predicted, smoothed across all local data points) applied to predictions for
unmeasured combinations. This makes the estimator improve for models the user has
never run, which is the whole point.

Ship the catalog (§8) with a `measured` block containing figures gathered on
reference hardware, so a first-run user gets a measured-quality number before
they have run anything. Local measurements always take precedence over catalog
figures.

**Telemetry is local-only in v1, and there is no consent prompt yet.** There is
no server to receive anything, so asking would be asking for nothing. Do not
build an upload path, and do not prompt.

Prepare for it without shipping it:

- `config.ron` carries `telemetry: Unasked` (enum `Unasked | Enabled |
  Declined`), defaulting to `Unasked`. When the endpoint exists, first start
  finds `Unasked` and prompts once; the enum already distinguishes "never
  asked" from "asked and declined", so nobody gets prompted twice.
- The measurement record schema (above) is the wire format. Keep it free of
  anything identifying: no paths, no hostnames, no usernames, no model paths —
  only hardware class, model id, configuration, and the numbers.
- Add a `schema_version` field now so a future server can accept records written
  by older clients.

Sharing measurements is the highest-value telemetry imaginable for this
product — it is what would let a first-run user on any GPU get a *measured*
number instead of a prediction — so the schema deserves care now even though the
transport is deferred.

### 7.4 OOM diagnosis and advice

The supervisor captures the child's stderr into a ring buffer and classifies the
exit. Classification table:

| Signal | Classification | Advice |
|---|---|---|
| stderr contains `CUDA_ERROR_OUT_OF_MEMORY` / `out of memory` before first `/health` OK | **OOM at load** | Weights alone don't fit. Suggest the next smaller quant, or `--cpu`. |
| Same, after `/health` OK, during a request | **OOM at prefill** | Reduce `max_seq_len`; enable/deepen `CRANE_KV_QUANT` if supported; lower `CRANE_PREFILL_CHUNK`; lower `--max-concurrent`. |
| Exit code 137 / SIGKILL | **Host OOM killer** | System RAM, not VRAM. Model too large to even stage; suggest more swap or a smaller model. |
| `panic` containing `invalid CRANE_ISQ` | **Bad config** | CraneStudio bug — validate before spawn (§2.7). |
| Port bind failure | **Port in use** | Retry on a different port automatically. |
| Exit 0 before `/health` OK | **Clean early exit** | Usually a bad `--model-type` or a missing tokenizer; show stderr verbatim. |

**Advice must be quantitative, not vague.** Compute the specific change:

```
  ✕ Out of memory during prefill (needed ~21.4 GiB, 20.4 GiB usable).

  Any ONE of these will fit:
    → context 32768 → 20480       saves 1.9 GiB   [apply]
    → KV cache int8 → int4        saves 4.0 GiB   [apply]
    → concurrency 4 → 2           saves 4.0 GiB   [apply]

  Recorded this failure; the estimate for this model is now more accurate.
```

Each suggestion is applicable with one keypress, which re-enters the wizard with
that value pre-set.

---

## 8. Model catalog and discovery

### 8.1 Curated catalog

`catalog/models.ron`, versioned in-repo and also fetched at runtime from the
GitHub raw URL (cached to `catalog-cache.ron`, in-repo copy as the offline
fallback). Never fail to start because the catalog fetch failed.

```ron
(
    schema_version: 1,
    updated: "2026-08-16",
    models: [
        (
            id: "qwen3.5-9b",
            display_name: "Qwen 3.5 9B",
            family: "qwen3_5",
            model_type: "qwen3_5",
            params: 9000000000,
            native_context: 262144,
            capabilities: [Text, Tools],
            license: "apache-2.0",
            gated: false,
            supports: (
                isq: true,
                kv_quant: true,      // §2.8 — qwen3_5 family only
                kv_swap: false,      // §2.11b — forces max_concurrent to 1
                vision: false,
            ),
            variants: [
                (
                    id: "qwen3.5-9b-q4km",
                    repo: "Qwen/Qwen3.5-9B-GGUF",
                    revision: "<sha>",
                    files: ["Qwen3.5-9B-Q4_K_M.gguf"],
                    format: Gguf,
                    quant: "Q4_K_M",
                    download_bytes: 5798205440,
                    measured: {
                        "cuda_sm86": (
                            max_context_achievable: 262144,
                            kv_quant: Some(Int4),
                            conc: 1,
                            peak_bytes: 14603811840,
                            decode_tps: 62.4,
                        ),
                    },
                ),
            ],
        ),
    ],
)
```

Rules:
- Every catalog entry must be a combination someone has **actually launched**.
  The catalog's value is that nothing it offers can fail to load.
- `measured.max_context_achievable` is required (§7.0) — a variant that cannot
  reach the 32k floor on any common hardware does not belong in the catalog.
- Adding a model is a **data-only change**. If it needs code, the schema is
  wrong (§2.12).

### 8.2 Filtered HuggingFace search

A second tab for power users. Query the HF API, then **filter by architecture**:
fetch each candidate's `config.json` and keep only those whose `model_type` /
`architectures` map to a `ModelType` in `model_factory.rs` (§2.12). Reuse the
same alias table — extract it into a shared constant rather than duplicating it,
and add a test asserting the alias list matches Crane's.

Show unsupported results greyed out with "Crane does not support this
architecture (<model_type>)". Silently omitting them makes users think search is
broken.

For GGUF repos there is often no `config.json`. Read the GGUF header's
`general.architecture` metadata instead — a range request for the first few KB
is enough, no full download.

### 8.3 Local filesystem models

Scan the configured models directory, plus an explicit "add local path" action.
For each candidate, run the same `auto` detection Crane would (§2.12) and show
what it resolved to, so the user can correct it before launching.

---

## 9. Download manager

Native Rust, `reqwest` + `tokio`. Requirements, all of them user-visible:

- **HTTP range resume.** These are 5–30 GiB downloads on home connections.
  Interrupted downloads must resume, not restart.
- **Progress**: bytes, percent, rate, ETA, per-file and aggregate.
- **Concurrency**: 2–4 parallel connections; configurable, defaulting low enough
  to not saturate a home link.
- **Integrity**: verify against the HF-reported sha where available.
- **Disk space precheck**: refuse to start if free space < size × 1.1, with a
  clear message. Do not discover this at 94%.
- **Gated repos**: Gemma 4 requires an HF token and accepted license. Detect the
  401/403, and explain exactly what to do (visit the model page, accept, paste a
  token) rather than showing a raw HTTP error.
- **Token storage**: `config.ron` with `0600` permissions, and never logged.
- **Cancel and clean up** partial files on user abort.
- **Atomic completion**: download to `<name>.part`, rename on success, so a
  half-file is never mistaken for a model.

Layout downloads as `<models_dir>/<org>/<repo>/<revision>/…` so multiple
revisions coexist and profiles pinning a sha keep working.

---

## 10. Connection instructions and security

### 10.1 Bind to loopback by default

crane-serve defaults to `--host 0.0.0.0` with **no authentication** (§2.13).
CraneStudio must:

- default the gateway to `127.0.0.1`;
- always spawn children on `127.0.0.1` — children are only ever reached through
  the gateway;
- require an explicit opt-in for LAN exposure, and when enabled, **require a
  bearer token** and display a plain warning that anyone on the network can use
  the GPU and read the prompts.

### 10.2 What to print when the server is up

For **opencode** (v1 — OpenAI-compatible provider):

```
  opencode — add to ~/.config/opencode/opencode.json:

  {
    "provider": {
      "cranestudio": {
        "npm": "@ai-sdk/openai-compatible",
        "options": { "baseURL": "http://127.0.0.1:1234/v1" },
        "models": { "qwen3.5-9b": { "name": "Qwen 3.5 9B (local)" } }
      }
    }
  }
```

For **Claude Code** (v1.1, once `/v1/messages` exists):

```
  export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
  export ANTHROPIC_AUTH_TOKEN=cranestudio
  claude
```

Verify these against the current opencode and Claude Code documentation at
implementation time; both move. Do not ship instructions you have not executed
end to end on a real machine.

Also print a `curl` one-liner — it is the fastest way for a user to prove the
server works before blaming their client config.

### 10.3 Context policy: 256k target, 32k hard floor

**Target: 262144 tokens, always, by default.** That is the size at which a local
model is actually useful for agentic coding, and the solver (§7.0, §7.2) treats
it as the objective rather than as an option.

**Floor: 32768 tokens, enforced as a refusal.** Claude Code and opencode send a
large system prompt plus tool schemas before the user types a word. Below 32k the
setup is not merely degraded, it is broken — and it fails in a confusing way that
looks like a CraneStudio bug.

So, below 32k, CraneStudio **does not start the server**:

```
  ✕ This configuration reaches only 24576 tokens of context.

    Coding agents need at least 32768 to function at all — their system
    prompt and tool schemas alone consume much of that.

    Fix it by changing one of:
      → KV cache f16 → int4          +73728 tokens   [apply]
      → weights Q6_K → Q4_K_M        +41984 tokens   [apply]
      → use Qwen 3.5 4B instead      reaches 256k    [switch]
```

The refusal always carries at least one concrete way forward — a knob to move or
a smaller model to choose. **Never refuse without an actionable alternative.**

Models whose *native* maximum context is below 32k are excluded from the catalog
for agent use entirely; they can still be run from a local path, with the
limitation stated plainly.

---

## 11. Upstream tracks (work that belongs in Crane, not here)

### 11.0 The division of responsibility

**This repository owns the user-facing application and nothing else.** Anything
that belongs in an inference engine belongs in Crane, and is PR'd to
`lucasjinreal/Crane`.

Belongs **here**: TUI/GUI, hardware probing, the catalog, the VRAM solver, the
download manager, process supervision, the gateway and its API translation,
profiles, connection instructions.

Belongs **upstream**: model support, quantization, KV-cache behaviour, sampling,
the OpenAI/SGLang API surface, device selection, multi-GPU, performance.

**Rule: do not work around a Crane limitation in studio code.** If Crane is
missing something or behaving wrongly, fix it upstream. A workaround here is
acceptable only as a temporary bridge, and only with a code comment linking the
upstream issue or PR that will remove it. Every such comment is a debt entry;
grep for them before a release.

This is not just tidiness — CraneStudio's entire value proposition is exposing
Crane to people who cannot build it themselves. Divergence between what
CraneStudio can drive and what Crane can do destroys that.

### 11.1 U1 — Tool calling in crane-serve (**DONE — merged 2026-08-17**)

**Status: merged as `2078b18`** ([PR #102](https://github.com/lucasjinreal/Crane/pull/102)),
on upstream `main` at `4242e9c` and in the §3.4 pin. Nothing here blocks v1.

Retained below as an integration map — this is where to look when tool calling
misbehaves.

**What the PR touched** (+908 / −30 across 9 files):

| File | Δ | What it is |
|---|---|---|
| `crane-serve/src/tools.rs` | **+302, new** | The tool-calling module — parsing and emission |
| `crane-serve/README.md` | +212 | Documentation |
| `crane-serve/tests/tool_calling.rs` | **+154, new** | Integration tests |
| `crane-serve/src/openai_api.rs` | +114 | Request/response types (`tools`, `tool_choice`, `tool_calls`) |
| `crane-serve/src/chat_template.rs` | +87/−… | Template rendering with tools |
| `crane-serve/src/handlers/sse.rs` | +35 | Streaming tool-call deltas |
| `crane-serve/src/handlers/openai.rs` | +31 | Handler wiring |
| `crane-serve/src/lib.rs` | +1 | Module registration |
| `crane-serve/src/reasoning.rs` | +2/−1 | Minor |

That covers every part of the feature — types, template rendering, output
parsing, streaming, and tests. The request/response surface is documented from
the merged source in §2.3.

**Open items CraneStudio still needs to verify** (none block starting work):

1. **`tool_choice` is advisory** (§2.3) — `"none"` works; `"required"` and
   named-function forcing are accepted and ignored. Exercise this during M0's
   opencode test. If it bites in practice, constrained decoding is upstream
   work, not a gateway workaround (§11.0).
2. **Per-model tool support is not obviously reported.** The catalog's
   `capabilities: [Tools]` should be read from the server rather than guessed,
   so a tool-incapable model (Qwen 2.5) is never offered to a coding agent.
   Check whether `/v1/models` or `/model_info` exposes this; if not, adding it
   is a small, well-scoped upstream PR.
3. **Read `crane-serve/src/tools.rs` and `tests/tool_calling.rs`** before relying
   on specifics — which families are wired up, and whether streaming emits
   correct `index` values across multiple parallel tool calls.

**Original scope, retained as a map of what lives where:**

1. **Request types** (`crane-serve/src/openai_api.rs`): add `tools:
   Option<Vec<ToolDef>>` and `tool_choice: Option<ToolChoice>` to
   `ChatCompletionRequest`. Add `tool_calls: Option<Vec<ToolCall>>` and
   `tool_call_id: Option<String>` to `ChatMessage`, and accept `role: "tool"`.
2. **Prompt rendering** (`crane-serve/src/chat_template.rs`): route through
   `AutoTokenizer::apply_chat_template_with_tools`, which already exists and is
   already HF-byte-identical — see the working reference in
   `example/src/ornith_tools.rs`.
3. **Output parsing**: models emit tool calls as structured text
   (`<tool_call>{...}</tool_call>` for Qwen). Parse into `tool_calls` and strip
   from `content`. Mirror the existing `reasoning_content` split in
   `crane-serve/src/reasoning.rs` — it solves the structurally identical problem
   of extracting a tagged region from the token stream, including in streaming
   mode, and is the right file to read before starting.
4. **Streaming**: emit tool calls as OpenAI streaming deltas with correct
   `index` values, and set `finish_reason: "tool_calls"`.
5. **Tests**: no GPU needed for most of it. Test template rendering against
   fixtures, and parser round-trips against recorded model outputs. Model an
   integration test on `crane-serve/tests/thinking_control.rs`.

Per-family support differs — Qwen 3.5/Ornith have proper tool templates; Qwen 2.5
does not. Surface which loaded model supports tools via `/v1/models` or
`/model_info`, so CraneStudio's catalog `capabilities` field can be trusted
rather than guessed.

### 11.2 U2 — `--quant` (ISQ) beyond Qwen 3.5 (does not block v1)

Today `--quant` hard-fails for any `ModelType` other than `Qwen3_5`
(`model_factory.rs:446`, verified — it `bail!`s with a message pointing at
GGUF). That leaves every other family dependent on someone having published a
GGUF.

Worth doing upstream, in rough priority order:

1. **Extend ISQ to the other dense families** (Qwen 3, Qwen 2.5, Gemma 4,
   Hunyuan). The `QMatMul` machinery already exists; the work is per-family
   loader plumbing.
2. **Fix the two documented ISQ weaknesses** from `AGENTS.md` §"Known gaps",
   which is what would make ISQ genuinely competitive rather than a fallback:
   quantize `embed_tokens` instead of leaving it dense (`EmbeddingLayer` already
   accepts a `QTensor` unchanged — described upstream as a small change), and
   support a **per-tensor-class dtype mix** so `ffn_down` and `output` can take
   Q6_K the way Q4_K_M does. `candle` also offers `QTensor::quantize_imatrix`,
   which Crane's ISQ does not currently use.
3. **Allow `--quant` on VL checkpoints** rather than requiring `--text-only`.

**Priority is driven by catalog gaps:** whenever a model users want has no
published GGUF, that family moves up this list. Until then, the catalog prefers
GGUF (§2.9) and this track stays non-blocking.

### 11.3 U3 — Multi-GPU (post-v1, separate effort)

Out of scope for this repository entirely (§2.6). Noted here only so nobody
attempts to emulate it in studio code.

---

## 12. Release pipeline

`.github/workflows/release.yml`, triggered on tags.

**v1 matrix:**

| Target | Runner | Features | Notes |
|---|---|---|---|
| `x86_64-unknown-linux-gnu` + CUDA | `ubuntu-latest` + CUDA toolkit | `cuda` | Set `CUDA_COMPUTE_CAP` explicitly (no GPU on the runner). Build for a baseline cap; PTX JITs forward. Consider `CUDA_COMPUTE_CAP=75` for wider reach. |
| `aarch64-apple-darwin` + Metal | `macos-latest` | `metal,accelerate` | |

**Planned, not in v1 — but keep the doors open (§13):**

| `x86_64-unknown-linux-gnu` CPU-only | `ubuntu-latest` | *(none)* | Always builds. Ship as the universal fallback. |
| `x86_64-unknown-linux-gnu` + ROCm | — | `rocm` | Blocked on §2.5. |

Requirements:
- Static-link what can be statically linked. The CUDA build will dynamically
  link the CUDA driver — that is unavoidable and expected; the user's NVIDIA
  driver provides it. Document the minimum driver version.
- Build against an **old glibc** (build in a manylinux-style container or on the
  oldest available Ubuntu runner). A binary built on the newest Ubuntu will not
  run on Debian stable, which is a large share of the target audience.
- On startup, if the CUDA build cannot initialise a device, **do not crash** —
  print a clear diagnostic (driver missing / too old / no NVIDIA GPU) and point
  at the CPU build.
- Publish sha256sums with the release.
- Name assets `cranestudio-<version>-<target>-<backend>.tar.gz`, e.g.
  `cranestudio-0.1.0-x86_64-linux-cuda.tar.gz`. The backend must be in the
  filename — this is exactly the confusion that sinks first-run experiences.

**CI (on every PR):** `cargo fmt --check`, `cargo clippy -- -W clippy::pedantic
-D warnings` (**gating**, unlike Crane's), `cargo test --workspace`, and a
CPU-only build. Do not attempt GPU tests in CI.

---

## 13. Keeping the doors open for CPU-only and ROCm

Neither ships in v1, but nothing may be designed in a way that blocks them:

- `Backend` is an enum (`Cuda | Metal | Cpu | Rocm`) from the start, not a
  boolean. Every backend-conditional code path matches on it exhaustively so
  adding a variant produces compile errors at exactly the sites needing work.
- The estimator takes backend-specific constants (context overhead, correction
  factors) from a table keyed by `Backend`, not from `#[cfg]` blocks.
- Hardware probing has a per-backend trait implementation. Adding ROCm means
  adding one impl (and it can crib from
  `crane/src/utils.rs::rocm_gpu_memory_info`).
- Measurement DB keys already include the backend (§7.3), so ROCm data will not
  contaminate CUDA predictions.
- Catalog `measured` blocks are keyed by backend string
  (`cuda_sm86`, `metal_m3`, `rocm_gfx1101`, `cpu`).
- The CPU path needs an honest UX: a 9B model on CPU is ~2 t/s. Show the
  predicted rate and let the user decide, rather than letting them discover it.
- **ROCm additionally needs a first-run check for `hipcc`** (§2.5) with a clear
  message if absent, since kernels compile on the user's machine.

---

## 14. Milestones

Each milestone is independently demoable. Do not start the next until the
acceptance criteria pass.

### M0 — Skeleton
Workspace with the §3.3 crate layout. Rename per §3.5. `rust-toolchain.toml`.
Crane pinned per §3.4 (plain upstream rev, no `[patch]`).
**Accept:** `cranestudio __serve -m <path> -p 8080` behaves identically to
`crane -m <path> -p 8080`, and a `curl` request carrying a `tools` array gets a
`tool_calls` response back. This proves the single-binary architecture and the
tool-calling dependency in one step, de-risking everything downstream.

### M1 — Hardware probing
§6, all backends this build supports.
**Accept:** `cranestudio doctor` prints an accurate report on a CUDA machine and
an Apple Silicon machine, including *free* VRAM.

### M2 — Catalog and search
§8. Catalog loading (remote + bundled fallback), the browser screen with fit
verdicts stubbed to "unknown", filtered HF search, local path scanning.
**Accept:** browsing the catalog and searching HF both work offline-tolerantly;
unsupported architectures are shown greyed with a reason.

### M3 — Download manager
§9.
**Accept:** a 5 GiB download can be interrupted with Ctrl-C and resumes from
where it stopped; disk-space precheck fires; gated-repo error is actionable.

### M4 — Estimator and solver
§7.0, §7.1, §7.2. Wire real fit verdicts into M2's browser.
**Accept:** the unit test reproducing Qwen 3.8-27B's 64 KB/token from its real
`config.json` passes. The solver returns a 256k-capable configuration for
Qwen 3.5 9B on a 24 GiB card, and correctly returns `Short` or `Unusable` with
actionable suggestions where 256k is out of reach. Predictions for three known
models land within 20% of measured reality.

### M5 — Supervisor and daemon
§3.1, §3.1a, §3.2 (control API only, no `/v1` multiplexing yet). Spawn,
health-poll, capture logs, classify exits (§7.4), detach lease.
**Accept:** the daemon starts and stops children; killing a child does not kill
the daemon; a deliberately over-provisioned launch produces a correct OOM
classification with quantitative advice. **Orphan test: `kill -9` the control
client and assert every child is gone within the grace period** — run this on
every CI build, it is the regression that matters most.

### M6 — Gateway
§3.2. `/v1/models` aggregation, routing by model name, on-demand start, LRU
eviction, SSE passthrough.
**Accept:** with two models configured, a single unchanged base URL serves both,
selecting by the `model` field, starting the second on demand.

### M7 — TUI
§4, all screens including the solver-led wizard and chat pane, plus the quit
prompt (§3.1a).
**Accept:** a user who has never used CraneStudio goes from launch to a running
server without reading documentation, and the resulting server has 256k context.
Quitting the TUI never leaves a model resident without an explicit "keep
serving".

### M8 — Profiles and measurement
§5, §7.3. Save/load/delete named profiles; measurement DB; measured-vs-predicted
labelling; correction factors.
**Accept:** a second launch of the same configuration shows a measured number;
an OOM'd configuration is remembered and pre-warned on the next attempt.

### M9 — Release pipeline
§12.
**Accept:** a tagged release produces working CUDA and Metal binaries, verified
by downloading and running them on clean machines (**not** the build machines —
a binary that only runs where it was built is the failure mode this milestone
exists to catch).

### M10 — v1 ship gate
- Crane pinned to a plain upstream `rev`, **no `[patch]` section present**
  (§3.4). *(All upstream prerequisites — PRs #99, #100, #102 — landed by
  2026-08-17 and are in the pin. There are no outstanding upstream
  dependencies.)*
- opencode connects and successfully completes a multi-turn tool-using task
  against a locally served model **at 256k context**.
- No Python anywhere in the repository (§2.15).
- No un-linked Crane workarounds in studio code (§11.0).
- All §14 acceptance criteria pass on both shipped platforms.

### M11 — v1.1: Anthropic surface
`/v1/messages` in `studio-gateway`: request/response mapping and the SSE event
grammar (`message_start`, `content_block_start`/`_delta`/`_stop`,
`message_delta`, `message_stop`). Tool-use blocks map to/from OpenAI
`tool_calls`.
**Accept:** Claude Code connects via `ANTHROPIC_BASE_URL` and completes a
multi-turn tool-using task. Fully testable against a mock backend without a GPU —
build it that way.

### Post-v1 (not planned in detail here)
GUI over the same control API; per-model "apps" (VoxCPM2 et al. — already on
upstream main, §2.16, so unblocked); TTS/ASR/OCR model types; CPU-only and ROCm
targets (§13); multi-GPU once
Crane supports it (§2.6); opt-in community measurement sharing.

---

## 15. Rules for implementers

1. **Verify against the Crane source before assuming.** This document cites file
   paths and line numbers precisely so you can check. Crane moves; if you find a
   discrepancy, **update this document in the same PR** rather than working
   around it silently.
2. **Never guess architecture-specific memory math.** Read the modeling code.
   Guessed formulas produce confident wrong numbers, which is worse than
   admitting uncertainty.
3. **`studio-core` stays free of terminal code.** Every leak of rendering into
   core is a tax on the future GUI.
4. **Prefer measurement over prediction everywhere**, and always tell the user
   which one they are looking at.
5. **Error messages are the product.** The target user cannot read a Rust
   backtrace. Every failure path needs a plain-language cause and a specific,
   quantitative next action.
6. **Keep files under 400 lines** (§2.17). Split before you exceed it.
7. **Clippy pedantic is gating.** Do not add `#![allow]` at crate level.
8. **Do not ship instructions you have not run.** Every copy-paste block in §10
   must have been executed end to end on a real machine.
9. **No Python in this repository** (§2.15) — not runtime, not build, not CI,
   not tooling. Use Rust or `cargo xtask`.
10. **RON for everything CraneStudio serializes** (§5). Not TOML, not JSON.
11. **Fix Crane upstream, don't work around it here** (§11.0). Any temporary
    bridge carries a comment linking the upstream issue or PR.
12. **256k is the goal, 32k is a hard floor** (§7.0, §10.3). The wizard proposes
    a solution; it does not present a blank form of knobs.
13. **Never leave a process behind** (§3.1a). When in doubt, kill it.
