# CraneStudio

**A local model studio in the terminal.** Pick a model, press enter, get an
OpenAI-compatible server your coding agent can talk to. No Python, no toolchain,
no menu-diving.

CraneStudio is a Rust TUI + daemon built on the
[Crane](https://github.com/lucasjinreal/Crane) inference stack (Candle-based,
pure Rust — no llama.cpp).

```
                                  ▀▚▄▄▖                      ▗▄▄▞▀
                                       ▀▀▚▄▄▖          ▗▄▄▞▀▀
                             ▂▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▚▄▟█▙▄▞
                                                    ▀▀▚▄▄▄▄▄▄▂
                            ██████╗██████╗  █████╗ ███╗   ██╗███████╗
                           ██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝
                           ██║     ██████╔╝███████║██╔██╗ ██║█████╗
                           ██║     ██╔══██╗██╔══██║██║╚██╗██║██╔══╝
                           ╚██████╗██║  ██║██║  ██║██║ ╚████║███████╗
                            ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝
                           ███████╗████████╗██╗   ██╗██████╗ ██╗ ██████╗
                           ██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║██╔═══██╗
                           ███████╗   ██║   ██║   ██║██║  ██║██║██║   ██║
                           ╚════██║   ██║   ██║   ██║██║  ██║██║██║   ██║
                           ███████║   ██║   ╚██████╔╝██████╔╝██║╚██████╔╝
                           ╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝

                             run local models › pick one and press enter
                                  ✓ hardware   ✓ models   ✓ catalog
                                            press any key
```

---

## Download

No toolchain, no Python, no install — one binary. Grab it from
**[Releases](../../releases)** (the GitHub mirror carries the same archives):

| Archive | For |
| --- | --- |
| `cranestudio-<version>-x86_64-linux-cuda.tar.gz` | Linux with an NVIDIA GPU (driver 525 or newer) |
| `cranestudio-<version>-x86_64-windows-cuda.zip` | Windows with an NVIDIA GPU (driver 525 or newer) |
| `cranestudio-<version>-aarch64-macos-metal.tar.gz` | Apple Silicon Macs |
| `cranestudio-<version>-x86_64-linux-cpu.tar.gz` | Linux without a usable GPU — runs everywhere, slow on big models |
| `cranestudio-<version>-x86_64-windows-cpu.zip` | Windows without a usable GPU |

```sh
tar xzf cranestudio-*-x86_64-linux-cuda.tar.gz
cd cranestudio-*-x86_64-linux-cuda
./cranestudio doctor     # what does this machine have?
./cranestudio            # run it
```

```powershell
Expand-Archive cranestudio-*-x86_64-windows-cuda.zip -DestinationPath .
cd cranestudio-*-x86_64-windows-cuda
.\cranestudio.exe doctor
.\cranestudio.exe
```

Not sure which one? Take the CPU archive and run `cranestudio doctor` — it prints
the GPU, VRAM, driver and disk it found, and says so plainly when a GPU build
would find no GPU.

On Windows, run it in **Windows Terminal** (or any modern terminal), not the
legacy console host: the meters and the splash use block and braille glyphs that
`conhost.exe` renders as boxes.

Every archive is built by the pipeline from the tagged commit — never uploaded by
hand — and ships with its SHA-256 plus a `BUILD-PROVENANCE.txt` naming the
commit, the Rust version, the pinned Crane revision and the job that produced it,
so you can rebuild the same binary yourself (see [Building](#building)).

---

## Quick start

```sh
cranestudio            # that's it — the TUI starts the daemon for you
```

The splash gives way to the **launchpad**: live hardware on top, everything
runnable underneath.

```
◆ CraneStudio                                              CUDA · NVIDIA GeForce RTX 30… · :1234

╭ Hardware ────────────────────────────────────────────────────────────────────────────────────╮
│ cpu    ████████████░░░░░░░░░░░░░░░░░░   39%  AMD Ryzen 9 5900X · 12c/24t                     │
│        ▆▆▆▅▅▅▄▄▃▃▃▂▂▂▂▁▁▁▁▁▂▂▂▂▃▃▃▄▄▅  cores ▁▂▃▅▆▇▁▂▄▅▆▇▁▂▄▅▆▇▁▃▄▅▆▇                        │
│ ram    ██████████████████░░░░░░░░░░░░   59%  19.0 GiB / 32.0 GiB                             │
│ vram   █████████░░░░░░░░░░░░░░░░░░░░░   29%  7.0 GiB / 24.0 GiB · NVIDIA GeForce RTX 3090    │
│ disk   ██████████████████░░░░░░░░░░░░   60%  712.0 GiB free for models                       │
╰──────────────────────────────────────────────────────────────────────────────────────── CUDA ╯
╭ Models  ·  4 on disk ────────────────────────────────────────────────────────────────────────╮
│   ● Qwen3.5-9B-Instruct-Q4_K_M                                               127.0.0.1:41100 │
│     ready — enter to open apps                                                               │
│ ▌ ○ gemma-4-4b-it-Q6_K                                                               3.2 GiB │
│     gemma4  ·  GGUF  ·  Q6_K  ·  unsloth/gemma-4-4b-it-Q6_K-GGUF                             │
│   ○ MiniCPM-V-4_6-Q4_K_M                                                             2.4 GiB │
│     minicpmv4_6  ·  GGUF  ·  Q4_K_M  ·  unsloth/MiniCPM-V-4_6-Q4_K_M-GGUF                    │
│   ✕ Phi-3-mini-4k-instruct-Q8_0                                                      3.8 GiB │
│     GGUF  ·  Q8_0  ·  unsloth/Phi-3-mini-4k-instruct-Q8_0-GGUF  ·  phi3 is not a Crane-suppo │
│                                                                                              │
╰─────────────────────────────────────────────────────────────────────────────────── 1 serving ╯

↑↓ select   ⏎ run   c configure   g get models   d hardware   f2 theme   q quit
```

`⏎` on a model solves for the best configuration this machine can actually run
and starts it — no wizard in the way. When it's up you land on the **ready**
screen, where the model's apps live: **Chat** for text/VL models, **TTS
Playground** for speech models (VoxCPM2). A speech model skips the solver — it
has no context/KV trade-offs to pick — and starts straight away.

### Keys

| Screen | Keys |
| --- | --- |
| Launchpad | `↑↓`/`jk` select · `⏎` run (or open a running model's apps) · `c` launch options · `g` get models · `d` hardware report · `R` rescan disk · `q` quit |
| Get models | `tab`/`←→` catalog / on disk / HuggingFace · `/` search · `⏎` download or launch · `esc` back |
| Launch options | `↑↓` pick an alternative · `⏎` start · `esc` back |
| Ready | `↑↓` select app · `⏎` open (Endpoint toggles the connect panel) · `esc` back to models |
| Chat | `⏎` send · `esc` stop generating, then leave · `^a` attach image · `^p` system prompt · `^l` max tokens · `^t` temperature · `^n` new chat · `PgUp`/`PgDn` scroll |
| TTS Playground | `⏎` generate · `space`/`p` play or stop the selected clip · `↑↓` select a past clip · `^r` restart the voice model (fresh RNG) · `^n` clear · `esc` back |
| Anywhere | `F2` cycle theme · `^c` quit |

Quitting always asks what to do with loaded models: **keep serving** in the
background, or **stop everything**. It never leaves a model resident by accident.

### Connecting a client

The gateway is one stable loopback URL that does not change when you switch
models:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:1234/v1
export OPENAI_API_KEY=not-needed

curl $OPENAI_BASE_URL/chat/completions \
  -d '{"model":"Qwen3.5-9B-Instruct-Q4_K_M","messages":[{"role":"user","content":"hi"}]}'
```

Naming a model that isn't running starts it on demand. `/v1/models` lists
everything registered, running or not.

---

## Capabilities

**Hardware-aware, always.**
GPU/VRAM (free, not just total), CPU, RAM and disk are probed at startup and
re-sampled live while the TUI is open — the launchpad's meters are the real
numbers, so you can watch a model take VRAM as it loads.

**A solver, not a form.**
CraneStudio's target is the *full* context (256k where the model supports it).
It reads the model's own dimensions (GGUF header or `config.json`), predicts
weights + KV cache + prefill + runtime overhead, and searches for the
configuration that reaches it — KV-cache quantization included, but only for
families that actually support it. Then it tells you the answer and lets you pick
a different trade-off if you want one.

**Numbers you can trust.**
Every launch's peak VRAM is measured and remembered. On the next launch of the
same configuration you see `measured 11.9 GiB` instead of `predicted`, and a
configuration that OOM'd once is flagged before you try it again. Off-target
predictions feed a correction factor that shrinks the budget the solver plans
inside.

**Models from anywhere.**
A curated catalog (every entry actually launched and exercised), filtered
HuggingFace search that only shows architectures Crane can run, and a scan of
your own disk. Unsupported models are shown greyed with the reason rather than
hidden.

The scan follows symlinks, so models already living somewhere else (another
drive, a shared cache) don't need copying — just link them in:

```sh
ln -s /mnt/big/models/VoxCPM2 ~/.local/share/cranestudio/models/VoxCPM2
```

**Downloads that survive.**
Resumable, checksum-verified, with live throughput, ETA and per-file progress.
Interrupting and restarting picks up where it stopped. Gated repos give an
actionable error (`cranestudio config set hf-token …`).

**A daemon you don't have to think about.**
The TUI starts it, holds a lease on it, and shuts it down when you leave unless
you asked it to keep serving. Children are health-polled, their exits are
classified (OOM vs. bad flag vs. crash), and `kill -9` on the TUI never leaves
an orphaned model holding your VRAM. Port conflicts fall forward instead of
failing.

**Apps, not just a chat box.**
Anything running is reachable from the ready screen's app list. Chat —
streaming, token-rate readout, image attachments for vision models, editable
system prompt / `max_tokens` / `temperature`, dimmed chain-of-thought for
reasoning models. **TTS Playground** — type text, generate speech with a live
progress readout, and play the clip back in the terminal (every clip is also
saved as a `.wav` under the data dir). Benchmark and agent-tool apps are next.

**Supported model families:** Qwen 3.5/3.6/3.8 (incl. VL), Qwen 3, Qwen 2.5,
Hunyuan, Gemma 4 (incl. VL), MiniCPM-V 4.6, MiniCPM5, VoxCPM2 (text-to-speech)
— GGUF and safetensors.

Audio playback on Linux uses ALSA — install `libasound2` (Debian/Ubuntu:
`apt install libasound2`) if the TTS Playground reports no output device.
macOS and Windows need nothing extra.

---

## CLI

The TUI is the default, but everything it does has a plain command behind it.

```sh
cranestudio                       # TUI (same as `cranestudio tui`)
cranestudio doctor                # hardware report, plain text — attach this to bug reports
cranestudio catalog list          # curated catalog
cranestudio catalog scan [PATH]   # classify local checkpoints (default: models dir)
cranestudio catalog search QUERY  # filtered HuggingFace search
cranestudio download REPO REV FILES...   # resumable download
cranestudio config set hf-token TOKEN    # stored 0600, never logged
cranestudio launch PATH --model-type auto --max-seq-len 262144
cranestudio register NAME PATH --port 41100   # visible in /v1/models, starts on demand
cranestudio status                # what the daemon is running
cranestudio stop                  # stop every model and the daemon
cranestudio attach                # hold the daemon's lease from a script
cranestudio daemon                # run the daemon in the foreground
```

**Ports:** gateway `:1234`, control API `:41999` — both loopback only. Override
with `CRANESTUDIO_GATEWAY_PORT` / `CRANESTUDIO_CONTROL_PORT`; a busy port falls
forward to the next free one automatically.

**Paths** (Linux; macOS uses `~/Library/Application Support/cranestudio/`):

| What | Where |
| --- | --- |
| Settings, HF token, theme | `~/.config/cranestudio/config.ron` |
| Models | `~/.local/share/cranestudio/models/` (symlinked entries are followed) |
| Measurements, catalog cache | `~/.local/share/cranestudio/` |

---

## Building

Rust stable (`rust-toolchain.toml` pins the channel), edition 2024.

```sh
cargo build --release --locked                      # CPU
cargo build --release --locked --features cuda      # NVIDIA
cargo build --release --locked --features metal     # Apple Silicon
```

Other passthrough features: `rocm`, `cudnn`, `mkl`, `accelerate`. The GPU backend
is a **compile-time** choice — one binary per backend, and `cranestudio doctor`
warns when a GPU build finds no GPU.

`crane-serve` is pinned to a specific Crane revision in the workspace
`Cargo.toml`, so a plain clone builds without any local checkout. `--locked` is
what the release pipeline uses: same source, same lockfile, same binary.

---

## Contributing

### The plan is the spec

[`PLAN.md`](PLAN.md) is the design document, and it's the thing to read first —
especially §2, which records **verified facts about Crane that contradict common
assumptions** (one model per process, compile-time GPU selection, KV-cache
compression being Qwen-3.5-family only, and so on). Code comments cite plan
sections (`§7.2`, `§3.1a`) on purpose: when you change behaviour, update the
section it cites.

### Layout

| Crate | What lives there |
| --- | --- |
| `studio-core` | hardware probing, VRAM estimator + solver, catalog, downloader, config, paths, measurements |
| `studio-supervisor` | child processes: spawn, health, log ring, exit classification |
| `studio-gateway` | the `/v1` gateway and the control API the TUI drives |
| `studio-tui` | ratatui screens, theme, shared UI chrome |
| `cranestudio` | the binary: CLI, daemon entry point, `__serve` re-exec target |

Inside `studio-tui`:

- `ui/` — the shared chrome. `ui::shell` gives a screen its body `Rect`; screens
  never lay out the terminal themselves. Content is centered and capped at
  `ui::MAX_WIDTH` — the terminal is deliberately never "filled".
- `ui/bars.rs`, `ui/art.rs`, `ui/text.rs` — meters and sparklines, splash art,
  wrapping/truncation.
- `theme.rs` — semantic colors only. Screens ask for `theme.success`, never
  `Color::Green`; a new palette is one entry in `Theme::from_name`.
- `screens/` — one module per screen, each exposing `render` and (where it takes
  input) `handle_key`.

### Looking at the UI without launching a model

Every screen is rendered into a `TestBackend` with plausible fake state — a mock
GPU, models on disk, a real solver answer, a download mid-flight. Print them all:

```sh
cargo test -p studio-tui --lib preview -- --nocapture
```

That's the fastest way to iterate on layout, and it's also a regression test:
`every_screen_survives_extreme_sizes` draws all of them from 20×5 to 250×80, so
a screen that panics in someone's tmux split fails in CI instead. Add a preview
case in `crates/studio-tui/src/preview.rs` whenever you add a screen or a state
worth eyeballing.

### House style

- `cargo fmt` is gating.
- `cargo clippy --all-targets -- -W clippy::pedantic` must be clean.
- `cargo test --workspace` must pass.
- Files stay under ~400 lines; oversized modules get split.
- Comments explain *why*, not *what* — the existing ones are the reference. If a
  decision was surprising, or was made because of something verified against
  Crane's source or a live model, say so.
- Adding a catalog model is meant to be a pure data change to
  [`catalog/models.ron`](catalog/models.ron) — and every entry there has actually
  been downloaded, launched and exercised against a real request. Keep that bar;
  entries still awaiting that last step say `PENDING LAUNCH CHECK` so they can be
  found with one grep.

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -W clippy::pedantic
cargo test --workspace
```

The same three commands gate every push in CI. Two suites are *not* run there:
GPU tests (there is no GPU on a check runner) and the network-dependent
`--ignored` tests, which run on a schedule instead —
`every_variant_is_still_downloadable_at_its_pinned_sha` HEADs every file of every
catalog entry and fails when HuggingFace stops serving what the catalog promises.

### Releases

[`.gitlab-ci.yml`](.gitlab-ci.yml) is the whole release process, and the only
thing that ever produces a published binary. Tag a commit and the pipeline
builds one archive per backend, publishes them with checksums to GitLab
Releases, then mirrors the source *and* the identical archives to GitHub
Releases.

```sh
git tag -a v0.1.0 -m "CraneStudio 0.1.0" && git push origin v0.1.0
```

Because the backend is chosen at compile time, each one needs its own runner —
register them with the tags the pipeline expects:

| Tag(s) | Runner | Builds |
| --- | --- | --- |
| `docker` | any Linux docker executor | checks, Linux CPU archive, releases, mirror |
| `cuda` | Linux + NVIDIA GPU with the CUDA toolkit (`nvcc`) | Linux CUDA archive |
| `macos`, `metal` | Apple Silicon, shell executor with Xcode CLT | Metal archive |
| `windows` | Windows shell executor (PowerShell) + MSVC build tools + NASM | Windows CPU archive |
| `windows`, `cuda-windows` | the same, plus the Windows CUDA toolkit | Windows CUDA archive |

The Windows jobs install rustup, import the Visual Studio developer environment
and install NASM via Chocolatey when they're missing, so a bare Windows box only
needs the MSVC build tools present (or reachable through `vswhere`). NASM isn't
avoidable: `reqwest` pulls `rustls`, which builds aws-lc's assembly, and
`crane-serve` enables the same stack.

Set `GITHUB_TOKEN` (masked, `contents: write` on the mirror) in the project's
CI/CD variables; everything else uses GitLab's own job token. A shell runner that
already has CUDA and rustup can set `CRANESTUDIO_SHELL_RUNNER=1` to skip the
job's own toolchain setup.

### License

MIT.
