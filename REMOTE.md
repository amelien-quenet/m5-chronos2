# Running the heavy stages on the GPU tower

The laptop that wrote this study is an Intel MacBook: CPU only, torch pinned at
2.2.2. The LoRA fine-tuning of notebook 08 needs about **40 hours** there. On a
GTX 1080 reached over SSH it needs **under one**. This file is how that works,
why each piece is the way it is, and what to do when it breaks.

Measured on 2026-10-12, 100 series, context 1829, batch 32:

| | 1000 LoRA steps | inference, 500 series |
|---|---|---|
| MacBook, CPU | 764 min | — |
| Tower, GTX 1080 | 32 min | 4 s |

The factor on the operation that matters is about **24×**.

---

## 1. The architecture

```
MacBook                              Tower (Windows 10 -> WSL2 -> GTX 1080)
  decide, code, plot, write
  git push ------------------------>  git pull          CODE travels by git
  ssh "run stage X" -------------->  tmux + python     COMMANDS travel by ssh
  rsync <------------------------- artifacts/*.parquet RESULTS travel by rsync
  notebooks read parquets
```

Three rules follow, and they are the whole design:

**The unit of work is a stage, not a notebook.** A stage is a pure function
`config -> artifacts/<name>.parquet`: autonomous (everything is on the command
line, nothing in a kernel's state), it writes to disk, and it is resumable.
`chronos2.cached()` already implemented that contract; `scripts/run_stage.py`
only exposes it on the command line so that a long run can be launched and
forgotten.

**The notebooks compute nothing.** Stages call exactly the same functions, to
exactly the same artifact paths. So after a run on the tower and an
`rsync`, every notebook re-runs end to end in seconds: each `cached()` finds its
parquet. The notebooks are the narrative layer — scores, figures, prose.

**Two decisions stay human**, and the runner refuses to make them:
`artifacts/context_choice_<roster>.json`, written by notebook 07 after you look
at the context sweep, and the fine-tune context passed explicitly to
`finetune_test`. Deriving either automatically from the data would amount to
selecting on the test window, which notebook 03's protocol forbids.

---

## 2. Everyday use

```bash
./scripts/remote.sh check            # is the tower reachable, which commit, which device
./scripts/remote.sh run --all        # launches detached, returns immediately
./scripts/remote.sh log              # follow the log (Ctrl-C stops watching, not the run)
./scripts/remote.sh status           # TERMINE / EN COURS / INTERROMPU
./scripts/remote.sh stop             # kill the run; written artifacts are kept
./scripts/remote.sh pull             # artifacts tower -> laptop
./scripts/remote.sh push             # artifacts laptop -> tower (needed for context_choice.json)
python scripts/run_stage.py --list   # the stages, their cost, their dependencies
```

`run` pushes the code, checks that both sides are on the same commit, and runs
under `tmux`: closing the laptop does not stop anything. It **refuses** to launch
with uncommitted changes in `src/`, `scripts/` or the pyproject — otherwise the
tower would compute with code older than yours, and the artifact's `.meta.json`
would carry the wrong commit.

### The full sequence, from nothing

```bash
./scripts/remote.sh run --all                       # ~3 min: vanilla, covariates,
                                                    # cross-learning, blanks, context sweep
./scripts/remote.sh pull
# run notebook 07 on the laptop -> writes context_choice_<roster>.json
./scripts/remote.sh push                            # the tower must see that decision
./scripts/remote.sh run --stages finetune,finetune_predict      # ~16 min
./scripts/remote.sh pull
./scripts/remote.sh run --stages finetune_test --finetune-context 1099,1829   # ~20 min
./scripts/remote.sh pull
# run notebook 08 on the laptop, skipping the timing cell (see §6)
```

---

## 3. The routing rule

Transfers on a LAN are negligible (a prediction parquet is ~450 KB), so the
threshold is low. Two criteria, not one:

**Remote** if (GPU-accelerable **and** local time > 10 min) **or** (local time >
45 min, whatever the nature) **or** (RAM > ~12 GB).

**Local, always**, if the result is a figure, a protocol decision, or anything
you will iterate on ten times tonight.

Note the second criterion: moving the full 30,490 series to the tower is not
about speed — without GPU acceleration a 6-core i7-8700 is barely faster than
the laptop — it is about **RAM and asynchrony**. You launch and keep working.
That is a valid reason, but it is a different reason, and it should be named as
such.

A stage marked `needs_gpu` refuses to run on CPU. With a factor of 24, doing so
is not a choice but an oversight; `--allow-slow` lifts the refusal when it is
deliberate.

---

## 4. The invariant

**Any two rows whose difference you interpret must share their device.** CPU and
GPU differ in reduction order and in kernels. The gaps are tiny — and quite
enough to invert a close ranking.

Consequences, as applied here:

- **vanilla, the covariate runs, the context sweep and the fine-tuned models all
  come from the GPU.** Their differences are sometimes in the third decimal of
  WQL, so they must share hardware. The CPU corpus computed before the tower
  existed was archived to `artifacts/_cpu_reference/` and re-computed; it was not
  reused.
- **XGBoost stays on CPU**, and that is harmless: the methodological gap with
  Chronos-2 is orders of magnitude above hardware noise.
- **The naive baselines** (zero, seasonal naive, 28-day mean) are deterministic
  arithmetic, insensitive to the device.

Every artifact produced by the runner carries a `<name>.parquet.meta.json`:
device, GPU, capability, torch/transformers/chronos/polars/numpy versions, seed,
roster, git commit, duration, dtype. Two checks worth running after any session:

```bash
# an experiments/ parquet without its meta is a CPU intruder, computed by a notebook
for f in artifacts/experiments/*.parquet; do [ -f "$f.meta.json" ] || echo "NO META: $f"; done

# and every meta must say cuda
grep -l '"device": "cpu"' artifacts/experiments/*.meta.json
```

Exception: `predictions/chronos2_best_*.parquet` has no meta, and should not —
it is a `stitch()` concatenation of already-computed GPU forecasts, with no
model involved.

---

## 5. Hardware constraints that changed the code

The GTX 1080 is **Pascal (sm_61, 2016)**, and two of its properties are
counter-intuitive enough that the code documents them in place.

**Half precision is a slowdown, not a speedup.** Pascal has no bfloat16 hardware
at all and runs float16 at **1/64** of its float32 rate. Half precision is the
usual default on CUDA, so left alone the GPU would be slower than the laptop.
`get_pipeline` forces float32; `pipeline_dtype()` reads back what was actually
loaded — the only trustworthy confirmation — and `run_stage.py` aborts before
engaging hours of compute if it is not float32. No `fp16`/`bf16` flag is passed
to the trainer.

**8 GB of VRAM** cannot hold batch 32 at context 1829 in float32. `finetune`
keeps `batch_size` as the *effective* batch — the one that defines the update and
therefore the result — and adds `micro_batch_size` with gradient accumulation:
4 series at a time, update after 8 of them. Same gradient, a fraction of the peak
memory. `suggest_micro_batch` splits on a small GPU and leaves CPU alone.

**`sm_61` is absent from torch's arch list** (`sm_50, sm_60, sm_70, …`) and it
works anyway, through CUDA's minor-version binary compatibility: code built for
`sm_60` runs on any `sm_6x`. Reading the list would have concluded, wrongly, that
the card was unsupported. Measure, do not read release notes.

**torch versions**: `2.2.2` on darwin, `2.6.0` on linux, by platform marker in a
single `pyproject.toml`. Only torch varies; everything else is pinned
identically across machines, which is the condition for the two sides to be
comparable. The CUDA 12.x builds are the last to include Pascal — it was dropped
from 12.8+ and does not exist in CUDA 13.x — so **torch ≥ 2.7 published as cu128
is out of reach for this card**.

---

## 6. Traps, all verified on 2026-10-12

**Do not install an NVIDIA driver package inside Ubuntu.** In WSL, `nvidia-smi`
and `libcuda.so` are *injected by Windows* into `/usr/lib/wsl/lib`. They are not
Ubuntu packages. `nvidia-smi: command not found` never means a package is
missing; it means Windows injected nothing, and the problem is on the Windows
side. Installing a Linux driver over it overwrites `libcuda.so` and breaks the
passthrough for good.

**A non-interactive SSH command does not inherit the interactive PATH.** Ubuntu's
`~/.bashrc` returns early for non-interactive shells, and `sshd` runs with a
minimal environment. Anything installed per-user — `uv` in `~/.local/bin`,
`nvidia-smi` in `/usr/lib/wsl/lib` — must be exported at the **top** of
`~/.bashrc`, above that guard. Symptom: `uv: command not found` from `ssh`, while
the same command works in a terminal on the tower.

**`Connection reset by peer` at key exchange means WSL is asleep**, not a network
fault. The port redirection survives the virtual machine shutting down, so the
connection is accepted and then dropped. Open a terminal on the tower to wake it.
`remote.sh` names this cause first when the tower does not answer.

**Never `rm uv.lock`.** Doing so lost a working resolution and pulled in
`pyarrow` 26, which refuses numpy 1.x at import — an incompatibility that is not
expressed in package metadata, so the resolver cannot avoid it. Hence the
`pyarrow>=25.0.1,<26` ceiling, a direct consequence of `numpy<2`, itself a
consequence of the Intel Mac. Use `uv lock --upgrade-package <name>` to move one
package, and commit the lock with every change.

**Do not take torch from `download.pytorch.org/whl/cu126`.** uv reads that
wheel's metadata correctly but writes **none of its `nvidia-*` dependencies**
into the lock: torch installs and then fails at import with
`libcudart.so.12: cannot open shared object file`. Take `torch==2.6.0` from PyPI
instead (CUDA 12.4 build, Pascal included) — torch and its CUDA dependencies then
come from the same registry and the lock is complete.

**Do not add `[tool.uv] environments`.** Restricting platforms shrank the lock
from 454 KB to 189 KB and fixed nothing; it is an optimisation with no benefit
that obscures the real resolution.

**Windows 10 specifics.** WSL2 GPU support needs build **19044+** (1909 is not
enough, and Windows Update will not offer the upgrade to an out-of-service build
— use the Update Assistant). sshd is reached from the LAN through a persistent
`netsh portproxy` from 2222 to `127.0.0.1:22`, plus a logon task that starts WSL.
`systemd=true` in `/etc/wsl.conf` is what makes `systemctl enable ssh` persist.

**Two notebook 08 cells would compute locally.** The timing cell — the one
starting `with tempfile.TemporaryDirectory() as scratch:` — runs 20 fine-tune
steps plus a full inference pass to print a cost estimate: ~20 min of laptop CPU
for a number you already have in this file. Skip it. And
`run_finetunes(TEST_ORIGIN, …)` in section 4 would run a full 1000-step fine-tune
on the laptop (~12 h) and drop a CPU checkpoint into the GPU corpus; the
`finetune_test` stage produces those artifacts instead, which is why section 4
finds them cached.

---

## 7. Rebuilding the tower from scratch

1. Windows 10 **22H2 (build 19045)** or Windows 11; `winver` to check.
2. `wsl --install -d Ubuntu-22.04`, then `wsl -l -v` must show version **2**.
3. Update the **Windows** NVIDIA driver. Install nothing NVIDIA inside Ubuntu.
   `nvidia-smi` in Ubuntu must show the card.
4. `sudo apt install -y openssh-server tmux git rsync`.
5. `/etc/wsl.conf` with `[boot] systemd=true`, `wsl --shutdown`, then
   `sudo systemctl enable --now ssh`.
6. From PowerShell as admin, once:
   ```powershell
   netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=22 connectaddress=127.0.0.1
   New-NetFirewallRule -DisplayName "WSL SSH 2222" -Direction Inbound -LocalPort 2222 -Protocol TCP -Action Allow
   schtasks /create /tn "Demarrer WSL" /tr "wsl.exe -d Ubuntu-22.04 -- true" /sc onlogon /rl HIGHEST
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   ```
7. DHCP reservation for the tower on the router, so its address never moves.
8. On the laptop: `ssh-keygen -t ed25519`, `ssh-copy-id -p 2222`, a `Host gpu`
   block in `~/.ssh/config` (with `ServerAliveInterval 30` and `ControlMaster
   auto`), then `PasswordAuthentication no` on the tower.
9. At the **top** of `~/.bashrc` on the tower:
   `export PATH=$HOME/.local/bin:/usr/lib/wsl/lib:$PATH`.
10. `.wslconfig` in the Windows user folder: `memory=11GB`, `processors=12`,
    `swap=8GB`, `[experimental] autoMemoryReclaim=gradual`.
11. `uv` installed, repo cloned onto the **ext4** filesystem (`~/m5-chronos2`,
    never `/mnt/c/`: an order of magnitude of I/O on thousands of small files),
    `uv sync`, M5 data downloaded there directly with the Kaggle CLI into
    `data/raw/`.
12. `python scripts/benchmark_device.py` on both machines, to measure the factor
    rather than assume it.
