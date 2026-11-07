#!/usr/bin/env bash
#
# Run a stage on the GPU tower from the Mac, and bring the results back.
#
#   ./scripts/remote.sh check                      state of the tower
#   ./scripts/remote.sh run --all                  launch, detach, return the prompt
#   ./scripts/remote.sh run --stages finetune
#   ./scripts/remote.sh log                        follow the running job's log
#   ./scripts/remote.sh status                     is the job still running?
#   ./scripts/remote.sh stop                       stop the job
#   ./scripts/remote.sh pull                       bring artifacts/ back to the Mac
#   ./scripts/remote.sh push                       send artifacts/ to the tower
#
# The computation runs inside tmux on the tower: closing the Mac's lid, losing Wi-Fi or
# interrupting this script does not stop it. `run` returns the prompt immediately.
#
# Any argument after `run` is passed through to scripts/run_stage.py unchanged.

set -euo pipefail

HOST="gpu"                 # alias from ~/.ssh/config
REMOTE_DIR="m5-chronos2"   # relative to the tower's home directory
SESSION="m5"               # tmux session name
LOG_DIR="artifacts/logs"

cd "$(dirname "$0")/.."

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# --- Is the tower answering? --------------------------------------------------------
# This is the setup's recurring failure: the port forward outlives the WSL virtual
# machine, so the connection is accepted and then dropped. The message has to name the
# cause rather than hand back a network error.
check_alive() {
  if ! ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST" true 2>/dev/null; then
    die "The tower is not answering.

In order of likelihood:
  1. the WSL virtual machine is stopped -> open an Ubuntu terminal on the tower
     (typical symptom: 'Connection reset by peer' during the key exchange);
  2. nobody is logged in to Windows -> the task that starts WSL is triggered by
     the logon;
  3. the tower is off, asleep, or not on the local network."
  fi
}

cmd_check() {
  check_alive
  say "tower reachable"
  ssh "$HOST" "cd $REMOTE_DIR && git log --oneline -1 && \
    uv run python -c \"
from m5_chronos2.chronos2 import device_report, get_pipeline, pipeline_dtype
import json; print(json.dumps(device_report(), default=str))
print('dtype', pipeline_dtype(get_pipeline()))\""
}

# --- Code travels by git -------------------------------------------------------------
sync_code() {
  local dirty
  dirty=$(git status --porcelain -- src scripts pyproject.toml uv.lock)
  if [[ -n "$dirty" ]]; then
    die "Uncommitted changes in src/, scripts/ or the pyproject:

$dirty

The tower would compute with code older than yours, and the artifact would carry the
wrong commit in its .meta.json. Commit first."
  fi
  say "pushing code"
  git push --quiet
  say "pull and sync on the tower"
  ssh "$HOST" "cd $REMOTE_DIR && git pull --quiet && uv sync --quiet"
  local here there
  here=$(git rev-parse --short HEAD)
  there=$(ssh "$HOST" "cd $REMOTE_DIR && git rev-parse --short HEAD")
  [[ "$here" == "$there" ]] || die "different commits: Mac $here, tower $there"
  say "same commit on both sides: $here"
}

# --- Launch -------------------------------------------------------------------------
cmd_run() {
  [[ $# -gt 0 ]] || die "pass run_stage.py's arguments, for example --all"
  check_alive
  sync_code

  if ssh "$HOST" "tmux has-session -t $SESSION" 2>/dev/null; then
    die "a '$SESSION' session is already running on the tower.
  ./scripts/remote.sh log     to follow it
  ./scripts/remote.sh stop    to stop it"
  fi

  local stamp log
  stamp=$(date +%Y%m%d-%H%M%S)
  log="$LOG_DIR/${stamp}.log"

  say "detached launch: run_stage.py $*"
  # The tmux session ends with the command, so its existence is a reliable signal for
  # "job running". The output itself stays in the log.
  ssh "$HOST" "cd $REMOTE_DIR && mkdir -p $LOG_DIR && \
    tmux new-session -d -s $SESSION \
    \"uv run python scripts/run_stage.py $* 2>&1 | tee $log; echo '--- finished ---' >> $log\""

  sleep 3
  ssh "$HOST" "cd $REMOTE_DIR && tail -n 20 $log 2>/dev/null || echo '(log not written yet)'"
  cat <<EOF

The computation is running on the tower, detached. This terminal can be closed.

  ./scripts/remote.sh log       follow the log
  ./scripts/remote.sh status    still running?
  ./scripts/remote.sh pull      bring the artifacts back when it is done
EOF
}

cmd_log() {
  check_alive
  local log
  log=$(ssh "$HOST" "cd $REMOTE_DIR && ls -t $LOG_DIR/*.log 2>/dev/null | head -1")
  [[ -n "$log" ]] || die "no log on the tower"
  say "log: $log   (Ctrl-C stops following; the computation continues)"
  ssh -t "$HOST" "cd $REMOTE_DIR && tail -f $log"
}

cmd_status() {
  check_alive
  local log
  log=$(ssh "$HOST" "cd $REMOTE_DIR && ls -t $LOG_DIR/*.log 2>/dev/null | head -1")
  if [[ -z "$log" ]]; then say "no log on the tower"; return; fi

  if ssh "$HOST" "cd $REMOTE_DIR && grep -q -- '--- finished ---' '$log'"; then
    say "FINISHED   $log"
  elif ssh "$HOST" "tmux has-session -t $SESSION" 2>/dev/null; then
    say "RUNNING    $log"
  else
    # Neither an end marker nor a session: the process died before finishing.
    say "INTERRUPTED (no end marker, no tmux session)   $log"
  fi
  ssh "$HOST" "cd $REMOTE_DIR && tail -n 8 '$log'"
}

cmd_stop() {
  check_alive
  ssh "$HOST" "tmux kill-session -t $SESSION" 2>/dev/null \
    && say "session stopped; the artifacts already written are kept" \
    || say "no session to stop"
}

# --- Large files travel by rsync ----------------------------------------------------
# macOS ships rsync 2.6.9 (2006): no --info=progress2, no -h, and filter patterns with
# ** are unreliable there. So this sticks to --progress and plain --exclude patterns,
# which also work with an rsync 3.x.
#
# Intermediate Hugging Face checkpoints weigh hundreds of megabytes (weights, optimiser
# state, random-generator states) for a single useful piece of information:
# trainer_state.json, which notebook 08 reads the loss curve from. So the large files
# INSIDE each checkpoint-* are excluded, not the directory itself.
RSYNC_OPTS=(
  -az --progress
  "--exclude=checkpoint-*/*.safetensors"
  "--exclude=checkpoint-*/*.bin"
  "--exclude=checkpoint-*/*.pt"
  "--exclude=checkpoint-*/*.pth"
  "--exclude=checkpoint-*/rng_state*"
)

cmd_pull() {
  check_alive
  say "pulling artifacts/ from the tower"
  rsync "${RSYNC_OPTS[@]}" "$HOST:$REMOTE_DIR/artifacts/" artifacts/
  say "done. The notebooks can be re-run: every cached() call will find its parquet."
}

cmd_push() {
  check_alive
  say "sending artifacts/ to the tower"
  rsync "${RSYNC_OPTS[@]}" artifacts/ "$HOST:$REMOTE_DIR/artifacts/"
  say "done."
}


case "${1:-}" in
  check)  shift; cmd_check "$@" ;;
  run)    shift; cmd_run "$@" ;;
  log)    shift; cmd_log "$@" ;;
  status) shift; cmd_status "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  pull)   shift; cmd_pull "$@" ;;
  push)   shift; cmd_push "$@" ;;
  *)      sed -n '3,18p' "$0"; exit 1 ;;
esac
