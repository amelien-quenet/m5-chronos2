#!/usr/bin/env bash
#
# Lancer une etape sur la tour depuis le Mac, et rapatrier les resultats.
#
#   ./scripts/remote.sh check                      etat de la tour
#   ./scripts/remote.sh run --all                  lance, detache, rend la main
#   ./scripts/remote.sh run --stages finetune
#   ./scripts/remote.sh log                        suit le journal du run en cours
#   ./scripts/remote.sh status                     le run tourne-t-il encore ?
#   ./scripts/remote.sh stop                       arrete le run
#   ./scripts/remote.sh pull                       rapatrie artifacts/ -> Mac
#   ./scripts/remote.sh push                       envoie artifacts/ -> tour
#
# Le calcul tourne dans tmux cote tour : fermer le capot du Mac, perdre le Wi-Fi ou
# interrompre ce script ne l'arrete pas. `run` rend la main immediatement.
#
# Tout argument apres `run` est passe tel quel a scripts/run_stage.py.

set -euo pipefail

HOST="gpu"                 # alias de ~/.ssh/config
REMOTE_DIR="m5-chronos2"   # relatif au home de la tour
SESSION="m5"               # nom de la session tmux
LOG_DIR="artifacts/logs"

cd "$(dirname "$0")/.."

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# --- La tour repond-elle ? ----------------------------------------------------------
# C'est la panne recurrente du montage : la redirection de port survit a l'extinction
# de la machine virtuelle WSL, donc la connexion est acceptee puis coupee. Le message
# doit nommer la cause, pas renvoyer une erreur reseau.
check_alive() {
  if ! ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST" true 2>/dev/null; then
    die "La tour ne repond pas.

Dans l'ordre de probabilite :
  1. la machine virtuelle WSL est arretee -> ouvrir un terminal Ubuntu sur la tour
     (symptome typique : 'Connection reset by peer' a l'echange de cles) ;
  2. la session Windows n'est pas ouverte -> la tache qui demarre WSL se declenche
     a l'ouverture de session ;
  3. la tour est eteinte ou endormie, ou n'est pas sur le reseau local."
  fi
}

cmd_check() {
  check_alive
  say "tour joignable"
  ssh "$HOST" "cd $REMOTE_DIR && git log --oneline -1 && \
    uv run python -c \"
from m5_chronos2.chronos2 import device_report, get_pipeline, pipeline_dtype
import json; print(json.dumps(device_report(), default=str))
print('dtype', pipeline_dtype(get_pipeline()))\""
}

# --- Le code voyage par git ----------------------------------------------------------
sync_code() {
  local dirty
  dirty=$(git status --porcelain -- src scripts pyproject.toml uv.lock)
  if [[ -n "$dirty" ]]; then
    die "Modifications non committees dans src/, scripts/ ou le pyproject :

$dirty

La tour calculerait avec un code anterieur au tien, et l'artifact porterait le
mauvais commit dans son .meta.json. Committer d'abord."
  fi
  say "push du code"
  git push --quiet
  say "pull et sync sur la tour"
  ssh "$HOST" "cd $REMOTE_DIR && git pull --quiet && uv sync --quiet"
  local here there
  here=$(git rev-parse --short HEAD)
  there=$(ssh "$HOST" "cd $REMOTE_DIR && git rev-parse --short HEAD")
  [[ "$here" == "$there" ]] || die "commits differents : Mac $here, tour $there"
  say "meme commit des deux cotes : $here"
}

# --- Lancer -------------------------------------------------------------------------
cmd_run() {
  [[ $# -gt 0 ]] || die "donner les arguments de run_stage.py, par ex. --all"
  check_alive
  sync_code

  if ssh "$HOST" "tmux has-session -t $SESSION" 2>/dev/null; then
    die "une session '$SESSION' tourne deja sur la tour.
  ./scripts/remote.sh log     pour la suivre
  ./scripts/remote.sh stop    pour l'arreter"
  fi

  local stamp log
  stamp=$(date +%Y%m%d-%H%M%S)
  log="$LOG_DIR/${stamp}.log"

  say "lancement detache : run_stage.py $*"
  # tmux garde le shell en vie apres la fin de la commande, pour que la derniere
  # sortie reste lisible ; le journal, lui, est sur disque de toute facon.
  ssh "$HOST" "cd $REMOTE_DIR && mkdir -p $LOG_DIR && \
    tmux new-session -d -s $SESSION \
    \"uv run python scripts/run_stage.py $* 2>&1 | tee $log; echo; echo '--- termine ---'; sleep 86400\""

  sleep 3
  ssh "$HOST" "cd $REMOTE_DIR && tail -n 20 $log 2>/dev/null || echo '(journal pas encore ecrit)'"
  cat <<EOF

Le calcul tourne sur la tour, detache. Tu peux fermer ce terminal.

  ./scripts/remote.sh log       suivre le journal
  ./scripts/remote.sh status    toujours en cours ?
  ./scripts/remote.sh pull      rapatrier les artifacts quand c'est fini
EOF
}

cmd_log() {
  check_alive
  local log
  log=$(ssh "$HOST" "cd $REMOTE_DIR && ls -t $LOG_DIR/*.log 2>/dev/null | head -1")
  [[ -n "$log" ]] || die "aucun journal sur la tour"
  say "journal : $log   (Ctrl-C pour arreter de suivre, le calcul continue)"
  ssh -t "$HOST" "cd $REMOTE_DIR && tail -f $log"
}

cmd_status() {
  check_alive
  if ssh "$HOST" "tmux has-session -t $SESSION" 2>/dev/null; then
    say "run en cours"
    ssh "$HOST" "cd $REMOTE_DIR && tail -n 5 \$(ls -t $LOG_DIR/*.log | head -1)"
  else
    say "aucun run en cours"
    ssh "$HOST" "cd $REMOTE_DIR && ls -t $LOG_DIR/*.log 2>/dev/null | head -3" || true
  fi
}

cmd_stop() {
  check_alive
  ssh "$HOST" "tmux kill-session -t $SESSION" 2>/dev/null \
    && say "session arretee ; les artifacts deja ecrits sont conserves" \
    || say "aucune session a arreter"
}

# --- Les gros fichiers voyagent par rsync -------------------------------------------
# Les checkpoints Hugging Face intermediaires pesent des centaines de Mo pour rien ;
# seul leur trainer_state.json est utile (courbe de perte du notebook 08).
RSYNC_FILTERS=(
  -f '+ */'
  -f '+ **/trainer_state.json'
  -f '- finetuned/**/checkpoint-*/**'
  -f '+ *'
)

cmd_pull() {
  check_alive
  say "rapatriement artifacts/ depuis la tour"
  rsync -az --info=progress2 "${RSYNC_FILTERS[@]}" \
    "$HOST:$REMOTE_DIR/artifacts/" artifacts/
  say "fait. Les notebooks peuvent etre re-executes : chaque cached() trouvera son parquet."
}

cmd_push() {
  check_alive
  say "envoi artifacts/ vers la tour"
  rsync -az --info=progress2 "${RSYNC_FILTERS[@]}" \
    artifacts/ "$HOST:$REMOTE_DIR/artifacts/"
  say "fait."
}

case "${1:-}" in
  check)  shift; cmd_check "$@" ;;
  run)    shift; cmd_run "$@" ;;
  log)    shift; cmd_log "$@" ;;
  status) shift; cmd_status "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  pull)   shift; cmd_pull "$@" ;;
  push)   shift; cmd_push "$@" ;;
  *)      sed -n '3,20p' "$0"; exit 1 ;;
esac
