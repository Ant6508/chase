#!/usr/bin/env bash
# Lance, suit ou arrête un run de scripts/run_llm.py sur le pod, détaché de
# toute session SSH ou Claude Code (il survit à la déconnexion). Tout va dans
# /workspace/results/<nom>/ (journal.jsonl, table.md, run.log) sur le volume
# réseau. Relancer `start` avec le même nom et les mêmes arguments reprend aux
# seeds manquantes ; d'autres arguments sur le même journal sont refusés.
#
#   campaign.sh start  <nom> <arguments de run_llm.py>
#   campaign.sh status <nom>
#   campaign.sh stop   <nom>
set -euo pipefail

cmd=${1:?usage : campaign.sh start|status|stop <nom> [arguments de run_llm.py]}
name=${2:?nom du run manquant}
shift 2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DIR=/workspace/results/$name
PATTERN="journal $DIR/journal.jsonl"   # identifie le processus du run dans `ps`

case $cmd in
start)
    if pgrep -f "$PATTERN" >/dev/null; then
        echo "$name tourne déjà" >&2
        exit 1
    fi
    mkdir -p "$DIR"
    cd "$REPO"
    setsid nohup /workspace/venv/bin/python -u -m scripts.run_llm "$@" \
        --base-url http://127.0.0.1:1234/v1 \
        --journal "$DIR/journal.jsonl" --out "$DIR/table.md" \
        >> "$DIR/run.log" 2>&1 < /dev/null &
    echo "$name lancé ; suivi : bash $0 status $name"
    ;;
status)
    if pgrep -f "$PATTERN" >/dev/null; then echo "$name : en cours"; else echo "$name : arrêté"; fi
    echo "épisodes terminés : $(cat "$DIR/journal.jsonl" 2>/dev/null | grep -c '"seed"' || true)"
    tail -n 5 "$DIR/run.log" 2>/dev/null || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    ;;
stop)
    if pkill -f "$PATTERN"; then echo "$name arrêté"; else echo "$name ne tournait pas"; fi
    ;;
*)
    echo "commande inconnue : $cmd (start|status|stop)" >&2
    exit 2
    ;;
esac
