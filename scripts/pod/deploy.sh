#!/usr/bin/env bash
# Depuis la machine locale (Git Bash) : envoie le code de travail (fichiers
# suivis + nouveaux, hors .gitignore) dans /workspace/chase_mas sur le pod,
# puis y exécute setup.sh. Les résultats (/workspace/results) ne sont pas
# touchés. IP et port SSH : `ssh.direct` de get-pod (ils changent à chaque
# démarrage du pod).
#
#   scripts/pod/deploy.sh <ip> <port>
#   ssh -p <port> root@<ip> bash /workspace/chase_mas/scripts/pod/campaign.sh start a1 \
#       --episodes 30 --concurrency 8 --set max_steps=60 size=15
#   ssh -p <port> root@<ip> bash /workspace/chase_mas/scripts/pod/campaign.sh status a1
#   ssh -p <port> root@<ip> bash /workspace/chase_mas/scripts/pod/campaign.sh start a2 \
#       --arm A2 --max-tokens 4000 --timeout 900 --episodes 30 --concurrency 8 \
#       --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
#       --trace /workspace/results/a2/trace
#   scp -P <port> root@<ip>:/workspace/results/a1/{table.md,journal.jsonl} results/
set -euo pipefail

host=${1:?ip du pod}
port=${2:?port SSH du pod}
cd "$(git rev-parse --show-toplevel)"
revision=$(git rev-parse --short HEAD)
[ -z "$(git status --porcelain)" ] || revision="$revision+modifs-locales"

ssh_pod() { ssh -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" "$@"; }
git ls-files -co --exclude-standard -z | tar --null -T - -czf - | ssh_pod \
    'rm -rf /workspace/chase_mas && mkdir -p /workspace/chase_mas && tar --no-same-owner -xzf - -C /workspace/chase_mas'
ssh_pod "echo '$revision' > /workspace/chase_mas/REVISION && bash /workspace/chase_mas/scripts/pod/setup.sh"
echo "code $revision déployé sur $host:$port"
