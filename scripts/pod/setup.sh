#!/usr/bin/env bash
# Prépare le pod RunPod pour une campagne LLM (jalon 2) : serveur LM Studio
# headless avec le modèle chargé, et environnement Python du projet.
#
# Idempotent, donc sûr à relancer après un stop/start du pod : LM Studio vit
# sur le disque du conteneur (qui repart à neuf), alors que le volume monté
# sur /workspace garde le modèle, le venv, le code et les résultats.
set -euo pipefail

MODEL=google/gemma-4-12b
MODEL_DIR=/workspace/models/lmstudio-community/gemma-4-12B-it-GGUF
MODEL_FILES="gemma-4-12B-it-Q6_K.gguf mmproj-gemma-4-12B-it-BF16.gguf"
HF=https://huggingface.co/lmstudio-community/gemma-4-12B-it-GGUF/resolve/main
IDENTIFIER=gemma-4-12b-a1   # = LLMConfig.model
# Contexte total partagé entre les créneaux. Par défaut 8 créneaux de 8192 tokens :
# un appel A2 demande ~2500 tokens de prompt plus 4000 de complétion. La campagne A1
# de septembre tournait avec 81920 tokens et 32 créneaux (2560 chacun, ~90 % d'une
# carte 24 Go : 4090, 3090) : CONTEXT=81920 PARALLEL=32 bash setup.sh le retrouve.
CONTEXT=${CONTEXT:-65536}
PARALLEL=${PARALLEL:-8}
VENV=/workspace/venv
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PATH="/root/.lmstudio/bin:$PATH"

if ! command -v lms >/dev/null; then
    curl -fsSL https://lmstudio.ai/install.sh | bash
fi
# le modèle (9,8 Go) est sur /workspace, pas sur le disque du conteneur
mkdir -p "$MODEL_DIR"
if [ ! -L /root/.lmstudio/models ]; then
    [ -d /root/.lmstudio/models ] && rmdir /root/.lmstudio/models  # échoue s'il n'est pas vide
    ln -s /workspace/models /root/.lmstudio/models
fi
# nouveau volume : téléchargement direct depuis HuggingFace, pas `lms get` (mesuré
# 10x plus lent sur un pod Community, et abandonné sur timeout à 30 %) ; `-C -`
# reprend un .part laissé par une coupure
for f in $MODEL_FILES; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        curl -fL --retry 5 -C - -o "$MODEL_DIR/$f.part" "$HF/$f"
        mv "$MODEL_DIR/$f.part" "$MODEL_DIR/$f"
    fi
done
lms daemon up
# métadonnées LM Studio de google/gemma-4-12b (réglages d'outil et de raisonnement,
# les mêmes qu'aux pilotes) : le GGUF étant déjà en place, `lms get` ne
# télécharge que ~64 Ko. Elles vivent sur le disque du conteneur, d'où le test.
if [ ! -f "/root/.lmstudio/hub/models/$MODEL/model.yaml" ]; then
    lms get "$MODEL@q6_k" --gguf -y < /dev/null
fi
# moteur CUDA explicite : quand la détection du GPU échoue, LM Studio choisit sinon le
# moteur CPU sans prévenir (vu le 2026-09-30 sur un pod Community défectueux)
cuda=$(lms runtime ls 2>/dev/null | grep -o 'llama.cpp-linux-x86_64-nvidia-cuda[^ ]*' | head -1)
[ -n "$cuda" ] && lms runtime select "$cuda"
# toujours recharger : un modèle déjà chargé avec un autre contexte ou d'autres créneaux
# ne serait pas remplacé (environ 10 s)
lms unload --all || true
lms load "$MODEL" --gpu max --context-length "$CONTEXT" --parallel "$PARALLEL" \
    --identifier "$IDENTIFIER" -y
# garde-fou : le modèle (~9 Go) doit être en mémoire sur le GPU, sinon il tourne sur le
# processeur et la campagne serait des dizaines de fois plus lente
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${used:-0}" -lt 5000 ]; then
    echo "le modèle n'est pas sur le GPU (${used:-?} MiB utilisés) : moteur CPU ou GPU inaccessible" >&2
    exit 1
fi
if ! curl -sf http://127.0.0.1:1234/v1/models >/dev/null; then
    setsid nohup lms server start --port 1234 --bind 0.0.0.0 --cors \
        > /workspace/server.log 2>&1 < /dev/null &
    for _ in $(seq 30); do curl -sf http://127.0.0.1:1234/v1/models >/dev/null && break; sleep 1; done
fi
if ! curl -sf http://127.0.0.1:1234/v1/models | grep -q "\"$IDENTIFIER\""; then
    echo "le serveur LM Studio ne sert pas $IDENTIFIER" >&2
    exit 1
fi

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
# tokenizer de gemma-4 préchargé : le premier épisode ne télécharge rien
"$VENV/bin/python" -c "import sys; sys.path.insert(0, '$REPO'); from chase.llm.message import load_tokenizer; load_tokenizer()"
echo "pod prêt : $IDENTIFIER sur 127.0.0.1:1234, Python dans $VENV"
