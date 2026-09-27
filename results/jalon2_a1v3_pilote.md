# Jalon 2, bras A1 — pilote local de la perception à probabilité

Pilote du 2026-09-27 (13h37 → 15h17, heure de Paris), code `3145e07`, sur le
LM Studio local (RX 6750 XT, `google/gemma-4-12b` Q6_K, même fichier que les
campagnes RunPod, chargé avec `--context-length 10240 --parallel 4`). Mêmes
réglages de génération (`temperature=0`, `max_tokens=1600`, 2 retries), délai
porté à 400 s : à 4 parties en parallèle, ce GPU génère ~7 tokens/s par partie.
Perception du commit `0491505` : part de probabilité par issue.

```bash
python -m scripts.run_llm --first-seed 0 --episodes 4 --concurrency 4 \
    --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --base-url http://127.0.0.1:1234/v1 --timeout 400 \
    --journal results/jalon2_a1v3_local/journal.jsonl \
    --trace results/jalon2_a1v3_local/trace --out results/jalon2_a1v3_local/table.md
```

Seuls 4 épisodes : le pilote dit si le correctif agit, pas le taux de capture.
Le journal est prévu pour être étendu à 30 seeds avec la même commande
(`--episodes 30`), qui ne rejoue que les seeds manquantes.

## Résultat

| Seed | a1v2 (sans probabilité) | Pilote | R1 | R2 |
|---|---|---|---|---|
| 0 | non (0,010) | **capture au pas 22** | capture au pas 35 | capture au pas 22 |
| 1 | capture au pas 32 | **capture au pas 19** | non (0,010) | capture au pas 27 |
| 2 | non (0,816) | **capture au pas 28** | capture au pas 30 | capture au pas 30 |
| 3 | non (0,449) | **capture au pas 58** | capture au pas 39 | capture au pas 39 |
| **Captures** | 1 / 4 | **4 / 4** | 3 / 4 | 4 / 4 |

Entre parenthèses : confinement à T quand la cible n'est pas prise.

Traces (`python -m scripts.analyze_trace results/jalon2_a1v3_local/trace --set
max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6`), comparées à
celles de a1v2 sur les mêmes seeds :

| | a1v2, seeds 0–3 | Pilote |
|---|---|---|
| Allers-retours | 49 % | **26 %** |
| Cible cachée : vers l'issue la plus probable | — | 84 % |
| Cible cachée : vers la candidate la plus proche | 98 % | 60 % |
| Cible visible : coup qui la rapproche | 100 % | 100 % |
| Immobile, contre un mur, replis | 0 | 0 |
| Tokens de complétion par décision | 413 | 428 |

- Seed 2, où les deux poursuivants oscillaient entre 2 cases pendant tout
  l'épisode dans a1v2 : 4 % d'allers-retours, capture au pas 28.
- Seed 3 : encore 47 % d'allers-retours, capture tardive au pas 58.
- Réserve : même fichier de modèle que sur RunPod, mais autre matériel et
  autre moteur (Vulkan/AMD contre CUDA) ; à température 0, les réponses peuvent
  différer à la marge.
