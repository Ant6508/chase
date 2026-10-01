# Jalon 2, bras A2 — pilote sur RunPod, seeds 0 à 7

**Quand.** Du 2026-09-30, 19h40 UTC, au 2026-10-01, vers 05h00 UTC.

**Où.**
- Pod `otdj10kjtl7a4g` : A40 48 Go, Secure, CA-MTL-1.
- LM Studio avec le moteur CUDA, 8 créneaux de 8 192 tokens.
- Code `540dd1d`.

```bash
campaign.sh start a2 --arm A2 --max-tokens 5600 --timeout 900 --first-seed 0 --episodes 8 \
    --concurrency 8 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --trace /workspace/results/a2/trace
```

**Fichiers.** [`jalon2_a2_pilote/`](jalon2_a2_pilote/) contient `journal.jsonl`, `run.log` et
`trace/`. La campagne repart de ce même journal pour les seeds 8 à 29.

## Résultats par seed

| Seed | A2 | A1v3 | R1 | R2 | Replis A2 | Positions par message (moyenne) |
|---|---|---|---|---|---|---|
| 0 | capture au pas 24 | capture au pas 22 | capture au pas 35 | capture au pas 22 | 1 | 142 |
| 1 | capture au pas 21 | capture au pas 19 | non | capture au pas 27 | 1 | 135 |
| 2 | capture au pas 27 | capture au pas 28 | capture au pas 30 | capture au pas 30 | 0 | 107 |
| 3 | capture au pas 47 | capture au pas 58 | capture au pas 39 | capture au pas 39 | 3 | 144 |
| 4 | capture au pas 31 | capture au pas 33 | capture au pas 33 | capture au pas 33 | 1 | 113 |
| 5 | non | capture au pas 25 | capture au pas 27 | capture au pas 35 | 0 | 80 |
| 6 | non | non | non | capture au pas 56 | 1 | 150 |
| 7 | non | capture au pas 46 | capture au pas 55 | capture au pas 22 | 12 | 179 |
| **Captures** | **5 / 8** | 7 / 8 | 6 / 8 | 8 / 8 | 19 | 132 |

Ces 8 seeds ne permettent aucune conclusion. Le témoin est A1bis, à jouer sur le même pod :
A1v3 tournait sur une autre pile (Vulkan, carte AMD) et sans lieux nommés.

## Ce que montrent les traces

| | A1v3 (30 seeds) | A2, pilote (8 seeds) |
|---|---|---|
| Décisions | 2 778 | 660 |
| Replis | 2 | 19 (2,9 %), dont 12 à la seed 7 |
| Relances | — | 142 ; 156 tentatives coupées à 5 600 tokens |
| Cible visible : coup qui la rapproche | 100 % | 100 % |
| Cible cachée : vers l'issue la plus probable | 79 % | 78 % |
| Allers-retours | 36 % | 18 % |
| Tokens de prompt par décision | 596 | 1 686 |
| Tokens de complétion par décision | 446 | 2 569, dont 2 539 de pensée en moyenne |
| Positions par message | — | moyenne 132, médiane 162 (32 à 222) |
| Noms inconnus | — | 12 sur environ 640 messages |

- **Les critères du pilote sont remplis.**
  - Moins de 5 % de replis.
  - Des noms inconnus marginaux.
  - La seule erreur qui revient est la coupure à 5 600 tokens, presque toujours rattrapée par la
    relance.
- **Les allers-retours sont deux fois moins fréquents qu'en A1v3.** C'est peut-être un effet des
  lieux nommés plutôt que du canal : A1bis tranchera.
- **La seed 7 est un cas difficile.** La cible n'y est jamais vue, l'épisode a duré 8 h 40, et il
  concentre 12 replis : sa pensée s'emballe au-delà du plafond, même après relance.

## Coût

L'épisode médian a pris environ 4 h, 8 épisodes tournant en parallèle. Les 8 seeds ont demandé
environ 7 h 30 de pod. Sur la fin, peu d'épisodes restaient en cours, et une partie du débit était
perdue.
