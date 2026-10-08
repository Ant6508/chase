# Jalon 2, bras A3 — plafond mécanique du message minimal

Étape 1 du § Validation de la spec (`docs/superpowers/specs/2026-10-08-jalon2-a3-design.md`),
jouée le 2026-10-08 sans LLM. Code du commit 898d262 (`chase/llm/ceiling.py`).

```bash
python -m scripts.ceiling_a2 --episodes 2000 --set max_steps=60 size=15 n_loops=1 min_loop_len=6     min_spawn_dist=6 --out results/jalon2_a3_plafond.md
python -m scripts.compare_arms --episodes 2000 --set max_steps=60 size=15 n_loops=1 min_loop_len=6     min_spawn_dist=6 --pairs P3:R1 P3:P2 P3:R2
python -m scripts.ceiling_a2 --episodes 30 --set max_steps=60 size=15 n_loops=1 min_loop_len=6     min_spawn_dist=6
```

## En bref

P3 prend les décisions de R2, mais ne sait de son coéquipier que ce que transporte le message A3 du
pas précédent : sa case (`moi`), la case de la cible s'il la voyait (`cible`), et `intention`, que
P3 n'exploite pas. Sans `cible`, il n'y a rien à fusionner, et chacun décide avec sa propre croyance.

- **P3 garde 55 % de l'écart de captures entre R1 et R2** (74 % de captures, contre 60 % et 85 %).
  P2, avec le message A2 complet, en gardait 92 %.
- **`candidates` vaut donc environ 9 points de captures** pour un décideur parfait : 83 % pour P2
  contre 74 % pour P3.
- **Le message minimal porte quand même de l'information exploitable.** P3 dépasse R1 de 14 points,
  par la seule position du coéquipier et la cible qu'il voit.
- **R1, R2 et P2 redonnent exactement les chiffres d'A2** (`results/jalon2_a2_plafond.md`) : le
  code de P2 n'a pas dérivé avec la spec de message par bras.

Ce n'est pas une porte (spec A3, § Validation) : A3 est défini par son budget, pas par l'optimalité
de son protocole.

## Comparaisons appariées sur 2000 seeds

McNemar exact, bilatéral, sur les mêmes 2000 seeds.

| Paire | Captures | Seul le premier capture | Seul le second capture | p |
|---|---|---|---|---|
| P3 contre R1 | 1472 / 1193 | 443 | 164 | < 10⁻²⁹ |
| P3 contre P2 | 1472 / 1658 | 21 | 207 | < 10⁻³⁸ |
| P3 contre R2 | 1472 / 1699 | 57 | 284 | < 10⁻³⁶ |

- **P3 contre P2 : la perte vient presque toute de `candidates`.** Sur 228 seeds discordantes, P3 n'en
  capture seul que 21.
- **P3 contre R2 :** R2 connaît la position exacte du coéquipier et fusionne les croyances entières ;
  P3 n'a que la position du pas précédent et la cible vue.

## Sur les seeds de campagne (0 à 29)

| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|---|
| R1 | 63 % | ± 17 % | 0,072 | 0,296 | 34 |
| R2 | 83 % | ± 13 % | 0,029 | 0,232 | 30 |
| P2 | 90 % | ± 11 % | 0,030 | 0,225 | 30 |
| P3 | 73 % | ± 16 % | 0,042 | 0,231 | 28 |

R1, R2 et P2 sont ceux du tableau de `results/jalon2_a2.md`.

## Ce que cela laisse attendre d'A3

- **P3 est une référence mécanique, pas un majorant strict d'A3.** Il ignore `intention`, qu'un LLM
  peut exploiter.
- **Les bras LLM restent loin de leur plafond mécanique.** A2 capturait 18 seeds sur 30 contre 27 pour
  P2 ; le goulot est la phase de capture, pas le canal (`results/jalon2_a2p.md`, § 5.3).
- **A3 peut donc égaler A2 en captures** si le LLM n'exploitait pas `candidates`, ce qui était le cas
  sans consigne d'usage (`results/jalon2_a2.md`, § 6 (a)).

## Sortie brute de `ceiling_a2` sur 2000 seeds

Seeds 0..1999 (2000 épisodes), T = 60.

| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|---|
| R1 | 60% | ± 2% | 0.056 | 0.300 | 35 |
| R2 | 85% | ± 2% | 0.015 | 0.238 | 33 |
| P2 | 83% | ± 2% | 0.026 | 0.251 | 34 |
| P3 | 74% | ± 2% | 0.033 | 0.256 | 33 |

Écart de captures R1 -> R2 : +25%. P2 en garde 92%. Porte (au moins 50 %) : **franchie**.

P3 (message A3 : moi, cible, intention) en garde 55%, sans porte.

Paramètres :

```json
{
  "size": 15,
  "n_loops": 1,
  "min_loop_len": 6,
  "n_pursuers": 2,
  "view_size": 5,
  "min_spawn_dist": 6,
  "max_steps": 60,
  "target_memory": 6,
  "target_cycle_bias": 0.5,
  "comm_delay": 1,
  "track_threshold": 8
}
```
