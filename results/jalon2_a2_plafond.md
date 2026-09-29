# Jalon 2, bras A2 — plafond mécanique du protocole

Porte du § Validation, étape 1, de la spec (`docs/superpowers/specs/2026-09-29-jalon2-a2-design.md`),
jouée le 2026-09-29 sans LLM. Code du commit 1a7cc5b (`chase/llm/ceiling.py`).

```bash
python -m scripts.ceiling_a2 --episodes 2000 --set max_steps=60 size=15 n_loops=1 min_loop_len=6     min_spawn_dist=6 --out results/jalon2_a2_plafond.md
python -m scripts.compare_arms --episodes 2000 --set max_steps=60 size=15 n_loops=1 min_loop_len=6     min_spawn_dist=6 --pairs P2:R2 R1:P2 R1:R2
```

## Verdict : porte franchie

P2 prend les décisions de R2, mais ne sait de son coéquipier que ce que transporte le message A2 du
pas précédent, au grain des lieux. Il garde **92 %** de l'écart de captures entre R1 et R2. Le
protocole transporte donc l'essentiel de l'information utile. Il n'y a pas lieu de revoir le grain
des lieux avant les appels au LLM.

Comparaisons appariées sur les mêmes 2000 seeds (McNemar exact, bilatéral) :

| Paire | Captures | Seul le premier capture | Seul le second capture | p |
|---|---|---|---|---|
| P2 contre R2 | 1658 / 1699 | 72 | 113 | 0,003 |
| R1 contre P2 | 1193 / 1658 | 87 | 552 | < 10⁻⁸⁰ |
| R1 contre R2 | 1193 / 1699 | 83 | 589 | < 10⁻⁹⁰ |

- **P2 reste significativement sous R2, de 2 points seulement.** D'après la relecture de la tâche 10,
  qui a fait varier P2 sur ces mêmes 2000 seeds, la perte se répartit ainsi :
  - environ 0,8 point pour le grain de 5 cases ;
  - environ 0,7 point pour l'absence de message au premier pas (R2 connaît dès le départ la position
    du coéquipier) ;
  - environ 0,4 point pour l'absence d'accumulation, P2 étant sans mémoire comme un LLM.

  Des tronçons de 3 cases n'y changent rien : 82,8 %.
- **P2 est une référence mécanique, pas un majorant strict d'A2.** Il ignore `intention` et les
  pourcentages reçus, qu'un LLM peut exploiter. Ce qu'on peut attendre d'A2, c'est au mieux environ le
  taux de P2, à condition que le LLM lise le message aussi bien que P2.
- **Sur 200 seeds, cet écart se serait perdu dans le bruit.** Sur les seeds 0 à 199, P2 dépassait même
  R2 (87,5 % contre 86,5 %). D'où les 2000 seeds (plan, révision de la tâche 14).

## Tableau

Seeds 0..1999 (2000 épisodes), T = 60.

| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|---|
| R1 | 60% | ± 2% | 0.056 | 0.300 | 35 |
| R2 | 85% | ± 2% | 0.015 | 0.238 | 33 |
| P2 | 83% | ± 2% | 0.026 | 0.251 | 34 |

Écart de captures R1 -> R2 : +25%. P2 en garde 92%. Porte (au moins 50 %) : **franchie**.

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
