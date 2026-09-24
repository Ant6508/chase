# Jalon 1 — calibration sans LLM

Commande : `python -m scripts.calibrate --episodes 100`.
Seeds 0 à 99, soit 100 épisodes ; T = 250. Carte, placement initial et flux
aléatoires ne dépendent que de la seed : ils sont identiques pour les trois
politiques.

| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|---|
| R0 — aléatoire | 0 % | ± 0 % | 0.737 | 0.778 | — |
| R1 — gloutonne locale, sans communication | 52 % | ± 10 % | 0.109 | 0.370 | 118 |
| R2 — croyance fusionnée, délai 1 pas | 92 % | ± 5 % | 0.012 | 0.274 | 114 |
| R2 — croyance fusionnée, sans délai (`comm_delay=0`) | 90 % | ± 6 % | 0.012 | 0.276 | 108 |

- Confinement : taille de l'ensemble candidat d'équipe divisée par le nombre de
  cases libres, 0 après capture. Plus bas = mieux.
- Confinement moyen : moyenne sur les pas 0 à T.
- IC 95 % : intervalle de confiance d'une proportion (approximation normale).

## Paramètres retenus (figés)

```json
{
  "size": 27,
  "n_loops": 3,
  "min_loop_len": 12,
  "n_pursuers": 2,
  "view_size": 5,
  "min_spawn_dist": 12,
  "max_steps": 250,
  "target_memory": 6,
  "target_cycle_bias": 0.5,
  "comm_delay": 1,
  "track_threshold": 8
}
```

Valeurs par défaut de `chase/config.py`. MultiGrid est épinglé au commit
`6f46215` dans `requirements.txt`.

## Porte de sortie : où on en est

- **R2 > 85 %** : atteint. 92 % (IC 87–97 %).
- **Écart franc mais pas caricatural** : atteint. L'écart est de 40 points ;
  on est loin d'un 0 % contre 90 %.
- **R1 dans 30–40 %** : non atteint. 52 % (IC 42–62 %).

Aucune molette n'a permis d'amener R1 dans 30–40 % en gardant R2 au-dessus de
85 %. D'une configuration à l'autre, l'écart R1→R2 reste entre 30 et
40 points ; R1 et R2 montent ou descendent ensemble. Les réglages les plus
proches de la fenêtre :

| Configuration (2 poursuivants sauf mention) | R1 | R2 |
|---|---|---|
| vision 5, `n_loops` 3, grille 27, T 250 (**retenue**) | 52 % | 92 % |
| vision 5, `n_loops` 3, grille 27, T 200 | 46 % | 84 % |
| vision 5, `n_loops` 2, grille 27, T 200 | 49 % | 79 % |
| vision 3, `n_loops` 4, grille 27, T 200 (60 épisodes) | 30 % | 65 % |
| 3 poursuivants, vision 5, `n_loops` 3, grille 27, T 200 (60 épisodes) | 63 % | 95 % |

J'ai retenu le premier réglage : il franchit la condition sur R2, que la spec
tient pour la plus importante (« si R2 échoue, le labyrinthe est en cause »),
et c'est lui qui donne l'écart le plus large. À décider : accepter R1 à 52 %
ou continuer à chercher.

## Historique du réglage

Ces étapes ont modifié les règles. Le détail est dans le README, section
« Décisions prises au jalon 1 ».

1. Première version : cible qui voit dans une seule direction, exploration
   vers la case candidate la plus proche. R1 43 %, R2 57 %. R2 échouait parce
   qu'il ne trouvait jamais la cible : les cases vidées redevenaient
   candidates presque aussitôt.
2. Exploration guidée par une carte de probabilité posée sur l'ensemble
   candidat. R1 90 %, R2 100 %. R1 était trop fort, à cause de la cible.
3. Refuge de la cible sur les cycles. Peu d'effet seul : la cible, qui ne
   voyait que devant elle, perdait de vue le poursuivant qui la suivait.
4. Cible à 360° (même rayon, même occultation). L'écart apparaît :
   environ 55 % contre 90 %.
5. Poursuite à plusieurs par minimisation du territoire de la cible, pour la
   tenaille. Appliquée à R1, cette règle le faisait tomber à environ 25 %,
   sous la poursuite directe. R1 a donc gardé la poursuite directe, pour ne
   pas l'affaiblir artificiellement.
6. Balayage des molettes (vision, `n_loops`, grille, T, 3 poursuivants)
   jusqu'au réglage ci-dessus.
