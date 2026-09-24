# Chase

Environnement de poursuite coopérative en observabilité partielle. La
spécification figée est dans [`SPEC.md`](SPEC.md). Ce dépôt contient le
**jalon 1** : simulateur et heuristiques scriptées R0/R1/R2, sans aucun LLM.
Le jalon est franchi, avec R1 figé à 52 % de capture et R2 à 92 %.
Les résultats, les paramètres figés et la justification de l'écart à la
fenêtre 30–40 % visée pour R1 sont dans [`results/jalon1.md`](results/jalon1.md).

## Installation et exécution

```bash
pip install -r requirements.txt
python -m pytest -q
python -m scripts.calibrate --episodes 100 --out results/jalon1.md
python -m scripts.calibrate --episodes 30 --set view_size=3 n_loops=4   # surcharges
```

## Organisation

| Fichier | Rôle |
|---|---|
| `chase/config.py` | Paramètres du jeu ; les valeurs par défaut sont celles retenues au jalon 1 |
| `chase/maze.py` | Labyrinthe : arbre couvrant (backtracker) + `n_loops` cycles d'au moins `min_loop_len` cases |
| `chase/env.py` | `ChaseEnv`, sous-classe de `MultiGridEnv` : carte seedée, déplacements, capture, visibilité, croyance d'équipe, confinement |
| `chase/belief.py` | Ensemble candidat (propagation / observation) et carte de probabilité associée |
| `chase/graph.py` | Distances de chemin, cases sur cycle (2-cœur du graphe) |
| `chase/target.py` | Politique scriptée de la cible |
| `chase/policies.py` | R0, R1, R2 |
| `chase/runner.py` | Boucle d'épisode et agrégation |
| `scripts/calibrate.py` | Tableau du jalon 1 |

## Décisions prises au jalon 1

Ces points étaient ouverts ou ambigus dans la spec. Ils sont maintenant tranchés
et figés comme le reste des règles.

1. **Capture.** La cible est prise quand un poursuivant finit le pas sur sa
   case, ou quand un poursuivant et la cible échangent leurs cases pendant le
   même pas (croisement dans un couloir). La formulation « toutes ses cases de
   sortie occupées ou observées » de la spec n'a pas été retenue : une case
   observée n'empêche pas la cible d'y passer.
2. **Confinement.** Il est calculé par l'environnement, avec la même règle
   pour tous les bras : c'est la taille de l'ensemble candidat d'équipe
   (fusion exacte et sans délai de toutes les observations des poursuivants)
   à l'instant `T = max_steps`, divisée par le nombre de cases libres. Il vaut
   0 après capture. On rapporte aussi le confinement moyen sur l'épisode.
3. **Fusion R2.** Les poursuivants partagent une croyance unique, mise à jour
   à chaque pas avec les observations de tous, avec **un pas de délai** comme
   le canal des bras LLM. Chacun raisonne sur la croyance partagée du pas
   `t-1`, propagée puis intersectée avec sa propre observation à `t`, et
   connaît la position de ses coéquipiers à `t-1`. `comm_delay=0` donne la
   variante sans délai. Intersecter deux croyances propagées séparément
   aurait été moins précis que cette fusion.
4. **Vision.**
   - Les poursuivants gardent le champ **orienté** de MiniGrid
     (`agent_view_size`, occultation par les murs).
   - Chaque action est un déplacement absolu nommé (`NORTH`, `SOUTH`, `EAST`,
     `WEST`, `STAY`) : l'agent s'oriente puis avance d'une case. Tous les
     agents vont donc à 1 case par pas. Contre un mur, l'agent se contente de
     s'orienter.
   - La cible voit à **360°** : c'est l'union des quatre orientations du
     masque MiniGrid, avec le même rayon et la même occultation. Avec un champ
     orienté, elle tournait le dos à son poursuivant en fuyant, l'oubliait et
     se faisait prendre sans qu'aucune coordination soit nécessaire. R1
     égalait alors R2.
5. **Politique de la cible** (`chase/target.py`, figée) :
   - Si un poursuivant est visible, ou a été vu dans les `target_memory`
     derniers pas, la cible court vers la case de cycle la plus proche qu'elle
     atteint strictement avant tout poursuivant connu. Si aucune case de cycle
     ne le permet, ou si elle est déjà sur un cycle, elle fuit de façon
     gloutonne pour maximiser sa distance au poursuivant le plus proche.
   - Sinon, elle marche au hasard sans faire demi-tour, avec un biais
     `exp(-bias · distance_au_cycle)` vers les cycles.

   Sans le refuge sur cycle, la fuite gloutonne mène la cible dans un
   cul-de-sac et un seul poursuivant suffit à la prendre. La topologie ne
   joue alors plus aucun rôle.

## Heuristiques des poursuivants

R1 et R2 exécutent le même code de décision. Deux choses seulement les
distinguent : d'où vient la croyance (propre à chacun, ou fusionnée), et si le
poursuivant connaît la position de son coéquipier.

- **Exploration** (ensemble candidat de plus de `track_threshold` cases) : on
  pose sur l'ensemble candidat une carte de probabilité, qui suit une marche
  aléatoire de la cible et a exactement le même support. On vise la case qui
  maximise « masse à moins de 3 cases / (distance + 4) », avec une hystérésis
  sur l'objectif. En R2, chaque poursuivant ne vise que les cases plus proches
  de lui que de ses coéquipiers (partition de Voronoï).
  Première version : viser la case candidate la plus proche. Elle échouait,
  parce que les cases vidées redeviennent candidates presque aussitôt et que
  les poursuivants piétinaient.
- **Poursuite** (cible localisée) : chaque poursuivant joue le pas qui réduit
  le plus le territoire de la cible, c'est-à-dire les cases qu'elle atteint
  strictement avant tout poursuivant connu. Seul, il la serre ; à deux, ils la
  prennent en tenaille sur un cycle.

## Invariant vérifié

`tests/test_chase.py` vérifie à chaque pas, sur plusieurs seeds, que la vraie
position de la cible appartient à toutes les croyances : croyance individuelle
R1, croyance R2 avec et sans délai, croyance d'équipe. Les tests vérifient
aussi que le labyrinthe a exactement `n_loops` cycles indépendants et que la
carte ne dépend que de la seed.
