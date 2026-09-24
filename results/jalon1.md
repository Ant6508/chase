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

## Décision : jalon franchi, R1 figé à 52 %

Ces paramètres et les règles du README sont figés. Le jalon 1 est franchi,
avec un écart assumé par rapport à la fenêtre de la spec pour R1.

- **R2 > 85 %** : atteint. 92 % (IC 87–97 %).
- **Écart franc mais pas caricatural** : atteint. L'écart est de 40 points ;
  on est loin d'un 0 % contre 90 %.
- **R1 dans 30–40 %** : non atteint. 52 % (IC 42–62 %). Écart accepté, pour
  les raisons ci-dessous.

### Pourquoi 52 % est acceptable

1. **La fenêtre 30–40 % n'est qu'un moyen.** Ce que la porte de sortie doit
   garantir, c'est que se passer de l'information du coéquipier coûte cher
   (SPEC §1, condition 1), sans que l'écart soit caricatural. C'est le cas :
   les intervalles de confiance de R1 (42–62 %) et de R2 (87–97 %) sont
   disjoints, et le taux d'échec passe de 48 % à 8 %, soit six fois moins. On
   est loin du défaut qui a fait abandonner MOBA-lite, où la politique locale
   était déjà quasi optimale.
2. **Le plancher réel sera celui des LLM, pas celui de R1.** Un agent LLM sans
   communication (bras A1) fera très probablement moins bien qu'une
   heuristique scriptée. Durcir l'environnement pour ramener R1 à 35 %
   risquerait de faire tomber tous les bras LLM près de 0 %, et la courbe de
   Pareto n'aurait plus d'amplitude. À 52 %, il reste de la marge en dessous.
3. **Le risque inverse est pire.** Les règles sont figées après ce jalon
   (SPEC §6). Un environnement trop dur ne se rattrape plus ensuite ; un R1
   un peu haut ne gêne pas la mesure.

### Réserve sur la métrique principale

Le confinement à l'instant T sépare bien R1 et R2 (0,109 contre 0,012), mais
c'est surtout l'effet des captures, qui le mettent à 0. En moyenne sur
l'épisode, l'écart est bien plus faible (0,370 contre 0,274), parce que R1 et
R2 passent tous deux la première centaine de pas à chercher. Les jalons
suivants doivent garder la mesure à l'instant T, comme le prévoit la spec
(§2), et ne pas la remplacer par la moyenne sur l'épisode.

### Leviers essayés pour faire baisser R1

Aucune molette n'a permis d'amener R1 dans 30–40 % en gardant R2 au-dessus de
85 %. D'une configuration à l'autre, l'écart R1→R2 reste entre 30 et
40 points ; R1 et R2 montent ou descendent ensemble.

Molettes de la spec. Voici les réglages les plus proches de la fenêtre :

| Configuration (2 poursuivants sauf mention) | R1 | R2 |
|---|---|---|
| vision 5, `n_loops` 3, grille 27, T 250 (**retenue**) | 52 % | 92 % |
| vision 5, `n_loops` 3, grille 27, T 200 | 46 % | 84 % |
| vision 5, `n_loops` 2, grille 27, T 200 | 49 % | 79 % |
| vision 3, `n_loops` 4, grille 27, T 200 (60 épisodes) | 30 % | 65 % |
| 3 poursuivants, vision 5, `n_loops` 3, grille 27, T 200 (60 épisodes) | 63 % | 95 % |

Le réglage retenu est le seul qui franchit la condition sur R2, que la spec
tient pour la plus importante (« si R2 échoue, le labyrinthe est en cause »),
et c'est aussi lui qui donne l'écart le plus large.

Autres paramètres. On part du réglage retenu, sur 100 épisodes, et on ne
change qu'un paramètre à la fois :

| Variante | R1 | R2 | Écart |
|---|---|---|---|
| Réglage retenu | 52 % | 92 % | +40 |
| `target_memory=3` | 57 % | 87 % | +30 |
| `target_memory=12` | 52 % | 85 % | +33 |
| `target_cycle_bias=1.5` | 52 % | 85 % | +33 |
| `min_loop_len=20` | 45 % | 83 % | +38 |
| `min_loop_len=8`, `n_loops=5` | 52 % | 73 % | +21 |
| `min_spawn_dist=20` | 58 % | 95 % | +37 |
| Poursuivants partant de la même case | 27 % | 40 % | +13 |

Ces lignes se reproduisent avec
`python -m scripts.calibrate --policies R1,R2 --set <paramètre>=<valeur>`,
sauf la dernière, qui a été obtenue par un essai ponctuel non versionné.

- Aucun de ces paramètres ne fait baisser R1 sans faire baisser R2. Le cas
  `min_loop_len=20` (45 %) reste dans la marge d'erreur d'environ ±10 points.
- Faire partir les poursuivants de la même case pénalise bien R1 : sans
  communication, les deux poursuivants font le même travail. Mais R2
  s'effondre aussi, à cause de sa règle de répartition. Quand deux
  poursuivants sont sur la même case, elle attribue tout le travail au
  premier et le second le suit. Pour exploiter ce levier, il aurait fallu
  corriger R2, puis changer une règle très tard dans le réglage, au risque
  du reproche d'environnement « taillé sur mesure ». Levier écarté.
- Piste non explorée : renforcer R2 avec une vraie stratégie de balayage de
  graphe (un poursuivant garde un carrefour pendant que l'autre fouille une
  branche), puis durcir l'environnement. Avec une vision de 3, R1 est déjà à
  30 %, mais R2 n'est qu'à 65 %. Écartée : c'est du travail d'heuristique
  sans garantie de résultat, et l'écart actuel suffit (voir plus haut).

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
