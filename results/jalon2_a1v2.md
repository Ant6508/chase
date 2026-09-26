# Jalon 2, bras A1 — deuxième campagne (perception corrigée)

Campagne du 2026-09-26 (17h22 → 21h47 UTC, 4 h 25), code `3e7b30d`, sur un
pod RunPod RTX PRO 4000 Blackwell (Secure, EU-RO-1). Même modèle et mêmes
réglages que la première campagne ([`jalon2_a1.md`](jalon2_a1.md)) :
`google/gemma-4-12b` Q6_K, `temperature=0`, `max_tokens=1600`,
`timeout_s=180`, 2 retries. Même profil réduit, mêmes seeds. Seule la
perception a changé (commit `51af56a`) : cible en mots cardinaux avec sa
distance de chemin, cases candidates comptées par issue au plus court.

```bash
bash scripts/pod/campaign.sh start a1v2 --episodes 30 --concurrency 8 \
    --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --trace /workspace/results/a1v2/trace
```

Journal : [`jalon2_a1v2_journal.jsonl`](jalon2_a1v2_journal.jsonl) ; sortie
brute : [`jalon2_a1v2_run.log`](jalon2_a1v2_run.log) ; traces pas à pas :
[`jalon2_a1v2_trace/`](jalon2_a1v2_trace/), analysées par
`python -m scripts.analyze_trace results/jalon2_a1v2_trace --set max_steps=60
size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6`.

## Résultat : A1 entre le hasard et R1

Seeds 0 à 29, T = 60 :

| Politique | Capture | IC 95 % | Confinement à T | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|
| R0 — aléatoire | 0 % | ± 0 % | 0.519 | — |
| A1, première campagne (ancienne perception) | 0 % | ± 0 % | 0.537 | — |
| **A1, perception corrigée** | **30 %** | ± 16 % | **0.281** | 33 |
| R1 — gloutonne locale, sans communication | 63 % | ± 17 % | 0.072 | 34 |
| R2 — croyance fusionnée, délai 1 pas | 83 % | ± 13 % | 0.029 | 30 |

- 9 captures (seeds 1, 4, 6, 7, 8, 11, 12, 13, 20), dont 3 là où R1 échoue
  (1, 6, 20).
- Aucun repli sur 3 144 décisions ; 434 tokens de complétion et 500 tokens
  de prompt par décision en moyenne.
- Les 21 échecs se partagent en deux groupes : **10 seeds où la cible est
  localisée sans être prise** (confinement ≤ 0,020 : 0, 5, 10, 14, 19, 22,
  24, 25, 26, 29) et **10 seeds où elle n'est jamais trouvée** (confinement
  ≥ 0,72 : 2, 9, 15, 16, 17, 18, 21, 23, 27, 28). La seed 3 est entre les deux
  (0,449).

## Ce que montrent les traces

Le modèle lit maintenant la perception sans faute :

| Sur les 3 144 décisions | |
|---|---|
| Immobile, ou contre un mur | 0 % |
| Cible visible : coup qui la rapproche (distance de chemin) | 100 % (356 décisions) |
| Cible dans l'axe nord/sud : bon sens / sens inverse | 28 / 0 |
| Cible cachée : coup vers une issue annoncée avec des candidates | 100 % |
| Cible cachée : coup vers l'issue dont la candidate est la plus proche | 93 % |
| **Allers-retours** (retour sur la case d'avant) | **50 %** |

Il se comporte donc comme une politique gloutonne « vers la case candidate
la plus proche ». C'est la première version de R1, abandonnée au jalon 1
parce que « les cases vidées redeviennent candidates presque aussitôt et les
poursuivants piétinaient » ([`jalon1.md`](jalon1.md)). Le champ de vision est
orienté : la case qu'on vient de quitter sort du champ et redevient candidate
au pas suivant, à 1 pas, derrière soi. Seed 18, poursuivant 0 :

```
pas 20 en (10, 13) -> EAST
- EST : praticable ; 47 cases candidates au plus court par là, la plus proche à 1 pas
- OUEST : praticable ; 44 cases candidates au plus court par là, la plus proche à 4 pas

pas 21 en (11, 13) -> WEST
- NORD : praticable ; 46 cases candidates au plus court par là, la plus proche à 3 pas
- OUEST : praticable ; 46 cases candidates au plus court par là, la plus proche à 1 pas

pas 22 en (10, 13) -> EAST   (même perception qu'au pas 20)
```

- **Cible jamais trouvée** : dans 5 de ces 10 seeds (2, 15, 18, 21, 23), les
  deux poursuivants font l'aller-retour entre 2 cases pendant tout
  l'épisode (93 à 97 % d'allers-retours, 2 à 4 cases visitées).
- **Cible localisée sans être prise** : un poursuivant la voit et la suit
  (100 % de coups qui la rapprochent), mais la cible va aussi vite que lui.
  Dans 4 de ces 10 seeds (10, 22, 24, 29), l'autre poursuivant oscille entre
  2 cases pendant tout l'épisode et ne vient jamais fermer l'autre côté.

Le piétinement est donc la cause commune des deux groupes. R1 l'évite avec
une carte de probabilité (la case qu'on vient de vider n'a presque plus de
masse) et une hystérésis sur son objectif. Chaque appel au LLM est un tour
isolé, sans mémoire : rien dans la perception ne le distingue d'une vraie
piste.

## Détail par seed

| Seed | Capture | Pas | Confinement à T | Temps mural (s) | Latence moy. (ms) | Replis |
|---|---|---|---|---|---|---|
| 0 | non | 60 | 0.010 | 4535.6 | 37560 | 0 |
| 1 | oui | 32 | 0.000 | 2739.0 | 38271 | 0 |
| 2 | non | 60 | 0.816 | 3998.8 | 33090 | 0 |
| 3 | non | 60 | 0.449 | 4323.8 | 35793 | 0 |
| 4 | oui | 33 | 0.000 | 2681.0 | 40212 | 0 |
| 5 | non | 60 | 0.010 | 5170.8 | 40663 | 0 |
| 6 | oui | 40 | 0.000 | 2897.2 | 35880 | 0 |
| 7 | oui | 52 | 0.000 | 4367.5 | 41723 | 0 |
| 8 | oui | 46 | 0.000 | 4193.6 | 44082 | 0 |
| 9 | non | 60 | 0.745 | 5531.1 | 46084 | 0 |
| 10 | non | 60 | 0.010 | 4872.0 | 39447 | 0 |
| 11 | oui | 10 | 0.000 | 732.2 | 36590 | 0 |
| 12 | oui | 25 | 0.000 | 1848.6 | 36953 | 0 |
| 13 | oui | 43 | 0.000 | 3304.5 | 36806 | 0 |
| 14 | non | 60 | 0.010 | 5016.9 | 41798 | 0 |
| 15 | non | 60 | 0.724 | 4921.2 | 41002 | 0 |
| 16 | non | 60 | 0.765 | 4825.6 | 40209 | 0 |
| 17 | non | 60 | 0.745 | 5520.7 | 46003 | 0 |
| 18 | non | 60 | 0.857 | 4202.3 | 35016 | 0 |
| 19 | non | 60 | 0.010 | 4769.3 | 38544 | 0 |
| 20 | oui | 31 | 0.000 | 2376.4 | 38325 | 0 |
| 21 | non | 60 | 0.827 | 4469.8 | 37245 | 0 |
| 22 | non | 60 | 0.020 | 4441.9 | 37012 | 0 |
| 23 | non | 60 | 0.816 | 4993.7 | 40472 | 0 |
| 24 | non | 60 | 0.010 | 4517.5 | 37643 | 0 |
| 25 | non | 60 | 0.010 | 4480.2 | 37332 | 0 |
| 26 | non | 60 | 0.010 | 4146.1 | 34548 | 0 |
| 27 | non | 60 | 0.735 | 3869.7 | 32244 | 0 |
| 28 | non | 60 | 0.827 | 3417.3 | 28474 | 0 |
| 29 | non | 60 | 0.010 | 3129.3 | 26074 | 0 |

Paramètres du jeu :

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
