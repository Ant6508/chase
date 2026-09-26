# Jalon 2, bras A1 — LLM sans communication

Campagne du 2026-09-25 (19h25 → 01h26, 6 h 01), code `e026511`, sur un pod
RunPod RTX 3090 (LM Studio headless, `google/gemma-4-12b` Q6_K, identifiant
`gemma-4-12b-a1`, `temperature=0`, `max_tokens=1600`, `timeout_s=180`,
2 retries). Commande, sur le pod :

```bash
bash scripts/pod/campaign.sh start a1 --episodes 30 --concurrency 8 \
    --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6
```

Journal par épisode : [`jalon2_a1_journal.jsonl`](jalon2_a1_journal.jsonl) ;
sortie brute : [`jalon2_a1_run.log`](jalon2_a1_run.log).

## Résultat : A1 au niveau du hasard

Mêmes seeds (0 à 29), même profil réduit, même instant de mesure T = 60 :

| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen |
|---|---|---|---|---|
| Poursuivants immobiles (`STAY` à chaque pas) | 0 % | ± 0 % | 0.753 | — |
| R0 — aléatoire | 0 % | ± 0 % | 0.519 | 0.595 |
| **A1 — LLM sans communication** | **0 %** | ± 0 % | **0.537** | — |
| R1 — gloutonne locale, sans communication | 63 % | ± 17 % | 0.072 | 0.296 |
| R2 — croyance fusionnée, délai 1 pas | 83 % | ± 13 % | 0.029 | 0.232 |

- R0, R1 et R2 : `python -m scripts.calibrate --episodes 30 --set max_steps=60
  size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6`. La ligne « immobiles »
  vient d'un calcul ponctuel non versionné (politique qui renvoie `STAY` pour
  les deux poursuivants).
- Le confinement moyen sur l'épisode n'est pas journalisé pour A1 : le
  journal ne garde que le confinement à T.

Lecture :

- **A1 ne fait pas mieux que R0.** Confinement à T de 0,537 contre 0,519, pire
  que R0 sur 17 seeds sur 30, aucune capture. Il fait mieux que l'immobilité
  (0,753) : le modèle agit, mais ses déplacements n'apportent pas plus
  d'information qu'une marche aléatoire.
- **Distribution bimodale.** Sur 7 seeds (8, 11, 15, 19, 20, 24, 25), A1
  ramène l'ensemble candidat d'équipe sous 10 % des cases libres, sans capture.
  Sur 17 seeds, il reste au-dessus de 0,7.
- **L'échec n'est pas technique.** 28 replis sur `STAY` sur 3 600 décisions
  (0,8 %). Environ 426 tokens de prompt et 540 tokens de complétion par
  décision ; latence moyenne 39 s par appel à 8 épisodes en parallèle.
- **Le profil réduit affaiblit la porte de sortie du jalon 1.** R2 y est à
  83 %, sous les 85 % de la SPEC (§4), et l'écart R1 → R2 tombe à 20 points,
  contre 40 sur le profil du jalon 1.

Ce résultat n'est pas figé comme point A1 définitif : la cause de l'échec
reste à établir (pilote de diagnostic avec traces pas à pas) avant de toucher
au prompt ou à la perception.

## Détail par seed

| Seed | Capture | Pas | Confinement à T | Temps mural (s) | Latence moy. (ms) | Replis |
|---|---|---|---|---|---|---|
| 0 | non | 60 | 0.827 | 4312.0 | 35929 | 0 |
| 1 | non | 60 | 0.776 | 4990.8 | 41586 | 0 |
| 2 | non | 60 | 0.469 | 8892.9 | 51873 | 3 |
| 3 | non | 60 | 0.816 | 4045.1 | 33706 | 0 |
| 4 | non | 60 | 0.490 | 9064.9 | 45997 | 4 |
| 5 | non | 60 | 0.898 | 4328.0 | 36064 | 0 |
| 6 | non | 60 | 0.204 | 6479.0 | 40056 | 3 |
| 7 | non | 60 | 0.827 | 4895.9 | 37960 | 1 |
| 8 | non | 60 | 0.010 | 5463.3 | 40772 | 0 |
| 9 | non | 60 | 0.224 | 5013.0 | 41774 | 0 |
| 10 | non | 60 | 0.765 | 7140.7 | 50143 | 1 |
| 11 | non | 60 | 0.031 | 9105.0 | 55626 | 2 |
| 12 | non | 60 | 0.908 | 3942.5 | 32853 | 0 |
| 13 | non | 60 | 0.796 | 4474.0 | 36320 | 0 |
| 14 | non | 60 | 0.714 | 4114.0 | 34282 | 0 |
| 15 | non | 60 | 0.010 | 7139.1 | 47494 | 2 |
| 16 | non | 60 | 0.888 | 5249.4 | 42733 | 0 |
| 17 | non | 60 | 0.786 | 4855.2 | 40460 | 0 |
| 18 | non | 60 | 0.837 | 4243.2 | 35360 | 0 |
| 19 | non | 60 | 0.082 | 5919.3 | 41420 | 2 |
| 20 | non | 60 | 0.061 | 6311.7 | 42799 | 2 |
| 21 | non | 60 | 0.796 | 4855.0 | 40458 | 0 |
| 22 | non | 60 | 0.847 | 4775.5 | 39795 | 0 |
| 23 | non | 60 | 0.194 | 6725.7 | 41841 | 2 |
| 24 | non | 60 | 0.082 | 5641.8 | 41195 | 1 |
| 25 | non | 60 | 0.031 | 5021.8 | 39009 | 1 |
| 26 | non | 60 | 0.235 | 5545.0 | 28036 | 4 |
| 27 | non | 60 | 0.867 | 3643.7 | 30363 | 0 |
| 28 | non | 60 | 0.776 | 2975.8 | 24797 | 0 |
| 29 | non | 60 | 0.867 | 3032.4 | 24381 | 0 |

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
