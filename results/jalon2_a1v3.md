# Jalon 2, bras A1 — troisième campagne (probabilité par issue)

Campagne locale, code `3145e07` (perception du commit `0491505` : part de
probabilité par issue), sur le LM Studio de la machine de travail
(RX 6750 XT, `google/gemma-4-12b` Q6_K, même fichier que les campagnes
RunPod, chargé avec `--context-length 10240 --parallel 4`). Mêmes réglages de
génération que [`jalon2_a1v2.md`](jalon2_a1v2.md) (`temperature=0`,
`max_tokens=1600`, 2 retries), délai porté à 400 s. Même profil réduit, mêmes
seeds 0 à 29.

- Seeds 0–3 : pilote du 2026-09-27, 13h37 → 15h17
  ([`jalon2_a1v3_pilote.md`](jalon2_a1v3_pilote.md)).
- Seeds 4–29 : du 2026-09-27 à 22h53 au 2026-09-28 à 23h42, avec une pause
  ratée à midi (voir [Incident](#incident--pause-ratée-et-journal-nettoyé)).

```bash
python -m scripts.run_llm --first-seed 0 --episodes 30 --concurrency 4 \
    --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --base-url http://127.0.0.1:1234/v1 --timeout 400 \
    --journal results/jalon2_a1v3_local/journal.jsonl \
    --trace results/jalon2_a1v3_local/trace --out results/jalon2_a1v3_local/table.md
```

Dans [`jalon2_a1v3_local/`](jalon2_a1v3_local/) :
- journal : `journal.jsonl` ;
- tableau par seed : `table.md` ;
- sorties brutes : `run.log` (pilote), `run30.log`, puis `run30b.log` après la reprise ;
- vérification par rejeu (voir [Réserves](#réserves)) : `rejeu.jsonl` et `rejeu.log` ;
- traces pas à pas : `trace/`, analysées par `python -m scripts.analyze_trace
  results/jalon2_a1v3_local/trace --set max_steps=60 size=15 n_loops=1
  min_loop_len=6 min_spawn_dist=6`.

## Résultat : A1 rejoint R1 sans le dépasser

Seeds 0 à 29, T = 60 :

| Politique | Capture | IC 95 % | Confinement à T | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|
| R0 — aléatoire | 0 % | ± 0 % | 0.519 | — |
| A1, première campagne (ancienne perception) | 0 % | ± 0 % | 0.537 | — |
| A1v2, perception corrigée | 30 % | ± 16 % | 0.281 | 33 |
| **A1v3, probabilité par issue** | **53 %** | ± 18 % | **0.106** | **29** |
| R1 — gloutonne locale, sans communication | 63 % | ± 17 % | 0.072 | 34 |
| R2 — croyance fusionnée, délai 1 pas | 83 % | ± 13 % | 0.029 | 30 |

Les 16 captures : seeds 0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 15, 20, 25, 26
et 28. Pour situer le LLM, l'essai mécanique d'avant campagne donne un point
de repère ([spec, révision du 2026-09-27](../docs/superpowers/specs/2026-09-24-jalon2-a1-llm-design.md)) :
un poursuivant qui suit toujours l'issue la plus probable capture 60 %.

Comparaison appariée sur les mêmes seeds (test de McNemar exact, bilatéral,
sur les seeds où une seule des deux politiques capture) :

| | Seul le premier capture | Seul le second capture | p |
|---|---|---|---|
| A1v3 contre A1v2 | 8 (0, 2, 3, 5, 15, 25, 26, 28) | 1 (6) | **0,04** |
| A1v3 contre R1 | 3 (1, 20, 26) | 6 (14, 17, 21, 22, 23, 24) | 0,51 |
| A1v3 contre R2 | 2 (12, 13) | 11 | **0,02** |
| R1 contre R2, pour mémoire | 2 (12, 13) | 8 | 0,11 |

- **Le correctif agit.** A1v3 capture significativement plus que A1v2, et
  son confinement moyen à T passe de 0,281 à 0,106.
- **A1v3 ne se distingue pas de R1.** Le LLM tire de la perception à peu près
  autant que l'heuristique qui en calcule le contenu. Il ne la recopie pas
  pour autant : sa trajectoire s'écarte de celle de R1 dès les 5 premiers pas
  dans 27 seeds, et aux pas 11 et 16 dans 2 autres (16 et 12). Seule la
  seed 10 est identique de bout en bout, et sans capture pour les deux.
- **R2 fait significativement mieux que A1v3.** Même R1 contre R2 n'est pas
  tranché à 30 seeds.

## Ce que montrent les traces

| | A1v2 | A1v3 |
|---|---|---|
| Décisions | 3 144 | 2 778 |
| Immobile, ou contre un mur | 0 % | 0 % (hors 2 replis) |
| Replis sur `STAY` | 0 | 2 (seeds 6 et 19) |
| Cible visible : coup qui la rapproche (distance de chemin) | 100 % (356) | 100 % (345) |
| Cible dans l'axe nord/sud : bon sens / sens inverse | 28 / 0 | 72 / 0 |
| Cible cachée : vers une issue annoncée avec des candidates | 100 % | 100 % |
| Cible cachée : vers la candidate la plus proche | 93 % | 69 % |
| Cible cachée : vers l'issue la plus probable | — | 79 % |
| **Allers-retours** (retour sur la case d'avant) | **50 %** | **36 %** |
| Tokens de prompt / de complétion par décision | 500 / 434 | 596 / 446 |

La lecture reste sans faute. Quand la cible est cachée, le modèle suit
l'issue la plus probable 79 % du temps. Le reste du temps, il revient à la
candidate la plus proche, et l'aller-retour réapparaît. Cela arrive surtout
quand la probabilité se partage presque à parts égales. C'est le cas de la
seed 18 : cible jamais vue, 97 % d'allers-retours, 4 cases visitées sur tout
l'épisode. Poursuivant 0 :

```
pas 10 en (10, 13) -> EAST
- EST : praticable ; 47 cases candidates au plus court par là (52 % de la probabilité), la plus proche à 1 pas
- OUEST : praticable ; 44 cases candidates au plus court par là (48 % de la probabilité), la plus proche à 4 pas

pas 11 en (11, 13) -> WEST
- NORD : praticable ; 46 cases candidates au plus court par là (53 % de la probabilité), la plus proche à 3 pas
- OUEST : praticable ; 46 cases candidates au plus court par là (47 % de la probabilité), la plus proche à 1 pas
  « La cible est potentiellement très proche à l'ouest (à seulement 1 pas) »

pas 12 en (10, 13) -> EAST   (même perception qu'au pas 10)
```

Au pas 11, le modèle prend l'issue la moins probable parce que sa candidate
est à 1 pas. Le poursuivant mécanique aurait pris le nord.

## Les 14 échecs

Colonne « Cible vue » : nombre de pas où au moins un poursuivant voit la cible.

| Groupe | Seeds | Cible vue | R1 | R2 |
|---|---|---|---|---|
| Jamais trouvée (confinement à T ≥ 0,62) | 14, 18, 19, 22 | 0 pas | prend 14 et 22 | prend les 4 |
| Trouvée tard | 6, 17, 23, 27 | 1 à 8 pas | prend 17 et 23 | prend 6, 17 et 23 |
| Poursuivie sans être prise | 9, 10, 16, 21, 24, 29 | 13 à 39 pas | prend 21 et 24 | prend 9, 10, 21 et 24 |

- **La recherche s'est nettement améliorée.** A1v2 avait 11 seeds sans
  jamais trouver la cible, A1v3 n'en a plus que 4.
- **Le mode d'échec dominant est la poursuite sans prise.** La cible va aussi
  vite que le poursuivant qui la suit. Sur les seeds 9 et 10, R2 capture
  alors que R1 et A1v3 échouent. Ce qui manque est la coordination du second
  poursuivant, que la communication doit apporter.
- **Un indicateur simple de tenaille ne sépare pas les captures des échecs.**
  L'indicateur regarde, quand la cible est vue, si le poursuivant le plus
  éloigné est derrière l'autre sur le plus court chemin vers elle. C'est le
  cas 63 % du temps dans les captures d'A1v3, 58 % dans ses échecs localisés
  et 53 % chez R2. Il faudra un diagnostic plus fin pour dire comment R2
  referme la cible.
- **La marge d'A2 est de 11 seeds.** R2 capture 11 des 14 seeds manquées par
  A1v3. C'est ce que le bras A2 peut espérer récupérer sur ce profil.

## Réserves

1. **L'écart à R1 tient entièrement aux 12 seeds rejouées après l'incident**
   (17, 19–29).
   - Sur ces 12 seeds : A1v3 en capture 4, R1 7.
   - Sur les 18 autres : 12 chacun.
   - Si les 16 captures étaient réparties au hasard, 12 seeds tirées en
     contiendraient 4 ou moins avec une probabilité de 0,08.

   Code, fichier de modèle et paramètres de chargement sont les mêmes. Pour
   écarter une différence de service, 24 décisions ont été rejouées le
   2026-09-29 sur le serveur de la reprise, toujours chargé
   ([`rejeu.jsonl`](jalon2_a1v3_local/rejeu.jsonl)). Ce sont 12 décisions
   tirées avant la pause et 12 après la reprise, choisies parmi celles à au
   moins 2 issues praticables et renvoyées avec leur perception exacte :

   ```bash
   python -m scripts.replay_trace results/jalon2_a1v3_local/trace \
       --groups avant=4-16,18 après=17,19-29 -n 12 --concurrency 4 \
       --base-url http://127.0.0.1:1234/v1 --timeout 400 \
       --out results/jalon2_a1v3_local/rejeu.jsonl
   ```

   | | Avant la pause | Après la reprise |
   |---|---|---|
   | Même coup | 11 / 12 | 12 / 12 |
   | Même pensée, mot pour mot | 1 / 12 | 2 / 12 |

   - **Le serveur n'a pas changé de comportement à la reprise.** Les deux
     groupes se reproduisent aussi bien.
   - **Le seul coup différent est un choix serré.** À la seed 14, pas 52, le
     modèle avait pris l'ouest à 43 % de probabilité ; au rejeu, il prend le
     sud à 40 %.
   - **Le déficit des 12 seeds rejouées ne vient donc pas du service.** Il
     tient au hasard des seeds et au bruit décrit ci-dessous.

   Température 0 ne veut pas dire déterminisme ici. La pensée diffère
   presque toujours d'une exécution à l'autre, y compris dans une même
   session de serveur, parce que le calcul par lots dépend des requêtes
   voisines. Le coup, lui, reste le même environ 23 fois sur 24. Sur une
   partie d'une centaine de décisions, quelques coups changent quand même,
   et la trajectoire diverge. Rejouer une seed ne rejoue donc pas l'épisode :
   la comparaison appariée apparie des cartes et des départs, pas des
   parties identiques.
2. **Le matériel a changé en cours de route.** A1 et A1v2 tournaient sur
   RunPod (CUDA), A1v3 en local (Vulkan sur carte AMD). Le fichier de modèle
   est le même, mais les réponses peuvent différer à la marge, même à
   température 0.
3. **Latence et temps mural ne sont pas représentatifs.**
   - Latence moyenne par appel : 111 s en médiane sur les épisodes (51 à
     177 s).
   - Ce qui l'explique : 4 parties en parallèle sur une carte de 12 Go, avec
     un débordement probable de la VRAM pendant la nuit.
   - Temps cumulé des épisodes : 89 h, pour environ 26 h de campagne réelle.
4. **30 seeds donnent ± 18 points.** L'écart entre A1v3 et R1 est hors de
   portée d'une campagne de cette taille ; seules les comparaisons appariées
   ci-dessus sont informatives.

## Incident : pause ratée et journal nettoyé

Le 2026-09-28 vers 12h30, la campagne a été mise en pause en gelant le
processus, puis le modèle a été déchargé.

- **Le mauvais processus a été gelé.** C'était le lanceur du venv
  (`.venv\Scripts\python.exe`), pas l'interpréteur Python qu'il lance, et
  celui-ci a continué de tourner.
- **12 épisodes se sont joués sans modèle.** Ce sont les épisodes en cours ou
  en attente, seeds 17 et 19 à 29. Ils se sont terminés en replis sur `STAY` :
  de 21 à 120 par épisode, 120 signifiant que tout l'épisode s'est joué sans
  modèle.
- **Ils ont été retirés du journal, avec leurs traces.** La seed 18, terminée
  avant le déchargement sans aucun repli, a été gardée.
- **Ils ont été rejoués depuis le début** après rechargement du modèle, à
  13h22.

Le journal d'avant nettoyage est conservé :
[`jalon2_a1v3_local/journal_avant_nettoyage.jsonl`](jalon2_a1v3_local/journal_avant_nettoyage.jsonl).

## Détail par seed

Entre parenthèses : confinement à T quand la cible n'est pas prise. Latence
et temps mural : [`jalon2_a1v3_local/table.md`](jalon2_a1v3_local/table.md).

| Seed | A1v2 | A1v3 | R1 | R2 |
|---|---|---|---|---|
| 0 | non (0,010) | **capture au pas 22** | capture au pas 35 | capture au pas 22 |
| 1 | capture au pas 32 | **capture au pas 19** | non (0,010) | capture au pas 27 |
| 2 | non (0,816) | **capture au pas 28** | capture au pas 30 | capture au pas 30 |
| 3 | non (0,449) | **capture au pas 58** | capture au pas 39 | capture au pas 39 |
| 4 | capture au pas 33 | **capture au pas 33** | capture au pas 33 | capture au pas 33 |
| 5 | non (0,010) | **capture au pas 25** | capture au pas 27 | capture au pas 35 |
| 6 | capture au pas 40 | non (0,061) | non (0,010) | capture au pas 56 |
| 7 | capture au pas 52 | **capture au pas 46** | capture au pas 55 | capture au pas 22 |
| 8 | capture au pas 46 | **capture au pas 30** | capture au pas 28 | capture au pas 28 |
| 9 | non (0,745) | non (0,020) | non (0,020) | capture au pas 48 |
| 10 | non (0,010) | non (0,010) | non (0,010) | capture au pas 37 |
| 11 | capture au pas 10 | **capture au pas 57** | capture au pas 20 | capture au pas 20 |
| 12 | capture au pas 25 | **capture au pas 26** | capture au pas 57 | non (0,010) |
| 13 | capture au pas 43 | **capture au pas 40** | capture au pas 35 | non (0,010) |
| 14 | non (0,010) | non (0,714) | capture au pas 20 | capture au pas 20 |
| 15 | non (0,724) | **capture au pas 10** | capture au pas 8 | capture au pas 8 |
| 16 | non (0,765) | non (0,010) | non (0,010) | non (0,010) |
| 17 | non (0,745) | non (0,010) | capture au pas 34 | capture au pas 26 |
| 18 | non (0,857) | non (0,857) | non (0,776) | capture au pas 30 |
| 19 | non (0,010) | non (0,622) | non (0,510) | capture au pas 49 |
| 20 | capture au pas 31 | **capture au pas 56** | non (0,041) | capture au pas 30 |
| 21 | non (0,827) | non (0,010) | capture au pas 29 | capture au pas 29 |
| 22 | non (0,020) | non (0,806) | capture au pas 45 | capture au pas 52 |
| 23 | non (0,816) | non (0,020) | capture au pas 31 | capture au pas 28 |
| 24 | non (0,010) | non (0,020) | capture au pas 40 | capture au pas 44 |
| 25 | non (0,010) | **capture au pas 24** | capture au pas 48 | capture au pas 24 |
| 26 | non (0,010) | **capture au pas 27** | non (0,010) | capture au pas 27 |
| 27 | non (0,735) | non (0,010) | non (0,745) | non (0,827) |
| 28 | non (0,827) | **capture au pas 48** | capture au pas 46 | capture au pas 56 |
| 29 | non (0,010) | non (0,010) | non (0,010) | non (0,010) |
| **Captures** | 9 / 30 | **16 / 30** | 19 / 30 | 25 / 30 |
