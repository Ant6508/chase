# Jalon 2, bras A3 — canal JSON minimal sous budget borné, contre A1bis et A2

**En bref.**
- **Captures : A3 ne fait pas mieux que les autres bras LLM.** Il capture 16 fois sur 30 (53 %),
  contre 19 pour A1bis (p = 0,51), 18 pour A2 (p = 0,73) et 15 pour A2' (p = 1,00). Il reste sous
  R2 (p = 0,04) et sous P2 (p = 0,003).
- **Le message est six fois plus court qu'en A2** : 22 positions en moyenne, contre 138. La borne
  d'`intention` (3 lieux) est respectée sans aucun rejet.
- **Les usages positifs du canal survivent, en partie.**
  - **L'écartement en exploration est intact.** Les deux poursuivants sont à 3 cases ou moins
    10,6 % du temps, contre 27,5 % pour A1bis (p < 0,001) et 12,6 % pour A2 (p = 0,20).
  - **Le récepteur rejoint encore la cible vue par son coéquipier**, mais moins souvent : 81 %,
    contre 63 % pour A1bis (p = 0,02) et 92 % pour A2 (p = 0,04).
- **Les messages sont justes.** `cible` est exacte chaque fois qu'elle est vue, et seuls 0,7 % des
  messages relaient la cible du coéquipier (A2' : 14,4 %). `moi` est moins précis qu'en A2 : 73 %
  de cases exactes, contre 95 %.
- **Le coût de pensée est divisé par deux par rapport à A2** (1 179 tokens par décision, contre
  2 455), et il n'y a presque plus de relances (0,4 % des décisions, contre 16 %).
- **Le goulot reste la phase de capture.** A3 laisse à la cible un territoire de 18,7 cases, contre
  12,2 pour P3 et 11,7 pour R2. Aucun message ne change cela.

## 1. Campagne

| | A3 |
|---|---|
| Seeds | 0 à 29 |
| Dates (UTC) | pilote, seeds 0 à 7 : le 2026-10-08, de 15h34 à 17h47. Seeds 8 à 29 : le 2026-10-09, de 05h39 à environ 14h10 |
| Durée sur le pod | 2 h 13 de pilote et environ 8 h 30 de campagne |
| Fichiers | [`jalon2_a3/`](jalon2_a3/), [`jalon2_a3_pilote.md`](jalon2_a3_pilote.md), [`jalon2_a3_plafond.md`](jalon2_a3_plafond.md) |

- **Matériel.** Pod RunPod `0jj7z07rmajhuv` : A40 48 Go, Secure, CA-MTL-1. C'est la même carte et
  le même data center que la campagne A2.
  - LM Studio, moteur `llama.cpp-linux-x86_64-nvidia-cuda12-avx2@2.41.0`, `google/gemma-4-12b`
    Q6_K, 8 créneaux de 8 192 tokens, 13 911 Mo de VRAM. C'est la pile d'A2 et d'A1bis.
- **Code.** `1c0d5b2`. Le pod a été mis en pause entre le pilote et la campagne, puis redémarré
  sur la même machine : le journal du pilote a été repris tel quel.

```bash
campaign.sh start a3 --arm A3 --max-tokens 5600 --timeout 900 --first-seed 0 --episodes 30 \
    --concurrency 8 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --trace /workspace/results/a3/trace
```

Le rapatriement a été vérifié par empreintes MD5 (35 fichiers). Aucun épisode n'a échoué.

## 2. Tableau

Seeds 0 à 29, T = 60. Les IC sont à 95 %, approximation normale.

| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|---|
| R0 — aléatoire | 0 % | ± 0 % | 0,519 | 0,595 | — |
| R1 — gloutonne, sans communication | 63 % | ± 17 % | 0,072 | 0,296 | 34 |
| R2 — croyance fusionnée, délai 1 pas | 83 % | ± 13 % | 0,029 | 0,232 | 30 |
| P2 — décisions de R2, message A2 seul | 90 % | ± 11 % | 0,030 | 0,225 | 30 |
| P3 — décisions de R2, message A3 seul | 73 % | ± 16 % | 0,042 | 0,231 | 28 |
| A1v3 — LLM sans lieux nommés | 53 % | ± 18 % | 0,106 | 0,362 | 29 |
| A1bis — LLM avec lieux nommés, témoin | 63 % | ± 17 % | 0,118 | 0,352 | 34 |
| A2 — LLM avec canal JSON | 60 % | ± 18 % | 0,059 | 0,318 | 31,5 |
| A2' — canal JSON et consigne d'usage | 50 % | ± 18 % | 0,085 | 0,323 | 32 |
| **A3 — LLM avec canal JSON minimal** | **53 %** | ± 18 % | **0,127** | **0,320** | **34** |

R0 à A2' viennent de [`jalon2_a2p.md`](jalon2_a2p.md), P3 de [`jalon2_a3_plafond.md`](jalon2_a3_plafond.md).

**La forme des échecs.**

| Bras | Cible perdue (confinement à T > 0,3) | Localisée sans capture (0,1 ou moins) |
|---|---|---|
| A1bis | 5 seeds : 9, 14, 15, 19, 27 | 6 seeds |
| A2 | 2 seeds : 7, 11 | 10 seeds |
| A2' | 3 seeds : 3, 7, 28 | 11 seeds |
| A3 | 5 seeds : 11, 14, 19, 21, 28 | 9 seeds |

- **A3 perd la cible aussi souvent qu'A1bis.** Il retrouve la proportion du témoin, et son
  confinement à T (0,127) aussi.
- **A2 la perdait moins.** C'est ce que `candidates` apportait : le partage de la recherche, au
  grain des lieux, dont le confinement moyen ne garde qu'une faible trace (0,318 contre 0,320).
- **A3 échoue aussi en finale** (9 seeds) : la cible est localisée, mais pas prise.

## 3. Comparaisons appariées

Test de McNemar exact, bilatéral, sur les seeds où une seule des deux politiques capture.

```bash
python -m scripts.compare_arms --episodes 30 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 \
    min_spawn_dist=6 --journal A1bis=results/jalon2_a1bis/journal.jsonl \
    --journal A2=results/jalon2_a2/journal.jsonl --journal A2p=results/jalon2_a2p/journal.jsonl \
    --journal A3=results/jalon2_a3/journal.jsonl --pairs A3:A1bis A3:A2 A3:A2p A3:P3 A3:R2 A3:P2 R1:R2
```

| Paire | Captures | Seul le premier | Seul le second | p |
|---|---|---|---|---|
| **A3 contre A1bis** | 16 / 19 | 3 (6, 15, 27) | 6 (5, 11, 16, 20, 24, 28) | 0,51 |
| **A3 contre A2** | 16 / 18 | 3 (6, 7, 8) | 5 (10, 14, 21, 24, 28) | 0,73 |
| A3 contre A2' | 16 / 15 | 5 (3, 7, 12, 26, 27) | 4 (9, 10, 11, 24) | 1,00 |
| A3 contre P3 | 16 / 22 | 3 (6, 18, 27) | 9 (5, 11, 14, 19, 21, 22, 23, 24, 28) | 0,15 |
| A3 contre R2 | 16 / 25 | 3 (12, 13, 27) | 12 (5, 9, 10, 11, 14, 19, 20, 21, 22, 23, 24, 28) | **0,04** |
| A3 contre P2 | 16 / 27 | 1 (27) | 12 (5, 9, 10, 11, 14, 19, 20, 21, 22, 23, 24, 28) | **0,003** |
| R1 contre R2, pour contrôle | 19 / 25 | 2 (12, 13) | 8 (1, 6, 9, 10, 18, 19, 20, 26) | 0,11 |

- **Le canal minimal ne change pas les captures (A3 contre A1bis).** C'était déjà le cas du canal
  complet (A2 contre A1bis : p = 1,00).
- **Retirer `candidates` et `je_couvre` ne coûte pas de captures démontrables (A3 contre A2).** À
  30 seeds, seul un écart d'une vingtaine de points serait visible.
- **A3 n'atteint pas son plafond mécanique (P3), sans écart démontré (p = 0,15).** Le message
  minimal, exploité par les décisions de R2, capture 6 seeds de plus.
- **La ligne R1 contre R2 redonne 19 / 25 avec p = 0,11** : le profil passé à `--set` est le bon.

## 4. La courbe de Pareto

Positions par message et par épisode, recalculées de la même façon pour les trois bras à canal,
sur les décisions sans repli.

| Bras | Positions par message : moyenne | médiane (min. – max.) | Positions par épisode : moyenne | Confinement à T | Captures |
|---|---|---|---|---|---|
| A1bis | 0 | — | 0 | 0,118 | 63 % |
| **A3** | **22,3** | 22 (15 – 42) | **2 049** | 0,127 | 53 % |
| A2' | 119,2 | 131 (31 – 305) | 10 892 | 0,085 | 50 % |
| A2 | 138,0 | 167 (31 – 258) | 12 257 | 0,059 | 60 % |

- **Sur les captures, la courbe est plate.** Aucun point ne se distingue d'un autre.
- **Sur le confinement à T, seul le message complet compte.** A2 divise par deux le confinement
  d'A1bis (p = 0,41 à 30 seeds, voir [`jalon2_a2.md`](jalon2_a2.md)), et A3 revient au niveau
  d'A1bis.
- **A3 se place face à `m = 32`**, l'une des longueurs latentes du papier RecursiveMAS (voir la
  spec A3, § Ce que fait le papier). A4 pourra être comparé à budget égal en visant cette longueur.
- **Le maximum de 42 positions** dépasse les 36 estimés par la spec. Les noms de case longs
  (`C10a.5`) et une `intention` de 3 lieux suffisent à l'atteindre, sans enfreindre la borne.

## 5. La qualité des messages

Mesures du rejeu de chaque épisode par [`jalon2_a2/msg_usage.py`](jalon2_a2/msg_usage.py), qui lit
aussi les messages A3 (sans `candidates`).

- **Messages valides.** 2 760 sur 2 760 (100 %). Aucun rejet pour la borne ou pour un champ en
  trop.
- **Relances.** Il y en a eu 11 sur toute la campagne, toutes rattrapées : 10 réponses coupées à
  5 600 tokens, et 1 réponse sans appel d'outil.
- **Noms inconnus.** 26, dans 26 messages (0,9 %). Ce sont tous des rangs qui n'existent pas
  (`C2a.6` dans un tronçon de 5 cases, par exemple), comme en A2.
- **`intention`.** 1 lieu dans 194 messages, 2 dans 2 457 et 3 dans 109. Le modèle n'approche
  jamais la borne : il écrit presque toujours deux lieux.
- **`moi` est moins précis qu'en A2.**

  | | A3 | A2 |
  |---|---|---|
  | Case exacte | 73,0 % | 94,9 % |
  | Case où il sera après son coup | 15,6 % | 3,2 % |
  | Ni l'une ni l'autre | 11,3 % | 1,9 % |
  | Bon lieu, rang compris ou non | 91,9 % | 98,6 % |

  A3 désigne plus souvent la case où il sera quand son coéquipier lira le message. C'est une
  interprétation défendable du champ, mais ce n'est pas celle du prompt (« ta case »). Le reste des
  erreurs porte sur le rang dans le couloir.
- **`cible` est juste.**
  - Quand l'émetteur voit la cible (453 messages), elle est exacte dans 100 % des cas.
  - Quand il ne la voit pas (2 307 messages), elle est inventée dans 0,7 % des cas (16 messages).
    Ces 16 messages relaient tous la cible reçue du coéquipier, à moins d'une case de la vraie
    position en moyenne.
  - **Le relais d'A2' ne revient pas** : 14,4 % des messages d'A2' relayaient la cible, contre
    0,7 % ici, comme en A2 (0,6 %). C'était bien un effet de la consigne d'usage.

## 6. L'usage des messages, lu dans les traces

Même méthode que les rapports A2 et A2' (§ 6) : permutations appariées par seed, 100 000 tirages.
Pour A1bis, le savoir du coéquipier est pris dans le message mécanique de P2.

| Question | A3 | A1bis | A2 | A3 contre A1bis | A3 contre A2 |
|---|---|---|---|---|---|
| (a) Cible vue par le coéquipier : vers une issue morte | 20,8 % (42 / 202) | 38,9 % | 9,0 % | p = 0,02 | p = 0,04 |
| (a) Lieux vus vides seulement : vers une issue morte | 46,2 % (120 / 260) | 16,0 % | 26,4 % | p = 0,01 | p = 0,007 |
| (b) Seul le coéquipier voit la cible : coup qui la rapproche | 81,0 % (183 / 226) | 63,3 % | 91,6 % | p = 0,02 | p = 0,04 |
| (c) Exploration : à 3 cases ou moins | 10,6 % (109 / 1 026) | 27,5 % | 12,6 % | p < 0,001 | p = 0,20 |
| (d) Phase de capture : en tenaille | 27,4 % (97 / 354) | 24,6 % | 21,0 % | p = 0,63 | p = 0,07 |
| (d) Phase de capture : territoire moyen | 18,7 cases | 20,7 | 22,0 | p = 0,70 | p = 0,42 |

### (a) et (b) La cible vue par le coéquipier

Le message porte `cible` dans 100 % des 226 situations (b). A3 rejoint la cible plus souvent
qu'A1bis, qui ne la connaît pas, mais moins souvent qu'A2.

**Exemple** (seed 1, pas 13, poursuivant 1).
- **Le message reçu :** `{"cible":"C5c.3","intention":["C5c","K5"],"moi":"C5b.4"}`.
- **La situation.** Le récepteur est en K3 ; sa perception donne SUD à 77 %, avec C5c à 7 pas.
- **Sa justification :** *« Moving towards the target's last known position (C5c) and converging
  with my teammate who is heading towards K5/C5c. »* Il joue SUD.

Le pas précédent montre la lecture inverse. Le même poursuivant lisait `"cible":"C5c.2"` comme
*« the teammate […] was looking for the target near C5c »*. Il allait dans la bonne direction, mais
comme vers la zone de recherche du coéquipier, pas comme vers une cible localisée.

### (a) Les lieux vus vides : sans `candidates`, l'information négative a disparu

A3 ne reçoit plus les lieux vus vides. Quand eux seuls écartent une issue, il prend une issue morte
**46 % du temps, contre 16 % pour A1bis et 26 % pour A2.** Quand son issue la plus probable est
morte (44 fois), il la suit à chaque fois.

Qu'il fasse moins bien qu'A2 est attendu, puisqu'il n'a plus l'information. Qu'il fasse moins bien
qu'A1bis l'est moins. **Hypothèse** : en se répartissant la recherche à partir de `moi` et
d'`intention`, le récepteur part à l'opposé de là où va son coéquipier, c'est-à-dire souvent vers
là d'où celui-ci vient, et qu'il a déjà vu vide. C'est le contresens déjà décrit dans le rapport
A2 (§ 6 (a)) : la position et l'intention sont lues comme une répartition des zones. Sans
`candidates`, rien ne vient le corriger. Les situations ne sont que de même nature d'un bras à
l'autre (260 pour A3, 181 pour A1bis), et ce mécanisme reste à vérifier dans les traces.

### (c) L'écartement en exploration est entièrement conservé

C'est l'usage le mieux conservé : 10,6 % des pas à 3 cases ou moins, autant qu'A2 (12,6 %) et
que P3 (9,9 %). La position et l'intention suffisent à se répartir. `candidates` n'y était pour
rien.

### (d) La phase de capture : rien de nouveau

A3 laisse à la cible un territoire de 18,7 cases et la prend en tenaille 27 % du temps. C'est
mieux qu'A2 sur les deux points, mais sans écart significatif, et c'est toujours loin de R2
(11,7 cases), de P2 (10,6) et de P3 (12,2). Comme dans le rapport A2' (§ 5.3), c'est là que se
joue l'écart avec le plafond mécanique.

### Les autres indicateurs de trace

Ils viennent de `python -m scripts.analyze_trace results/jalon2_a3/trace`.

| | A1bis | A2 | A3 |
|---|---|---|---|
| Décisions | 2 714 | 2 732 | 2 760 |
| Replis | 0 | 68 (2,5 %) | 0 |
| Allers-retours | 30 % | 18 % | 25 % |
| Cible visible : coup qui la rapproche | 100 % | 100 % | 100 % |
| Cible cachée : vers l'issue la plus probable | 81 % | 79 % | 78 % |

## 7. Les coûts

Valeurs recalculées de la même façon pour les quatre bras, sur les décisions sans repli. Elles
peuvent différer de quelques pour cent de celles des rapports précédents, qui n'utilisaient pas
tous le même dénominateur.

| Par décision | A1bis | A2 | A2' | A3 |
|---|---|---|---|---|
| Tokens de prompt | 1 152 | 1 703 | 1 841 | 1 448 |
| Tokens de complétion (tentative acceptée) | 548 | 2 658 | 2 531 | 1 267 |
| dont pensée | 492 | 2 455 | 2 344 | 1 179 |
| Positions de message | — | 138,0 | 119,2 | 22,3 |
| Décisions relancées | 0,1 % | 16,2 % | 15,9 % | 0,4 % |
| Relances par décision | 0,001 | 0,219 | 0,224 | 0,004 |
| Latence moyenne | 52 s | 162 s | 159 s | 95 s |
| Temps mural médian d'un épisode | 1,3 h | 5,7 h | 5,6 h | 2,4 h |

- **Le canal coûte de la pensée même sans liste.** A3 pense 2,4 fois plus qu'A1bis : le modèle lit
  le message reçu, puis rédige le sien.
- **La liste des lieux coûtait l'essentiel.** Sans elle, la pensée est divisée par deux et les
  relances disparaissent : les 16 % de relances d'A2 venaient des réponses coupées à 5 600 tokens
  pendant la rédaction de `candidates`.
- **En argent.**
  - Environ 8,14 $ de pod pour A3. Le premier jour, l'A40 était facturée 0,49 $/h ; au redémarrage,
    RunPod l'affichait à 0,59 $/h.
  - Le premier jour compte 3 h 45 de pod inactif après le pilote (environ 1,85 $), à cause d'une
    surveillance défaillante ([`jalon2_a3_pilote.md`](jalon2_a3_pilote.md), § Coût et durée).
  - Hors cette perte, le pilote et la campagne ont coûté environ 6,30 $.

## 8. Ce que dit A3

**Le message minimal garde ce qui sert aux LLM, et ce qui sert ne change pas les captures.**
- **L'écartement en exploration ne demande que `moi` et `intention`.** A3 le conserve entièrement.
- **Rejoindre la cible vue ne demande que `cible`.** A3 le conserve en partie : 81 %, contre 92 %
  pour A2.
- **L'information négative demandait `candidates`.** Sans consigne d'usage, A2 ne s'en servait
  déjà pas (rapport A2, § 6 (a)). A3 la perd sans perte de captures démontrable.

**Le budget minimal est donc le point le plus efficace de la courbe texte**, à performance égale :
six fois moins de positions qu'A2, deux fois moins de pensée, et presque aucune relance.

**La phase de capture reste le goulot** : aucun bras à canal ne s'approche de son plafond mécanique,
et l'écart est toujours dans la prise en tenaille.

**Le résultat reste limité à ce protocole, ce prompt et ce modèle**, comme pour A2 et A2'.

## 9. La revendication

- **Le schéma est un protocole conçu par nous** (SPEC, § 3). A3 mesure « le texte à charge utile
  fixée et bornée », pas des agents qui découvrent quoi se dire.
- **Le budget est imposé par le schéma** : trois champs, `intention` limitée à 3 lieux, aucun champ
  en trop, vérifiés après chaque appel. Un message qui enfreint la borne est rejeté et l'appel
  relancé.
- **A3 n'est pas le bras JSON complet** demandé par la SPEC (ensemble candidat pondéré, intention,
  issue couverte) : ce bras-là est A2. A3 est le point bas de la courbe.
- **Ce que fait le papier RecursiveMAS** : il fixe la longueur du message latent (`m` pas, de 16
  à 64 dans ses réglages publiés), mais ne borne son témoin texte que par une coupure de
  génération, sans comparaison à budget égal. A3 rend cette comparaison possible face à `m = 32`.

## 10. Les réserves

- **La taille de l'échantillon.** 30 seeds donnent environ ± 18 points sur les captures. Un écart de
  2 ou 3 captures n'est pas interprétable.
- **Le non-déterminisme du serveur.** À température 0, il n'est pas déterministe sous 8 créneaux
  parallèles (rapport A1v3). A3 a tourné sur la même carte, le même data center et la même pile
  qu'A2, mais pas sur la même machine.
- **La reprise après pause.** Le pilote et la campagne ont été joués à 12 heures d'écart, sur le
  même pod et le même code, après un redémarrage de LM Studio. Rien ne distingue les deux
  séries : 0 repli dans les deux, et la même pensée moyenne (1 060 tokens au pilote, 1 179 sur
  l'ensemble).
- **Les mesures d'usage.** Ce sont des décisions corrélées dans une partie, d'où les permutations
  par seed. Les trajectoires divergent entre les bras : les situations ne sont que de même nature.
  L'hypothèse du § 6 (a) n'est pas vérifiée.
- **`moi` relu au pas suivant.** Les mesures du § 5 comparent `moi` à la position vraie au moment de
  l'écriture et au pas suivant. Elles ne disent pas si le récepteur a été trompé par l'écart.

## Suite

- **Le point plafonné intermédiaire n'est pas nécessaire.** C'est le schéma d'A2 sous un budget
  de 64 positions (spec A3, § Points ouverts). A3 ne perd pas de captures face à A2 : un point
  entre les deux n'en gagnerait pas davantage.
- **A4 peut viser `m = 32`** pour une comparaison à budget égal avec A3, et un point plus long pour
  A2. La question d'A4 n'est plus « le latent fait-il mieux que le texte ? » sur les captures, que
  le texte ne fait pas bouger. Elle devient : le latent porte-t-il, à budget égal, ce que le texte
  minimal perd (l'information négative, la tenaille) ?
- **La phase de capture reste le vrai chantier** si l'on veut se rapprocher de P2. C'est une
  question de décision, hors du périmètre du canal.
