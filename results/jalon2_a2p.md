# Jalon 2, bras A2' — canal JSON avec consigne d'usage, contre A2 et A1bis

**En bref.**
- **Captures : A2' ne fait pas mieux.** Il capture 15 fois sur 30, contre 18 pour A2 (p = 0,55) et
  19 pour A1bis (p = 0,39). Il reste sous R2 (p = 0,01) et sous P2 (p < 0,001).
- **La consigne fait pourtant ce pour quoi elle a été écrite.**
  - **Les lieux vus vides sont enfin écartés.** Quand seuls ces lieux écartent une issue, A2'
    prend une issue morte dans 4,4 % des cas, contre 26,4 % pour A2 (p < 0,001).
  - **La cible vue par le coéquipier est rejointe** 99 % du temps, contre 92 % pour A2
    (p = 0,02).
  - **Les poursuivants s'écartent encore mieux en exploration.** Ils sont à 3 cases ou moins
    9 % du temps, contre 13 % pour A2 (p = 0,04). La crainte du test, un regroupement, est
    écartée.
- **Deux choses absorbent ce gain.**
  1. **Le goulot est la phase de capture, et le message n'y change rien.** Tous les bras LLM
     laissent à la cible un territoire d'environ 21 à 26 cases, contre 11 à 12 pour R2 et P2. Ils
     la poursuivent, mais la prennent rarement en tenaille. A2' localise la cible sans la
     capturer dans 11 seeds.
  2. **A2' écrit des messages faux.** Dans 22,8 % de ses messages, la vraie case de la cible est
     exclue, contre 4,6 % pour A2.
     - Il recopie en `cible` la cible que son coéquipier voyait au pas précédent, alors que le
       schéma la réserve à ce qu'on voit (14,4 % des messages).
     - Parfois, cette cible fait l'aller-retour entre les deux poursuivants : 75 échos, dont un
       de 9 pas.
- **Le coût est celui d'A2** : 2 435 tokens de pensée par décision, et 16 % de décisions
  relancées.

## 1. Campagne

| | A2' |
|---|---|
| Seeds | 0 à 29 |
| Dates (UTC) | du 2026-10-02, 11h27, au 2026-10-03, 14h27 |
| Durée sur le pod | 27 h |
| Fichiers | [`jalon2_a2p/`](jalon2_a2p/) |

- **Le bras.** `A2p` : le prompt d'A2, plus `USAGE_PARAGRAPH` (`chase/llm/prompts.py`, commit
  `37960bf`), placé juste avant la consigne de réponse. Le paragraphe est cité au
  [rapport du test](jalon2_a2p_test.md). L'empreinte du prompt est `8831bc5e2788ad96`. Celle
  d'A2 n'a pas changé.
- **Le matériel.** Pod RunPod `14rh3c8qgzdpmi` : A40 48 Go, Secure, EU-SE-1, 0,49 $/h.
  - LM Studio avec le moteur CUDA (llama.cpp 2.41.0), `google/gemma-4-12b` Q6_K, 13 913 Mo de
    VRAM.
  - `--context-length 65536 --parallel 8`.
  - C'est la même pile que pour A2 et A1bis, mais sur une autre machine.
- **Le code.** Commit `aae090f`.
- **Un faux départ.** Une première campagne, lancée le 2026-10-02 à 09h26 UTC sur un pod de
  CA-MTL-1, s'est bloquée vers 09h40 : le disque réseau `/workspace` avait gelé. Le pod a été
  supprimé, et la campagne relancée de zéro sur le pod de Suède. Aucune partie du faux départ
  n'est comptée.

```bash
campaign.sh start a2p --arm A2p --max-tokens 5600 --timeout 900 --first-seed 0 --episodes 30 \
    --concurrency 8 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --trace /workspace/results/a2p/trace
```

**Le contenu du dossier.** Comme pour A2 : `journal.jsonl`, `table.md`, `run.log` et `trace/`.
Le rapatriement a été vérifié par empreinte MD5. Le pod est en pause depuis le 2026-10-03 à
14h29 UTC.

## 2. Tableau

Seeds 0 à 29, T = 60. Les autres lignes viennent du [rapport A2](jalon2_a2.md) (§ 2).

| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|---|
| R1 — gloutonne, sans communication | 63 % | ± 17 % | 0,072 | 0,296 | 34 |
| R2 — croyance fusionnée, délai 1 pas | 83 % | ± 13 % | 0,029 | 0,232 | 30 |
| P2 — décisions de R2, message A2 seul | 90 % | ± 11 % | 0,030 | 0,225 | 30 |
| A1bis — LLM avec lieux nommés, témoin | 63 % | ± 17 % | 0,118 | 0,352 | 34 |
| A2 — LLM avec canal JSON | 60 % | ± 18 % | 0,059 | 0,318 | 31,5 |
| **A2' — canal JSON et consigne d'usage** | **50 %** | ± 18 % | **0,085** | **0,323** | **32** |

**La forme des échecs.**
- **A2' perd la cible dans 3 seeds** (3, 7 et 28) : aucun des deux poursuivants ne la voit de
  toute la partie.
  - Dans les trois bras, « perdre la cible » (confinement à T supérieur à 0,3) veut dire ne
    jamais la trouver. A2 la perd dans 2 seeds (7 et 11), A1bis dans 5.
  - Ces trois seeds sont aussi celles où A2' s'enlise : 44 à 52 % d'allers-retours, 5 à 10
    replis, et plus de 3 100 tokens de pensée par décision.
- **A2' échoue surtout en finale.** Il localise la cible sans la capturer dans 11 seeds
  (confinement à T de 0,1 ou moins, sans capture), contre 10 pour A2 et 6 pour A1bis.
  - Le cas extrême est la seed 26 : la cible est vue à 106 des 120 décisions, et elle n'est
    jamais prise.
- **Aucun écart de confinement n'est significatif.** Ces tests sont des permutations de signe
  appariées par seed.
  - Contre A2 : + 0,027 à T (p = 0,50), et + 0,005 en moyenne sur l'épisode (p = 0,91).
  - Contre A1bis : − 0,033 à T (p = 0,65), et − 0,029 en moyenne (p = 0,53).

## 3. Comparaisons appariées

McNemar exact, bilatéral, sur les seeds où une seule des deux politiques capture.

```bash
python -m scripts.compare_arms --episodes 30 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 \
    min_spawn_dist=6 --journal A1bis=results/jalon2_a1bis/journal.jsonl \
    --journal A2=results/jalon2_a2/journal.jsonl --journal A2p=results/jalon2_a2p/journal.jsonl \
    --pairs A2p:A2 A2p:A1bis A2p:R2 A2p:P2 A2:A1bis R1:R2
```

| Paire | Captures | Seul le premier | Seul le second | p |
|---|---|---|---|---|
| **A2' contre A2** | 15 / 18 | 4 (6, 8, 9, 11) | 7 (3, 12, 14, 21, 26, 27, 28) | 0,55 |
| A2' contre A1bis | 15 / 19 | 4 (6, 9, 10, 15) | 8 (3, 5, 7, 12, 16, 20, 26, 28) | 0,39 |
| A2' contre R2 | 15 / 25 | 1 (13) | 11 (3, 5, 7, 14, 19, 20, 21, 22, 23, 26, 28) | **0,01** |
| A2' contre P2 | 15 / 27 | 0 | 12 (3, 5, 7, 12, 14, 19, 20, 21, 22, 23, 26, 28) | **< 0,001** |
| A2 contre A1bis, pour contrôle | 18 / 19 | 5 | 6 | 1,00 |
| R1 contre R2, pour contrôle | 19 / 25 | 2 | 8 | 0,11 |

- **Les deux lignes de contrôle redonnent celles du rapport A2.** Les journaux et le profil sont
  donc les bons.
- **A2' capture 3 fois de moins qu'A2, mais avec 11 seeds discordantes.** C'est compatible avec
  le hasard, et avec le non-déterminisme du serveur.
- **P2 capture dans toutes les seeds où A2' capture.** Le protocole porte bien plus que ce que
  le LLM en fait, même quand on lui dit comment s'en servir.

## 4. L'usage du message

**La méthode est celle du [rapport A2](jalon2_a2.md) (§ 6).**
- Les épisodes sont rejoués dans l'environnement, et les croyances individuelles recalculées à
  chaque pas.
- Le savoir du coéquipier est pris dans le message *mécanique*, rédigé depuis sa vraie
  croyance. On évite ainsi de juger le récepteur sur un message faux (voir § 5.2).
- Les tests sont des permutations appariées par seed (100 000 tirages).

**Le script** est celui du rapport A2, généralisé à plusieurs bras :

```bash
PYTHONPATH=. python results/jalon2_a2/msg_usage.py A2p A2 A1bis
```

| Mesure | A2' | A2 | A1bis | A2' contre A2 | A2' contre A1bis |
|---|---|---|---|---|---|
| (a) seuls les lieux vus vides écartent une issue : coups vers une issue morte | **4,4 %** (11 / 252) | 26,4 % (52 / 197) | 16,0 % (29 / 181) | **p < 0,001** | p = 0,03 |
| (a) même cas, quand son issue la plus probable est morte : il la suit quand même | **6,8 %** (3 / 44) | 79,5 % (35 / 44) | 66,7 % (8 / 12) | **p < 0,001** | p = 0,004 |
| (a) le coéquipier voyait la cible : coups vers une issue morte | 1,3 % (2 / 150) | 9,0 % (15 / 167) | 38,9 % (65 / 167) | p = 0,02 | p < 0,001 |
| (b) seul le coéquipier voit la cible : coups qui rapprochent | **98,9 %** (177 / 179) | 91,6 % (186 / 203) | 63,3 % (114 / 180) | p = 0,02 | p < 0,001 |
| (c) exploration : pas à 3 cases ou moins l'un de l'autre | **9,0 %** (95 / 1 060) | 12,6 % (128 / 1 019) | 27,5 % (287 / 1 044) | p = 0,04 | p < 0,001 |

- **(a) La règle d'intersection est appliquée.** Ce que le test laissait voir sur 44 situations
  isolées se confirme en campagne, sur toutes les décisions. Avec la consigne, l'information
  négative passe sous A1bis, comme le demandait le rapport A2.
- **(b) La cible vue par le coéquipier est rejointe presque toujours.**
- **(c) La réserve du test ne se vérifie pas.** On craignait qu'A2' « rejoigne » son coéquipier
  plutôt qu'intersecter. En exploration, il s'en écarte au contraire un peu plus qu'A2.
  - La distance moyenne est de 17,8 cases, contre 17,3 pour A2.
  - Pour comparaison, R1 est à 16,3 % de pas à 3 cases ou moins, R2 à 17,9 % et P2 à 12,9 %.
- **Le message réellement reçu donne la même lecture.** Avec lui, A2' prend une issue morte
  dans 4,9 % des cas (18 / 368).

**Les autres indicateurs de trace** viennent de `python -m scripts.analyze_trace` :

| | A1bis | A2 | A2' |
|---|---|---|---|
| Décisions | 2 714 | 2 732 | 2 822 |
| Replis | 0 | 68 (2,5 %) | 80 (2,8 %), dont 10 à la seed 7 et 8 à la seed 28 |
| Immobile | 0 % | 2 % | 3 % |
| Allers-retours | 30 % | 18 % | 19 % |
| Cible visible : coup qui la rapproche | 100 % | 100 % | 100 % |
| Cible cachée : vers l'issue la plus probable pour lui seul | 81 % | 79 % | **74 %** |

A2' quitte plus souvent sa propre issue la plus probable : c'est l'effet attendu de (a).

## 5. Pourquoi pas plus de captures

### 5.1 Un gain réel, mais petit en nombre de décisions

**Les situations où la règle d'intersection change le coup sont rares.**
- Les situations de (a) « lieux vus vides » ne représentent que 9 % des décisions (252 sur
  2 822).
- Celles où l'issue la plus probable est morte n'en représentent que 1,6 % (44).

**Ordre de grandeur.** Au taux d'A2, A2' aurait pris environ 66 issues mortes sur ses 252
situations. Il en a pris 11. Cela fait environ 55 mauvais coups évités en 30 parties, soit moins
de 2 par partie, chacun coûtant quelques pas.

**C'est trop peu pour se lire dans les captures.** À 30 seeds, il faudrait un écart d'une
vingtaine de points pour qu'il soit visible. Il n'avance pas non plus le premier repérage de la
cible : sur les 26 seeds où A2 et A2' la trouvent, l'écart médian est nul. A2' la trouve plus
tôt dans 9 seeds, plus tard dans 12.

### 5.2 Un effet de bord : A2' écrit des messages faux

**La consigne ne parle que de la lecture du message, mais elle a changé son écriture.**

Les mesures ci-dessous portent sur les messages écrits quand l'émetteur ne voit pas la cible. Un
message est **faux** quand il exclut la vraie case de la cible : elle n'est ni dans un lieu de
`candidates`, ni égale à `cible`.

| Messages, cible cachée à l'émetteur | A2' | A2 |
|---|---|---|
| Messages | 2 258 | 2 219 |
| **Faux** | **22,8 %** (514) | 4,6 % (103) |
| `cible` renseignée sans voir la cible | **14,4 %** (325) | 0,6 % (14) |
| … dont recopie de la `cible` reçue | 99,4 % (323) | 100 % (14) |
| … dont faux | 91,6 % (296) | 78,6 % (11) |
| … dont en écho : le coéquipier ne la voyait pas non plus | 149, dont 148 faux | 0 |
| `candidates` seuls : faux | 11,2 % (217 / 1 933) | 4,2 % (92 / 2 205) |
| … dont simple sous-ensemble des `candidates` reçus | 139 | 21 |
| Part de sa propre masse de probabilité couverte (moyenne) | 75,8 % | 97,0 % |

**Les erreurs viennent de deux sources.**
- **Le relais de la cible.** Le récepteur recopie en `cible` la case où son coéquipier voyait la
  cible au pas précédent.
  - Elle se trouve alors à 1 case de la vraie position en médiane (1,9 en moyenne).
  - Par le schéma, ce message déclare pourtant tous les autres lieux vus vides. Or la cible a
    déjà bougé : le message est faux 9 fois sur 10.
  - Le prompt est clair sur ce point : `cible` est « la case de la cible si tu la vois, sinon
    null ».
- **La fusion sans propagation.** A2' liste souvent comme `candidates` ceux de son coéquipier
  (139 des 217 messages faux sans `cible`). Il fusionne donc sa perception et le message reçu,
  comme le proposait le point ouvert « Relais » de la spec, mais sans tenir compte du pas
  écoulé : la cible a pu passer dans un lieu voisin.

**Ces deux erreurs se nourrissent l'une l'autre.** La consigne dit au récepteur : « S'il voit la
cible, rejoins-la ». Une cible relayée passe donc pour une cible vue. Elle peut alors faire
l'aller-retour entre les deux poursuivants, sans qu'aucun ne la voie.
- **75 échos** ont été relevés, sur 158 pas en tout.
- La plupart durent 1 à 3 pas, et le plus long 9.

**Exemple : l'écho de la seed 22**, du pas 50 au pas 59.
- **Pas 50.** P0 voit la cible en C9h.1, et l'écrit.
- **Pas 51.** P1 recopie `"cible":"C9h.1"` : *« Mon coéquipier a vu la cible à C9h.1 et se
  dirige vers elle »*. P0, lui, a perdu la cible de vue et écrit un message juste : ses
  `candidates` ne gardent que C9g, où elle se trouve.
- **Pas 52.** P0 reçoit le relais de P1, c'est-à-dire sa propre observation du pas 50, qui lui
  revient. Sa perception ne laisse que 3 cases (C9h.1, C9g.4, C9g.3). Il lit pourtant
  *« The teammate says the target is at C9h.1 (which they saw) »*, et il relaie à son tour.
- **Pas 53 à 59.** La case C9h.1 fait l'aller-retour à chaque pas, pendant que la cible fuit par
  C9g, C9f, puis C9e.
- **Pas 56.** P0 se trouve lui-même dans C9h, et ne voit rien. Il doute (*« Wait, the teammate
  says they see the target at C9h.1? Let me re-read. »*), puis écrit encore
  `"cible":"C9h.1"`.
- **Pas 60.** La partie se termine sans capture.

**Ces messages faux n'expliquent pas, à eux seuls, les échecs.**
- Les seeds les plus touchées sont deux échecs : la seed 20 (51 messages faux sur 113) et la
  seed 23 (43 sur 90). Mais la seed 2 (24 sur 42) est une capture.
- Les trois seeds perdues (3, 7 et 28) en comptent peu : 6 à 12 sur plus de 110.
- Les mesures de § 4 sont prises avec le message mécanique. Elles disent ce que fait le
  récepteur d'un message juste. En campagne, une partie des messages reçus sont faux.

### 5.3 Le goulot : la phase de capture

On ne retient ici que les pas où au moins un poursuivant voit la cible.
- **Le territoire** de la cible est le nombre de cases qu'elle atteint strictement avant les deux
  poursuivants.
- **Une tenaille** est un pas où la cible se trouve sur un plus court chemin entre eux.

R2 joue en phase de capture le pas qui réduit ce territoire, ce qui produit la tenaille (voir
`chase/policies.py`).

| | Pas | Territoire moyen | Médiane | En tenaille |
|---|---|---|---|---|
| R1 | 408 | 20,7 | 13,5 | 27,7 % |
| **R2** | 323 | **11,7** | **5** | **31,6 %** |
| **P2** | 338 | **10,6** | **4** | 28,1 % |
| A1bis | 313 | 20,7 | 18 | 24,6 % |
| A2 | 347 | 22,0 | 19 | 21,0 % |
| **A2'** | 351 | **26,2** | **20** | **17,4 %** |

- **Les trois bras LLM laissent à la cible environ deux fois plus de territoire que R2 et P2.**
  - Ils la poursuivent par le plus court chemin : le coup qui rapproche, 100 % du temps quand ils
    la voient.
  - Ils lui coupent rarement la route. Or, menacée, la cible se réfugie sur la boucle du
    labyrinthe (c'est sa règle de fuite) et y tourne devant eux.
  - Le canal n'y change rien. `moi` est pourtant présenté au prompt comme ce qui sert à
    « préparer une prise en tenaille », et `je_couvre` n'est renseigné que dans 1,1 % des
    messages d'A2'.
- **A2' est le moins bon des trois, mais sans écart significatif.** Contre A2 : tenaille
  p = 0,48, et territoire p = 0,42.
  - La consigne « S'il voit la cible, rejoins-la » pousse à converger sur la cible plutôt qu'à la
    couper, et les relais de § 5.2 attirent les deux poursuivants vers une case périmée.
  - C'est une hypothèse : les chiffres vont dans ce sens, sans la prouver.

**C'est là que se joue l'écart avec P2.** En exploration, A2' s'écarte autant que P2
(9,0 % contre 12,9 % de pas à 3 cases ou moins), et il évite maintenant les issues mortes. Mais
une fois la cible trouvée, il ne sait pas la prendre en tenaille. Aucune consigne d'usage du
message ne peut corriger cela.

## 6. Les coûts

| Par décision | A1bis | A2 | A2' |
|---|---|---|---|
| Tokens de prompt | 1 152 | 1 661 | 1 789 |
| Tokens de complétion (tentative acceptée) | 548 | 2 592 | 2 459 |
| dont pensée | 492 | 2 528 | 2 435 |
| Positions de message émises | — | 135 | 116 |
| Décisions relancées | 0,1 % | 16,2 % | 15,9 % |
| Relances par décision | 0,001 | 0,219 | 0,224 |
| Latence moyenne | 52 s | 158 s | 155 s |
| Temps mural médian d'un épisode | 1,3 h | 5,7 h | 5,6 h |

- **La consigne ne coûte presque rien.** Le prompt compte 128 tokens de plus. La pensée est
  même un peu plus courte qu'en A2 : les messages sont plus courts (116 positions contre 135),
  en partie parce qu'ils sont incomplets (§ 5.2).
- **Les relances ne sont pas comptées.** A2' compte 712 tentatives ratées :
  - 681 tentatives coupées à 5 600 tokens ;
  - 19 erreurs du serveur ;
  - 12 réponses invalides, dont 10 `candidates` qui n'est pas un objet lieu → nombre.
- **En argent.** Environ 13,30 $ de pod pour la campagne (27 h 10 de pod à 0,49 $/h). S'y
  ajoutent environ 1 $ pour le faux départ, et 0,60 $ pour le test.

## 7. Ce que dit A2'

**Le rapport A2 laissait deux explications ouvertes** (spec, § Points ouverts) : soit le canal
ne porte pas l'information utile, soit un modèle de 12B ne l'exploite pas spontanément.

**A2' tranche en faveur de la seconde, avec une limite.**
- **gemma-4-12b applique la règle d'intersection quand on la lui donne.** Il le fait sur toutes
  les décisions d'une campagne, pas seulement sur des situations isolées.
- **Il garde les usages positifs du canal**, et il les renforce : rejoindre la cible vue, et
  s'écarter en exploration.
- **Mais il l'applique aussi là où on ne le lui demandait pas.** Il reporte la vue de son
  coéquipier dans son propre message, sans tenir compte du pas écoulé. Il relaie en `cible` ce
  qu'il n'a pas vu. Le protocole perd alors sa propriété la plus utile : un message dit ce que
  son auteur a perçu.

**Pour les captures, la conclusion est négative.** Exploiter l'information négative ne change
pas les captures, car le goulot est ailleurs : la phase de capture. Là, aucun bras LLM n'approche
R2, et le canal n'y aide pas.

**Le résultat reste limité à ce protocole, ce prompt et ce modèle**, comme pour A2 (§ 8 du
rapport A2).

## 8. Les réserves

- **La taille de l'échantillon.** Elle est la même que pour A2 : 30 seeds donnent environ
  ± 18 points sur les captures. Une baisse de 3 captures n'est pas interprétable.
- **La machine.** A2' a tourné sur une autre machine qu'A2 et A1bis. La pile est la même
  (image, moteur, quantification, réglages), mais le non-déterminisme du serveur sous 8
  créneaux joue entre machines comme entre exécutions.
- **Les mesures d'usage et de capture.**
  - Ce sont des décisions corrélées à l'intérieur d'une partie, d'où les permutations par seed.
  - Les trajectoires divergent entre les bras : les situations ne sont que de même nature.
- **Les deux nouvelles mesures** ont été ajoutées pour ce rapport, sans tests : la véracité des
  messages et la phase de capture, dans `msg_usage.py`.
  - Le territoire et la tenaille sont des indicateurs géométriques simples. Ils ne tiennent pas
    compte de ce que chaque poursuivant sait.
  - La décomposition des messages faux (relais, écho, sous-ensemble du reçu) vient de
    [`jalon2_a2p/msg_truth.py`](jalon2_a2p/msg_truth.py), et les suites d'écho de
    [`jalon2_a2p/echo_runs.py`](jalon2_a2p/echo_runs.py). Les deux se lancent depuis la racine
    du dépôt, avec `PYTHONPATH=.`.
- **L'appariement des échos.** Un relais est compté « en écho » quand le coéquipier ne voyait
  pas la cible au pas où il a écrit le message recopié.

## Suite

**Raffiner le prompt d'usage n'apporterait pas grand-chose.** A2' montre que le modèle suit une consigne d'usage du
message. Mais l'information négative, une fois exploitée, ne change pas les captures. Corriger
l'écriture (« ne mets en `cible` que ce que tu vois ») rendrait les messages justes, sans toucher
au goulot.

**L'ordre convenu reste le bon : A3 minimale, puis A4.**
- **A3** ne garde que `moi`, `cible` et `intention`. Elle mesure ce qui reste des usages
  positifs (§ 4 (b) et (c)) sans `candidates`, c'est-à-dire sans le coût de la sérialisation ni
  les omissions.
- **A2' dit ce qu'A3 risque de perdre : peu de captures.** Ce que `candidates` apporte (a) ne
  se lit pas dans les captures.
- **Le relais de `cible` est à surveiller en A3.** Le champ y restera, et la tentation de
  recopier la cible du coéquipier aussi, sans consigne d'usage.

**La phase de capture est le vrai chantier si l'on veut se rapprocher de P2.** C'est une question
de décision, pas de canal. Elle concerne aussi le choix du coup cible d'A4 : celui de R2 ou de P2
porte la tenaille, celui d'un LLM ne la porte pas.
