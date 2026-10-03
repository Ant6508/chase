# Jalon 2, bras A2 — canal de message JSON non contraint, contre le témoin A1bis

**En bref.**
- **Captures : A2 ne fait pas mieux que son témoin.** A2 capture 18 fois sur 30, A1bis 19 fois
  (p = 1,00).
- **Confinement à T : A2 fait deux fois mieux, mais l'écart n'est pas significatif.** Il est de
  0,059 contre 0,118 (p = 0,41).
- **Le canal est pourtant utilisé, de façon sélective.**
  - **Quand le coéquipier voit la cible**, le récepteur la rejoint. Il fait un coup qui
    rapproche 92 % du temps, contre 63 % pour A1bis (p = 0,003).
  - **En exploration**, les deux poursuivants s'écartent l'un de l'autre. Ils se trouvent à
    3 cases ou moins 13 % du temps, contre 28 % pour A1bis (p = 0,001).
  - **L'information négative, en revanche, est ignorée.** Le schéma dit qu'un lieu absent de
    `candidates` a été vu vide. Le récepteur ne s'en sert pas pour écarter une issue : il prend
    une issue morte 26 % du temps, contre 16 % pour A1bis (p = 0,24).
- **Le prix est élevé.** A2 pense 5 fois plus par décision (2 528 tokens, contre 492), et 16 % de
  ses décisions sont relancées.

## 1. Campagnes

| | A2 | A1bis |
|---|---|---|
| Seeds | 0 à 29 | 0 à 29 |
| Dates (UTC) | seeds 0 à 7 (pilote) : du 2026-09-30, 19h40, au 2026-10-01, 05h00. Seeds 8 à 29 : du 2026-10-01, 05h02, au 2026-10-02, 01h07 | 2026-10-02, de 01h07 à 06h43 |
| Durée sur le pod | environ 29 h 30 | environ 5 h 40 |
| Fichiers | [`jalon2_a2/`](jalon2_a2/) | [`jalon2_a1bis/`](jalon2_a1bis/) |

- **Matériel.** Pod RunPod `otdj10kjtl7a4g` : A40 48 Go, Secure, CA-MTL-1, 0,49 $/h.
  - LM Studio avec le moteur CUDA, `google/gemma-4-12b` Q6_K.
  - `--context-length 65536 --parallel 8`, soit 8 créneaux de 8 192 tokens.
- **Code.** Le même déploiement que le pilote ([`jalon2_a2_pilote.md`](jalon2_a2_pilote.md)),
  code `540dd1d`. Les deux bras tournent sur la même pile.
- **Enchaînement.** Un script détaché sur le pod (`chain.sh`) a lancé A1bis à la fin d'A2.

```bash
campaign.sh start a2 --arm A2 --max-tokens 5600 --timeout 900 --first-seed 0 --episodes 30 \
    --concurrency 8 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --trace /workspace/results/a2/trace
campaign.sh start a1bis --arm A1bis  # mêmes options, --trace /workspace/results/a1bis/trace
```

Chaque dossier contient le journal (`journal.jsonl`), le tableau par seed (`table.md`), la
sortie brute (`run.log`) et les traces pas à pas (`trace/`). Le rapatriement a été vérifié par
empreintes MD5. Le pod est en pause depuis le 2026-10-02 à 06h50 UTC.

## 2. Tableau

Seeds 0 à 29, T = 60. Les IC sont à 95 %, approximation normale.

| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen | Pas jusqu'à capture (médiane) |
|---|---|---|---|---|---|
| R0 — aléatoire | 0 % | ± 0 % | 0,519 | 0,595 | — |
| R1 — gloutonne, sans communication | 63 % | ± 17 % | 0,072 | 0,296 | 34 |
| R2 — croyance fusionnée, délai 1 pas | 83 % | ± 13 % | 0,029 | 0,232 | 30 |
| P2 — décisions de R2, message A2 seul | 90 % | ± 11 % | 0,030 | 0,225 | 30 |
| A1v3 — LLM sans lieux nommés | 53 % | ± 18 % | 0,106 | 0,362 | 29 |
| A1bis — LLM avec lieux nommés, témoin | 63 % | ± 17 % | 0,118 | 0,352 | 34 |
| **A2 — LLM avec canal JSON** | **60 %** | ± 18 % | **0,059** | **0,318** | **31,5** |

**D'où viennent les chiffres.**
- **R1, R2 et P2** viennent de `python -m scripts.ceiling_a2 --episodes 30` avec le profil réduit.
- **R0** vient de `chase.runner.evaluate`.
- **A1v3** vient de [`jalon2_a1v3.md`](jalon2_a1v3.md). Son confinement moyen manquait au
  journal : il a été recalculé en rejouant les traces dans l'environnement.
- **Vérification.** Le même rejeu, appliqué à A1bis et à A2, redonne au milliardième près le
  confinement moyen de leurs journaux.

**La forme des échecs diffère d'un bras à l'autre.**
- **A2 perd la cible** (confinement à T supérieur à 0,3) dans 2 seeds seulement : 7 et 11.
  A1bis la perd dans 5 seeds : 9, 14, 15, 19 et 27.
- **A2 échoue plus souvent en finale.** Il la localise sans la capturer dans 10 seeds
  (confinement à T de 0,1 ou moins, sans capture), contre 6 pour A1bis.
- **A2 trouve donc mieux la cible, mais il ne la convertit pas mieux en capture.** Les écarts de
  confinement par seed ne sont pas significatifs :
  - à T : − 0,060 en moyenne, p = 0,41 ;
  - en moyenne sur l'épisode : − 0,034, p = 0,39.
  Ces deux tests sont des permutations de signe appariées par seed.

## 3. Comparaisons appariées

Test de McNemar exact, bilatéral, sur les seeds où une seule des deux politiques capture.

```bash
python -m scripts.compare_arms --episodes 30 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 \
    min_spawn_dist=6 --journal A1v3=results/jalon2_a1v3_local/journal.jsonl \
    --journal A1bis=results/jalon2_a1bis/journal.jsonl --journal A2=results/jalon2_a2/journal.jsonl \
    --pairs A2:A1bis A2:R2 A2:P2 A1bis:A1v3 R1:R2
```

| Paire | Captures | Seul le premier | Seul le second | p |
|---|---|---|---|---|
| **A2 contre A1bis** | 18 / 19 | 5 (10, 14, 15, 21, 27) | 6 (5, 7, 8, 11, 16, 20) | 1,00 |
| A2 contre R2 | 18 / 25 | 3 (12, 13, 27) | 10 (5, 6, 7, 8, 9, 11, 19, 20, 22, 23) | 0,09 |
| A2 contre P2 | 18 / 27 | 1 (27) | 10 (5, 6, 7, 8, 9, 11, 19, 20, 22, 23) | **0,01** |
| A1bis contre A1v3 | 19 / 16 | 4 (16, 17, 18, 24) | 1 (15) | 0,38 |
| R1 contre R2, pour contrôle | 19 / 25 | 2 (12, 13) | 8 (1, 6, 9, 10, 18, 19, 20, 26) | 0,11 |

- **Le canal ne change pas les captures (A2 contre A1bis).** Il y a 11 seeds discordantes,
  partagées presque à égalité.
- **Rien n'est tranché face à R2 (p = 0,09).** A2 reste en dessous.
- **A2 contre P2 : le protocole porte plus que ce qu'en fait le LLM.** Le même message, exploité
  par les décisions de R2, capture significativement plus que A2.
- **Le référentiel de lieux seul ne change rien de significatif (A1bis contre A1v3).** Les deux
  ne tournaient pas sur la même pile : Vulkan sur une carte AMD pour A1v3, CUDA pour A1bis.
- **La ligne R1 contre R2 redonne bien 19 / 25 avec p = 0,11.** Le profil passé à `--set` est
  donc le bon.

## 4. Le point de Pareto d'A2

| Positions par message | | Positions par épisode | | Confinement à T | Captures |
|---|---|---|---|---|---|
| moyenne | médiane (min. – max.) | moyenne | médiane | | |
| 138 | 167 (31 – 258) | 12 257 | 11 866 | 0,059 | 60 % |

C'est le premier point de la courbe qu'A3 balaiera. Il est cher : environ 12 000 positions
échangées par épisode. Face à A1bis, il n'apporte rien sur les captures. Il divise par deux le
confinement à T, mais dans la marge d'erreur (p = 0,41).

## 5. La qualité des messages

Ces mesures viennent d'un rejeu de chaque épisode. Les coups enregistrés sont rejoués dans
l'environnement, et la cible scriptée retombe sur les mêmes coups : à chaque pas, la position de
chacun est vérifiée contre la trace. La croyance individuelle de chaque poursuivant est
recalculée à chaque pas, comme `LLMPursuers` la tient. Le script est
[`jalon2_a2/msg_usage.py`](jalon2_a2/msg_usage.py) (voir § Réserves).

- **Messages valides.** 2 664 décisions sur 2 732 (97,5 %) portent un message valide. Les 68
  autres sont les replis.
- **Noms inconnus.** On en compte 75, répartis dans 36 messages (1,4 %). Ce sont des variantes
  de noms (`C1_a`, `C1e`), des rangs qui n'existent pas (`C10a.6`) et une direction (`South`).
- **`moi`.** Il est exact dans 94,9 % des messages, et dans le bon lieu dans 98,6 %.
- **`cible`.**
  - Quand l'émetteur voit la cible (445 messages), elle est exacte dans 100 % des cas.
  - Quand il ne la voit pas (2 219 messages), elle est inventée dans 0,6 % des cas (14 messages).
- **Complétude de `candidates`.** On mesure la part de la vraie masse de probabilité de
  l'émetteur couverte par les lieux listés, quand la cible est cachée.
  - En moyenne, elle vaut 97,0 %, et la médiane est à 100 %.
  - 65 % des messages sont complets (99,5 % de la masse ou plus).
  - 5,8 % couvrent moins de 90 % de la masse, et 1,4 % moins de 50 %.
  - Sur les 23,7 lieux candidats de l'émetteur, en moyenne, il en liste 21,5. Il en omet 2,8,
    surtout des lieux à moins de 1 %. Seulement 0,6 lieu listé est vide.
  - **Une omission n'est pas anodine.** Par le schéma, un lieu omis est déclaré vu vide.
- **Champs facultatifs.**
  - `je_couvre` n'est renseigné que dans 2,4 % des messages.
  - `intention` liste 2,3 lieux en moyenne.

**Les messages sont fidèles.** L'émetteur recopie correctement sa perception au grain des lieux.
La faiblesse est ailleurs : à la réception.

## 6. L'usage des messages, lu dans les traces

**Comment on compare.** A1bis ne reçoit pas de message. Pour lui, le savoir du coéquipier au pas
précédent est pris dans le message *mécanique* (`write_message` de P2) rédigé depuis sa vraie
croyance. Pour comparer à l'identique, A2 est mesuré avec le même message mécanique. Le message
réellement reçu donne les mêmes chiffres à quelques situations près (voir (a)).

**Comment on teste.** Les décisions d'un même épisode ne sont pas indépendantes. Les tests sont
donc des permutations appariées par seed (100 000 tirages) : sous H0, les comptes des deux bras
s'échangent à l'intérieur d'une seed.

### (a) Le récepteur évite-t-il les issues vidées par son coéquipier ?

**La définition.** La cible est cachée au récepteur.
- Une issue est **morte** quand elle mène à des candidates de sa croyance, mais à aucune une fois
  cette croyance intersectée avec ce que savait le coéquipier, propagé d'un pas.
- Une issue est **vivante** quand elle en garde.

Les situations retenues ont au moins une issue morte et une vivante.

| Ce que savait le coéquipier | A2 : vers une issue morte | A1bis : vers une issue morte | p |
|---|---|---|---|
| Il voyait la cible | 9,0 % (15 / 167) | 38,9 % (65 / 167) | **0,004** |
| Il ne la voyait pas : seuls ses lieux vus vides écartent une issue | 26,4 % (52 / 197) | 16,0 % (29 / 181) | 0,24 |

- **Quand le coéquipier ne voyait pas la cible, l'information négative est ignorée.** Dans ce cas,
  l'issue la plus probable pour le récepteur seul était morte dans 44 situations. A2 l'a suivie
  quand même 35 fois (79,5 %), A1bis 8 fois sur 12 (66,7 %).
- **Avec le message réellement reçu**, les chiffres sont identiques quand le coéquipier voyait la
  cible. Quand il ne la voyait pas, A2 prend une issue morte dans 25,1 % des cas (52 / 207).

**Exemple : l'information négative lue à contresens** (seed 9, pas 47, poursuivant 1).
- **La situation.** Le récepteur est en C1b.2. Sa perception donne NORD à 39 %, et SUD à 61 %
  (K3, C6 … C13).
- **Le message reçu :**
  `{"candidates":{"C14":100},"cible":null,"intention":["K6","C12","C14"],"je_couvre":null,"moi":"C5b.2"}`.
  - Par le schéma, cela veut dire que tous les autres lieux ont été vus vides, SUD compris.
  - L'intersection ne laisse donc que C14, au nord.
- **Sa pensée.** Il lit au contraire *« the teammate thinks the target is at C14 »*, puis
  *« if the teammate is going for C14 […] I should perhaps explore the SOUTH which has a higher
  probability in my current view »*.
- **Son coup.** Il joue SUD : *« The teammate is heading towards C14 (NORTH), so I will move SOUTH
  to explore the area with a higher probability »*.
- **Le contresens.** Le message est compris comme une *intention de couverture* (« lui va là,
  donc moi ailleurs »), et non comme un *constat d'absence*. On retrouve ce que montraient déjà
  les situations témoins ([`jalon2_a2_situations.md`](jalon2_a2_situations.md)) : le modèle lit
  le message, mais ne fait pas l'intersection.

### (b) Rejoint-il la cible quand seul son coéquipier la voit ?

**La situation.** Le coéquipier voyait la cible au pas précédent, et le récepteur ne la voyait
ni alors ni maintenant.

| | A2 | A1bis | p |
|---|---|---|---|
| Coups qui rapprochent de la case où la cible était vue | **91,6 %** (186 / 203) | 63,3 % (114 / 180) | **0,003** |

Dans les 203 situations d'A2, le message reçu portait bien `cible`.

**Exemple** (seed 6, pas 47, poursuivant 1).
- **La situation.** Le récepteur est en C10a.5. Sa perception donne SUD à 57 % et NORD à 43 %.
- **Le message reçu** porte `"cible":"C2b.4"` et `"moi":"C2a.1"`.
- **Sa pensée.** Elle note d'abord l'écart avec sa propre perception, puis tranche : *« C2b is in
  the NORTH direction from me »*.
- **Son coup.** Il joue NORD, contre sa propre issue la plus probable : *« The teammate saw the
  target at C2b.4 […] I am moving NORTH to get closer to that area and provide backup for a
  potential capture or to cut off escape routes »*.

### (c) Les deux poursuivants se séparent-ils en exploration ?

On ne retient que les pas où aucun des deux ne voit la cible. La distance est mesurée en chemin.

| | Pas retenus | Distance moyenne | Médiane | À 3 cases ou moins |
|---|---|---|---|---|
| **A2** | 1 019 | 17,3 | 17 | **12,6 %** |
| A1bis | 1 044 | 13,1 | 11 | 27,5 % |
| R1 | 918 | 19,0 | 21 | 16,3 % |
| R2 | 799 | 20,0 | 22 | 17,9 % |
| P2 | 762 | 20,0 | 22 | 12,9 % |

- **A2 contre A1bis : p = 0,001**, pour la part de pas à 3 cases ou moins.
- **Sans canal, les deux LLM se regroupent plus que R1.** Ils suivent la même heuristique de
  l'issue la plus probable, qui converge souvent.
- **Avec le canal, ils retrouvent l'écartement de P2.** C'est sans doute pour cela qu'A2 perd
  moins souvent la cible (§ 2).

**Exemple** (seed 8, pas 8, poursuivant 0).
- **La situation.** Le récepteur est en C7.1. Sa perception donne OUEST à 69 % et EST à 31 %.
- **Le message reçu** place le coéquipier en C3b.5, avec `"intention":["C3c","K4"]`.
- **Sa pensée.** *« 69 % of candidates are in the West, it's tempting to go West. But my teammate
  is already there. To coordinate effectively, I should perhaps head East »*.
- **Son coup.** Il joue EST : *« Mon coéquipier se dirige vers l'ouest et le centre (C3b vers
  K4), je vais donc me diriger vers l'est […] »*.

**La même logique de partage se retourne contre lui en (a).** En (a), le message est lu comme
une simple répartition des zones, même quand il dit que la zone « de l'autre côté » est vide.

### Les autres indicateurs de trace

Ils viennent de `python -m scripts.analyze_trace` sur les deux traces.

| | A1v3 | A1bis | A2 |
|---|---|---|---|
| Décisions | 2 778 | 2 714 | 2 732 |
| Replis | 2 | 0 | 68 (2,5 %), dont 12 à la seed 7 et 9 à la seed 27 |
| Immobile | 0 % | 0 % | 2 % (surtout les replis) |
| Allers-retours | 36 % | 30 % | **18 %** |
| Cible visible : coup qui la rapproche | 100 % | 100 % | 100 % |
| Cible cachée : vers l'issue la plus probable | 79 % | 81 % | 79 % |

**Les allers-retours baissent en A2, et c'est un effet du canal.** Le pilote laissait la question
ouverte : les lieux nommés ou le canal ? A1bis tranche, avec 30 %, contre 18 % pour A2.

## 7. Les coûts

| Par décision | A1v3 | A1bis | A2 |
|---|---|---|---|
| Tokens de prompt | 596 | 1 152 | 1 661 |
| Tokens de complétion (tentative acceptée) | 446 | 548 | 2 592 |
| dont pensée | — | 492 | **2 528** |
| Positions de message émises | — | — | 135 |
| Décisions relancées | 0,3 % | 0,1 % | **16,2 %** |
| Relances par décision | 0,004 | 0,001 | 0,219 |
| Latence moyenne | — | 52 s | 158 s |
| Temps mural médian d'un épisode | 3,1 h (local, 4 créneaux) | 1,3 h | 5,7 h |

**Le surcoût de pensée d'A2 est le prix de la sérialisation.** Face à A1bis, il s'élève à
2 036 tokens par décision, soit 5,1 fois plus. La pensée rédige, puis relit, la liste des lieux
et de leurs parts. Ce constat avait déjà été fait au diagnostic
([`jalon2_a2_budget.md`](jalon2_a2_budget.md)).

**Les relances ne sont pas comptées.** `completion_tokens` et `thinking_tokens` ne retiennent que
la tentative acceptée. A2 compte 665 tentatives ratées : ses 597 relances, plus la dernière
tentative de chacun des 68 replis. Elles viennent de trois sources :
- **627 tentatives coupées** à 5 600 tokens (`finish_reason=length`) ;
- **22 erreurs du serveur** : 21 « Engine protocol predict stream », plus une interruption ;
- **16 réponses invalides** :
  - 12 `candidates` qui n'est pas un objet lieu → nombre ;
  - 3 `je_couvre` manquant ;
  - 1 réponse sans appel d'outil.

Les seules coupures représentent au moins 3,5 millions de tokens de complétion non comptés,
contre 7,1 millions comptés. **Le vrai coût de complétion d'A2 est donc d'au moins 3 870 tokens
par décision**, plutôt que 2 592. Pour A1bis, les 4 coupures n'ajoutent que 8 tokens environ par
décision.

**Les latences et les temps muraux** suivent les mêmes réserves qu'en A1v3. Ils dépendent de la
charge du serveur : 8 parties en parallèle, puis moins en fin de campagne. A1v3 tournait en local,
sur une autre pile. Ils ne mesurent pas le coût du bras seul.

**En argent.** Environ 14,40 $ de pod pour A2 (29 h 30) et 2,70 $ pour A1bis (5 h 40), à
0,49 $/h. Au total, avec l'installation, les situations témoins et le pilote, le pod a tourné
35 h 45 (17,50 $).

## 8. La revendication

**Le schéma n'est pas un langage émergent.** C'est un protocole conçu par nous (§ Le message de
la [spec](../docs/superpowers/specs/2026-09-29-jalon2-a2-design.md)) :
- `moi`, `cible`, `candidates`, `intention`, `je_couvre` ;
- un lieu absent de `candidates` est vu vide.

A2 mesure donc « le texte à charge utile fixée » (SPEC §3) : ce qu'un LLM fait d'un canal
textuel dont le contenu est prescrit, mais dont l'usage ne l'est pas. Le prompt ne dit pas
comment exploiter le message reçu, et c'est voulu.

**Ce que dit ce jalon est donc limité.** Avec ce protocole et ce prompt, gemma-4-12b :
- exploite l'information **positive** du message, c'est-à-dire la cible vue et la position du
  coéquipier ;
- n'exploite pas l'information **négative**, c'est-à-dire les lieux vus vides.

Il ne dit rien d'un canal dont le schéma serait libre.

## 9. Les réserves

- **La taille de l'échantillon.** 30 seeds donnent environ ± 18 points sur les captures. Un effet
  de 10 points du canal serait invisible.
- **La reproductibilité.** À température 0, le serveur n'est pas déterministe quand plusieurs
  créneaux tournent en parallèle (8 ici). Le rejeu d'A1v3 l'avait montré sous 4 créneaux
  ([`jalon2_a1v3.md`](jalon2_a1v3.md)).
- **Les mesures d'usage.**
  - Elles portent sur des décisions corrélées à l'intérieur d'un épisode, d'où les permutations
    par seed.
  - Les trajectoires divergent entre les bras : les situations comparées ne sont donc pas les
    mêmes, seulement de même nature.
  - Le seuil de 3 cases, en (c), est arbitraire. La distance moyenne va dans le même sens.
- **La référence mécanique d'A1bis est contrefactuelle.** C'est ce que son coéquipier *aurait pu*
  lui dire, rédigé avec les arrondis de P2.
- **Le script d'analyse.** Le rejeu a été écrit pour ce rapport, et il est joint aux résultats
  plutôt qu'à `scripts/`, sans tests : `PYTHONPATH=. python results/jalon2_a2/msg_usage.py`,
  depuis la racine du dépôt. Ses chiffres se recoupent avec ceux du journal : positions conformes
  à la trace à chaque pas, et confinement moyen identique.
- **Les replis d'A2** (des `STAY`) se concentrent sur quelques seeds difficiles : 12 à la seed 7,
  9 à la seed 27, 7 à la seed 9. Leur pensée s'emballe au-delà du plafond de 5 600 tokens.

## Suite

**Le déclencheur prévu est atteint.** La spec (§ Points ouverts) prévoyait la variante A2' si A2
ne faisait pas mieux qu'A1bis, et c'est le cas.

**Les traces disent précisément ce qui manque.** Ce n'est pas la lecture du message, qui est
fidèle et utilisée pour la cible et pour se répartir. C'est **l'intersection** : déduire d'un
lieu absent des `candidates` du coéquipier que la cible n'y est pas.

**A2' la rendrait explicite dans le prompt.** Le canal et le schéma resteraient les mêmes. On
dirait, par exemple : « un lieu absent des `candidates` de ton coéquipier a été vu vide : la
cible n'y est pas, sauf si elle a pu y revenir depuis ».

**Ce qui le mesurerait.**
- La ligne « lieux vus vides seulement » du § 6 (a) doit passer sous A1bis.
- A2' doit se rapprocher de P2, qui est le plafond de ce protocole.

**Fait depuis** : [`jalon2_a2p_test.md`](jalon2_a2p_test.md), puis
[`jalon2_a2p.md`](jalon2_a2p.md). La première condition est remplie, la seconde ne l'est pas.
