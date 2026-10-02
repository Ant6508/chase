# Test d'A2' sur 60 situations tirées de la campagne A2

**En bref.** Quand on lui demande explicitement d'écarter les lieux que son coéquipier a vus
vides, gemma-4-12b le fait.
- **L'issue morte.** Sur les 44 situations où l'issue la plus probable pour l'agent seul était
  morte d'après le message reçu, A2 rejoué la prend dans 77 % des cas, A2' dans 26 %
  (McNemar p < 0,001).
- **Le témoin.** A2' ne perd rien sur les situations où le coéquipier voit la cible.
- **Le coût.** Il ne pense pas plus longtemps.
- **La décision.** Les trois conditions de la règle fixée avant le test sont réunies (spec
  A2, § Points ouverts) : une campagne A2' est défendable.

## Protocole

- **Le bras A2' (`A2p`).** Il reprend le prompt d'A2, et y ajoute `USAGE_PARAGRAPH`
  (`chase/llm/prompts.py`, commit `37960bf`) juste avant la consigne de réponse. Ce paragraphe
  demande :
  - d'écarter les lieux absents des `candidates` du coéquipier ;
  - de préférer une direction qui mène à des lieux candidats des deux côtés ;
  - de rejoindre la cible s'il la voit ;
  - sinon, de se répartir la recherche.
- **Les situations.** Elles sont tirées par `python -m scripts.situations_a2p select`, dans les
  traces de la campagne A2 ([`jalon2_a2.md`](jalon2_a2.md), § 6). Ce sont des décisions sans
  repli, cible cachée au récepteur, avec le prompt utilisateur exact.
  - **Les 44 situations `vide_morte`.** Le coéquipier ne voyait pas la cible, et l'issue la
    plus probable pour le récepteur seul est morte une fois sa croyance intersectée avec le
    message reçu, propagé d'un pas. Dans la campagne, A2 y avait pris l'issue morte 35 fois.
  - **16 situations `cible_vue`**, tirées au hasard. Le coéquipier voyait la cible au pas
    précédent, le récepteur non.
- **Le jeu.** Chaque situation est jouée avec le prompt d'A2, puis avec celui d'A2', en appels
  entrelacés. Il a eu lieu le 2026-10-02, de 08h03 à environ 09h10 UTC.
  - Pod `otdj10kjtl7a4g` (A40), LM Studio avec le moteur CUDA, 8 appels en parallèle.
  - `--max-tokens 5600 --timeout 900`, comme la campagne.
  - Coût : 1 h 15 de pod, environ 0,60 $.
- **Les fichiers.** Dans [`jalon2_a2p_test/`](jalon2_a2p_test/) : `situations.jsonl`,
  `replay.jsonl` (une ligne par appel, pensée comprise) et `run.log`. Le bilan se refait avec
  `python -m scripts.situations_a2p report`.

## Résultats

| Famille | Bras | Mesure (hors replis) | Replis | Tentatives coupées | Pensée moyenne |
|---|---|---|---|---|---|
| vide_morte | A2 rejoué | vers une issue morte : 77 % (34/44) | 0 | 6 | 3 050 |
| vide_morte | **A2'** | vers une issue morte : **26 %** (11/43) | 1 | 8 | 2 689 |
| cible_vue | A2 rejoué | rapproche de la cible vue : 81 % (13/16) | 0 | 0 | 2 671 |
| cible_vue | **A2'** | rapproche de la cible vue : **100 %** (16/16) | 0 | 0 | 1 318 |

**Comparaison appariée**, hors replis (McNemar exact, bilatéral) :
- **vide_morte.** A2 seul prend l'issue morte 23 fois, A2' seul 1 fois. Les deux la prennent
  10 fois, aucun des deux 9 fois. p < 0,001.
- **cible_vue.** A2' seul rapproche 3 fois, A2 seul jamais, et les deux 13 fois. p = 0,25.

**La correction n'est pas portée par une seule seed.** Issues mortes prises, A2 rejoué puis A2' :
- seed 9 : 12/16, puis 4/16 ;
- seed 20 : 10/10, puis 5/10 ;
- seed 14 : 4/4, puis 1/3 ;
- seed 29 : 2/3, puis 0/3 ;
- seed 6 : 2/2, puis 0/2 ;
- seed 27 : 2/2, puis 1/2.

**Le rejeu d'A2 est fidèle à la campagne.** Il donne le même coup qu'elle dans 78 % des
situations (47/60), dans 35 des 44 situations `vide_morte`. Le serveur n'est pas déterministe sous
charge parallèle.

## La règle de décision, condition par condition

1. **Satisfaite.** Sur `vide_morte`, A2' prend une issue morte dans 26 % des cas, sous la borne
   de 40 %, et significativement moins qu'A2 rejoué.
2. **Satisfaite.** Sur `cible_vue`, A2' ne perd rien : il rapproche dans 16 situations sur 16,
   contre 13 pour A2.
3. **Satisfaite, avec une réserve.**
   - Les tentatives coupées passent de 6 à 8 sur `vide_morte`, et restent à 0 sur `cible_vue` :
     elles ne doublent pas.
   - Il y a 1 repli d'A2' sur 60 appels, contre 0 pour A2. Avec une base nulle, le « doublement »
     n'a pas de sens. Un repli sur 60 (1,7 %) reste sous le taux de la campagne A2 (2,5 %).
   - La pensée est même plus courte : − 12 % sur `vide_morte`, et environ deux fois moins sur
     `cible_vue`.

## Ce que montre une décision corrigée

**L'exemple suivant est celui de la seed 9, au pas 47, cité au § 6 (a) du rapport A2.**
- **Le message reçu.** `{"candidates":{"C14":100},"cible":null,"intention":["K6","C12","C14"],…}`.
  Tout sauf C14 a donc été vu vide, y compris le SUD.
- **A2 rejoué** prend encore le SUD : *« I will explore the South area which has a higher
  probability (61%) according to my perception to cover both possibilities »*.
- **A2'** prend le NORD : *« Le coéquipier indique que la cible est à C14. Je me déplace vers le
  NORD pour rejoindre son itinéraire […] »*.

**Il faut noter la forme du raisonnement.** A2' ne dit pas « le sud est vide ». Il lit
`{"C14":100}` comme « la cible est à C14 », puis il y va. Le coup est le bon, mais une partie
du gain pourrait venir d'une tendance à *rejoindre* le coéquipier plutôt que d'une vraie
intersection. **Le risque, en campagne :** A2' pourrait perdre une partie de l'écartement en
exploration, qui était un acquis d'A2 (§ 6 (c) du rapport A2).

## Réserves

- **Des situations hors contexte.** Ce sont des décisions isolées, sorties de leur partie. Le
  test mesure le mécanisme, pas l'effet sur les captures. Seule une campagne peut le mesurer.
- **Des situations concentrées et corrélées.** Elles se concentrent sur quelques seeds (16 à la
  seed 9, 10 à la seed 20), et les décisions d'une même seed sont corrélées.
- **Un témoin réduit.** Le témoin `cible_vue` ne compte que 16 situations.

## Suite

**Une campagne A2' sur les seeds 0 à 29**, sur la même pile que A2 et A1bis. Elle aurait son
propre journal et sa propre empreinte de prompt (bras `A2p`). Elle devrait mesurer, en plus des
captures :
- la ligne « lieux vus vides seulement » du § 6 (a) ;
- l'écartement en exploration du § 6 (c), pour vérifier que la consigne ne regroupe pas les
  poursuivants.
