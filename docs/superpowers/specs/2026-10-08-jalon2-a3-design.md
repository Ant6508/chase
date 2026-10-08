# Jalon 2, bras A3 — canal JSON minimal sous budget borné

Date : 2026-10-08
Statut : validé pour implémentation. Construit sur la spec d'A2
(`docs/superpowers/specs/2026-09-29-jalon2-a2-design.md`), dont tout ce qui n'est pas
redit ici reste valable.

## Contexte

Les bras A2 et A2' sont terminés et rapportés ([`results/jalon2_a2.md`](../../../results/jalon2_a2.md),
[`results/jalon2_a2p.md`](../../../results/jalon2_a2p.md)).

| Bras | Captures sur 30 | Positions par message (moyenne) | Pensée par décision |
|---|---|---|---|
| A1bis, témoin sans message | 19 | 0 | 492 tokens |
| A2, JSON non contraint | 18 | 138 | 2 528 tokens |
| A2', A2 avec consigne d'usage | 15 | 116 | 2 435 tokens |
| P2, décisions de R2, message A2 | 27 | — | — |

- **Le canal est utilisé, mais il ne change pas les captures.** Le récepteur rejoint la cible vue
  par son coéquipier (92 % contre 63 % en A1bis), et les deux poursuivants s'écartent en
  exploration. L'information négative (les lieux vus vides) n'est exploitée qu'avec une consigne
  (A2'), et même alors sans gain de captures.
- **Le goulot est la phase de capture**, pas le canal. Les bras LLM laissent à la cible un
  territoire deux fois plus grand que R2 et P2, et la prennent rarement en tenaille.
- **Le prix d'A2 vient de `candidates`.** Le modèle rédige puis relit la liste des lieux : 5 fois
  plus de pensée qu'A1bis, et 16 % de décisions relancées.

La suite convenue dans le rapport A2' est **A3 minimale** : ne garder que `moi`, `cible` et
`intention`, et mesurer ce qui reste des usages positifs du canal sans le coût de `candidates`.

## Ce que fait le papier pour limiter le budget

Lecture du papier RecursiveMAS (arXiv 2604.25917) et de son code (commit `cbfcaab`), le
2026-10-08.

- **Le canal latent a une longueur fixe.** L'émetteur fait exactement `m` pas latents : le message
  occupe toujours `m` positions, sans arrêt anticipé.
  - L'ablation (annexe D.2, table 9) balaie `m` de 0 à 128 par pas de 16. La performance sature
    vers `m = 80`.
  - Les réglages publiés (`inference/inference_utils/inference_mas.py`, l. 106 à 145) prennent
    `latent_length` entre 16 et 64 selon la tâche.
- **Le témoin texte n'est borné que par une coupure.** « Recursive-TextMAS » a la même structure,
  avec des messages en texte libre. Ils ne sont limités que par `max_new_tokens`, et pour Gemma par
  une troncature en caractères faite après coup (planificateur à 200 tokens et 1 000 caractères ;
  500 caractères pour GPQA avec le modèle 4B).
- **Aucune comparaison à budget égal.** Le texte n'est pas balayé en volume. Le papier rapporte
  34,6 à 75,6 % de tokens en moins, à structure identique.

**Ce qu'on en retient.**
- Il n'y a pas de protocole de budget texte à copier : la courbe texte de la SPEC (§ 3) est un
  apport propre au projet.
- La coupure brutale ne convient pas à un message JSON : coupé, il devient illisible. Couper
  `candidates` ferait en plus mentir le schéma d'A2, où un lieu absent veut dire « vu vide ».
- A3 minimale tombe à environ 30 positions, à côté de `m = 32`, une valeur que le papier utilise.
  Cela permet une comparaison à budget égal avec A4, que le papier ne fait pas.

## Objectif

Produire le point bas de la courbe de Pareto de la SPEC (§ 3) pour le texte :
- en abscisse, environ 30 positions par message, contre 0 pour A1bis et 138 pour A2 ;
- en ordonnée, le confinement à T et les captures ;
- à politique de décision identique (croyance individuelle de R1, même perception, même modèle).

## Non-objectifs

- **Un balayage de budgets.** A3 est un seul point. Un point plafonné intermédiaire est laissé aux
  Points ouverts.
- **La phase de capture.** C'est une question de décision, pas de canal : aucune consigne de
  tenaille.
- **Une consigne d'usage du message.** Le prompt dit à quoi sert chaque champ, pas quoi en faire,
  comme en A2 (décision de l'utilisateur, 2026-10-08).
- **Toucher aux bras joués.** Les prompts, outils et empreintes d'A1, A1bis, A2 et A2' ne changent
  pas.
- **Les règles du jeu, la croyance, la cible et le profil.** Le jalon 1 est figé.

## Le message d'A3

### Schéma

L'argument `message` de l'outil `move` a trois champs, tous obligatoires, et aucun autre.

| Champ | Type | Contenu | Rôle annoncé (repris d'A2) |
|---|---|---|---|
| `moi` | chaîne | ma case, par ex. `"C2a.3"` | se répartir la recherche, préparer la tenaille |
| `cible` | chaîne ou `null` | la case de la cible si je la vois à ce pas | la localiser |
| `intention` | liste de chaînes, **3 au plus** | les prochains lieux que je compte traverser, dans l'ordre | l'intention sur N pas |

**Ce qui est retiré, et pourquoi.** `candidates` (l'ensemble candidat pondéré) et `je_couvre`
(l'issue couverte). C'est l'effet du budget. La SPEC (§ 3) demande que le bras JSON « ne soit pas
estropié » : ce bras-là est A2. A3 est le point bas de la courbe, pas un JSON complet, et le rapport
le dit.

### Le budget

**Il est imposé par le schéma borné**, annoncé au modèle dans l'outil et le prompt, et vérifié
après chaque appel : un message qui dépasse la borne est invalide, et l'appel est relancé.

- **Pourquoi 3 lieux.** Dans les 2 664 messages d'A2, `intention` comptait 1 lieu dans 9,9 % des
  cas, 2 dans 59,7 % et 3 dans 25,8 %. 95,4 % des messages respectaient donc déjà la borne. Elle
  gêne peu le comportement spontané, mais elle plafonne le message.
- **Taille d'un message**, comptée avec le tokenizer de gemma-4 :

  | Message | Positions |
  |---|---|
  | `{"moi":"C2a.3","cible":null,"intention":["K1","C4a"]}` | 22 |
  | `{"moi":"C2a.3","cible":"C6b.3","intention":["C6a","C6b"]}` | 27 |
  | `{"moi":"C10a.5","cible":"C10b.5","intention":["C10a","C10b","C10c"]}`, pire cas | 36 |

- **Pourquoi pas un plafond en tokens.** Le modèle ne sait pas compter ses tokens : un plafond de N
  positions vérifié par le harnais produirait des relances en série, à environ 2 500 tokens
  chacune.
- **Pourquoi pas une troncature par le harnais.** C'est la méthode du papier pour le texte, mais
  un JSON tronqué n'est plus lisible, et une liste tronquée change le sens du message.

### Validation

Les règles d'A2 s'appliquent (spec A2, § Validation), avec deux causes de rejet de plus :
- `intention` compte plus de 3 lieux ;
- le message a un champ en plus des trois. A2 tolérait les champs en trop ; en A3, un
  `candidates` ajouté ferait sauter le budget.

Un message rejeté entraîne la règle d'A1 : 2 relances, puis `STAY` sans message, et le coéquipier
lit « Aucun message reçu. ».

Inchangé :
- un `message` écrit en chaîne JSON qui se décode en objet est accepté après décodage ;
- les noms inconnus sont transmis tels quels et comptés (`unknown_names`) ;
- le harnais ne corrige jamais le contenu.

### Insertion et mesure

Inchangées. Le message validé est réécrit en JSON compact, dans l'ordre des clés de l'émetteur.
`message_tokens` compte ses positions avec le même tokenizer qu'A2, et `thinking_tokens` compte la
pensée à part. A2 et A3 se comptent donc exactement de la même façon.

### Prompt système

C'est celui d'A2 : la phrase sur le coéquipier joignable, le paragraphe des lieux, puis le
paragraphe du canal, et la consigne de réponse d'A2. Seul le paragraphe du canal est réécrit :

```
À chaque pas, tu écris un message à ton coéquipier dans le champ `message` de l'outil
`move`. Il le lira au pas suivant ; de même, le message qu'il t'a écrit au pas précédent
figure à la fin de ta perception. Ta pensée et ta justification restent privées : seul le
message lui parvient. Le message a trois champs, tous obligatoires, et aucun autre :
- `moi` : ta case, pour vous répartir la recherche et préparer une prise en tenaille ;
- `cible` : la case de la cible si tu la vois, sinon null ;
- `intention` : les prochains lieux que tu comptes traverser, dans l'ordre, trois au plus.
Exemple : {"moi":"C2a.3","cible":null,"intention":["K1","C4a"]}
```

- Les phrases gardées sont mot pour mot celles d'A2.
- La phrase « Un lieu absent des `candidates`… » disparaît avec le champ.
- Pas de `USAGE_PARAGRAPH`.

### Contrôles figés, repris d'A2

- Même perception qu'A1bis et A2 (spec A2, § Perception commune).
- Même modèle (`google/gemma-4-12b` Q6_K), `temperature=0`, 2 relances.
- `--max-tokens 5600`, comme A1bis et A2 : même budget de raisonnement individuel.
- Même profil réduit (`max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6`), seeds 0
  à 29.
- Même pile : LM Studio, moteur CUDA, 8 créneaux de 8 192 tokens.

## Architecture du code

Branche `jalon2-a3`. Approche retenue : une spécification de message par bras.

```
chase/llm/
  message.py   # MessageSpec ; A2_SPEC (l'actuel, à l'identique) et A3_SPEC ; validate(obj, spec)
  prompts.py   # ARMS et CHANNEL_ARMS + "A3" ; CHANNEL_PARAGRAPH_A3 ; MESSAGE_SPECS[arm]
  client.py    # move_tool(spec | None) ; decide(..., message_spec=None) remplace with_message
  policy.py    # LLMPursuers prend la spec de son bras
  ceiling.py   # write_message(..., spec) ; ProtocolPursuers(cfg, spec) : P2 ou P3
scripts/run_llm.py      # --arm A3 ; tools.json et empreinte construits avec la spec du bras
scripts/ceiling_a2.py   # P3 à côté de R1, R2 et P2
```

### `message.py`

- **`MessageSpec`**, objet figé :
  - `name` ;
  - `fields`, les champs dans l'ordre ;
  - `schema`, le sous-schéma JSON déclaré dans l'outil ;
  - `max_intention`, la borne d'`intention` (`None` pour A2) ;
  - `closed`, vrai si les champs en trop sont interdits (faux pour A2).
- **`A2_SPEC`** reprend exactement `FIELDS` et `MESSAGE_SCHEMA` actuels : l'outil d'A2 et d'A2' ne
  change pas d'un caractère.
- **`A3_SPEC`** déclare `moi`, `cible` et `intention`, avec `"maxItems": 3` sur `intention` et
  `"additionalProperties": false`. Le schéma informe le modèle ; c'est `validate` qui fait foi,
  puisque LM Studio n'impose pas forcément le schéma à la génération.
- **`validate(obj, spec)`** applique les contrôles de type des champs présents dans la spec, puis la
  borne et la fermeture.
- **`unknown_names(msg, places)`** parcourt les champs présents : il vaut pour les deux specs.
- `render`, `count_tokens` et `load_tokenizer` sont inchangés.

### `prompts.py`

- `ARMS = ("A1", "A1bis", "A2", "A2p", "A3")`, `CHANNEL_ARMS = ("A2", "A2p", "A3")`.
- `MESSAGE_SPECS = {"A2": A2_SPEC, "A2p": A2_SPEC, "A3": A3_SPEC}`.
- `system_prompt("A3")` : le corps d'A2, avec `CHANNEL_PARAGRAPH_A3` à la place de
  `CHANNEL_PARAGRAPH`, et la consigne de réponse d'A2.
- `message_block` est inchangé.

### `client.py`

- `move_tool(spec)` : sans spec, l'outil d'A1 ; avec une spec, le paramètre `message`, obligatoire,
  porte `spec.schema`.
- `decide(system_prompt, user_prompt, message_spec=None)` remplace `with_message`. Un message
  invalide pour la spec déclenche une relance, comme en A2.

### `policy.py`

`LLMPursuers` prend `MESSAGE_SPECS.get(arm)` et la passe au client. Le reste ne change pas : boîtes
de réception et d'envoi, `inbox = outbox` après les deux appels, croyance de R1.

### `ceiling.py` : le plafond mécanique P3

P3 est P2 restreint au message A3. Il joue avec les décisions de R2 (`GreedyPursuers._act_one`),
mais ne connaît de son coéquipier que ce que transporte le message A3, au pas précédent :
- **sa position** : la case `moi` ;
- **la cible** : si le message donne `cible`, cette case est propagée d'un pas puis intersectée avec
  la croyance individuelle, comme en P2 ;
- **sinon, aucune intersection.** Sans `candidates`, il n'y a pas de croyance du coéquipier à
  fusionner : chacun décide avec la sienne.

`write_message(..., spec=A3_SPEC)` écrit `moi`, `cible`, et les 3 premiers lieux d'`intention` tels
que P2 les calcule (lieux successifs des 5 prochaines cases du chemin prévu).
`ProtocolPursuers(cfg, spec=A2_SPEC)` reste P2 ; avec `A3_SPEC`, il s'appelle P3.

**Ce que P3 mesure.** Ce que vaut le protocole minimal pour un décideur parfait. L'écart P2 − P3
mesure ce que `candidates` apporte quand il est bien exploité. P3 n'est pas une porte : le bras est
défini par son budget, pas par l'optimalité de son protocole.

### `scripts/run_llm.py`

- `--arm A3`.
- `DIR/tools.json` et l'empreinte du journal sont construits avec `move_tool(MESSAGE_SPECS.get(arm))`.
  Pour A2 et A2', l'outil est identique : leurs journaux restent reprenables.

### Ce qui ne change pas

La perception, la croyance individuelle, `StepLog`, `EpisodeLLMStats`, le journal et les traces.
`message_tokens`, `thinking_tokens` et `unknown_names` sont déjà indépendants du schéma.

## Gestion des erreurs

Le tableau d'A2 (spec A2, § Gestion des erreurs) s'applique, avec deux causes de relance de plus.

| Cas | Comportement |
|---|---|
| `intention` de plus de 3 lieux | relance, puis `STAY` sans message, `fallback=True` |
| Champ en plus de `moi`, `cible`, `intention` | idem |

Aucune exception n'interrompt un épisode ni une campagne.

## Tests (faux client, sans réseau)

- **Message** (`tests/test_llm_message.py`, étendu) :
  - `A3_SPEC` accepte un message minimal, et `cible: null` ;
  - elle rejette une `intention` de 4 lieux, un champ en trop, et chaque mauvais type ;
  - `unknown_names` relève un nom inconnu dans un message A3 ;
  - les tests actuels passent avec `A2_SPEC`, sans changer ce qu'ils vérifient.
- **Prompts** (`tests/test_llm_prompts.py`, étendu) :
  - `system_prompt("A3")` contient le paragraphe A3, ne contient ni `candidates`, ni `je_couvre`,
    ni `USAGE_PARAGRAPH`, et garde le reste du prompt d'A2 ;
  - **les empreintes d'A2 et d'A2'** (prompt système et outil) sont figées à leur valeur du commit
    `40e6fc3`.
- **Client** (`tests/test_llm_client.py`, étendu) :
  - `move_tool(A3_SPEC)` déclare les trois champs, `maxItems` et `additionalProperties: false` ;
  - un message A3 invalide déclenche une relance, puis le repli.
- **Canal** (`tests/test_llm_policy.py`, étendu) : en A3, un message émis par le poursuivant 0 au
  pas `t` arrive chez le poursuivant 1 au pas `t+1`, et chez lui seul ; le premier pas et le pas qui
  suit un repli donnent « Aucun message reçu. ».
- **Plafond** (`tests/test_llm_ceiling.py`, étendu) :
  - `write_message(spec=A3_SPEC)` produit des messages valides pour `A3_SPEC`, sans nom inconnu ;
  - `ProtocolPursuers(spec=A3_SPEC)` garde la vraie position de la cible dans la croyance de chaque
    poursuivant, à chaque pas ;
  - sans `cible` reçue, sa croyance de décision est la croyance individuelle.
- **`run_llm`** (`tests/test_run_llm.py`, étendu) : `--arm A3` est accepté, et la reprise refuse un
  journal écrit par un autre bras.

## Validation avant la campagne

### 1. Plafond mécanique, en local et sans LLM

R1, R2, P2 et P3 sur 2 000 seeds, comme pour la porte d'A2, puis sur les seeds 0 à 29 pour le
tableau du rapport. Résultats dans `results/jalon2_a3_plafond.md`. Pas de porte : si P3 est proche
de R1, le protocole minimal ne porte presque rien d'exploitable, et le rapport le dit.

### 2. Pilote sur RunPod, seeds 0 à 7

8 parties en parallèle, mêmes options que la campagne.

On surveille :
- les replis et leurs causes (`attempt_errors`) ;
- les messages rejetés, par motif : borne d'`intention`, champ en trop, type ;
- les tokens de pensée par décision, les réponses coupées (`finish_reason="length"`) et la latence.

**Règle fixée d'avance.** On s'arrête et on revoit le prompt avant la campagne si :
- plus de 5 % des décisions finissent en repli ;
- ou plus de 10 % des décisions sont relancées pour un message invalide.

Sinon, on enchaîne.

## Campagne

La même commande, avec `--episodes 30`, reprend le journal du pilote :

```bash
campaign.sh start a3 --arm A3 --max-tokens 5600 --timeout 900 --first-seed 0 --episodes 30 \
    --concurrency 8 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --trace /workspace/results/a3/trace
```

- **Pod.** `14rh3c8qgzdpmi` (A40, EU-SE-1, 0,49 $/h), qui a déjà le modèle et le venv, si sa
  machine a un GPU libre. Sinon, un nouveau pod A40 avec la même image et la même configuration
  (`scripts/pod/`).
- **Exécution.** Détachée sur le pod. À la fin : rapatrier dans `results/jalon2_a3/`, vérifier les
  MD5, **mettre le pod en pause**.
- **Démarrage.** En mode auto, `pod-action start` est bloqué : l'utilisateur démarre le pod.
- **Coût estimé.** Sans liste de lieux à sérialiser, la pensée devrait se rapprocher de celle
  d'A1bis (492 tokens par décision, 5 h 40 de campagne) plutôt que de celle d'A2 (2 528 tokens,
  29 h 30). Soit 6 à 10 h de pod pilote compris, 3 à 5 $. Le pilote donnera le vrai chiffre.
  Prévoir un rechargement de crédit avant la campagne (dernier solde noté : 7,47 $ le 2026-10-03).

Sorties : `results/jalon2_a3/{journal.jsonl,trace/,table.md,run.log}`.

## Rapport `results/jalon2_a3.md`

- **Tableau.** R0, R1, R2, P2, P3, A1bis, A2, A2' et A3 : confinement à T (métrique principale),
  confinement moyen, captures avec leur IC à 95 %, médiane des pas jusqu'à la capture.
- **Comparaisons appariées** (McNemar exact, bilatéral, `scripts/compare_arms.py`) :
  - **A3 contre A1bis**, l'effet du canal minimal (comparaison principale) ;
  - **A3 contre A2**, ce que coûte le retrait de `candidates` et de `je_couvre` ;
  - A3 contre P3 ;
  - A3 contre R2.
- **La courbe de Pareto.**
  - Positions par message (moyenne, médiane, étendue) et par épisode.
  - Points : A1bis (0), A3, A2' (116) et A2 (138), avec l'emplacement de `m = 32`, la valeur
    qu'A4 visera pour une comparaison à budget égal.
- **La qualité des messages.**
  - Taux de messages valides, motifs de rejet, noms inconnus.
  - Justesse de `moi`.
  - Véracité de `cible` : inventée quand l'émetteur ne voit pas la cible, ou **relayée** depuis le
    message du coéquipier (le point de vigilance hérité d'A2', § 5.2 de son rapport).
  - Distribution de la longueur d'`intention`.
- **L'usage des messages**, lu dans les traces avec `results/jalon2_a2/msg_usage.py`, étendu aux
  messages sans `candidates` :
  - (a) les issues mortes, seulement quand le coéquipier voyait la cible : sans `candidates`, A3 ne
    transporte plus d'information négative ;
  - (b) le récepteur rejoint-il la cible vue par son coéquipier ?
  - (c) les deux poursuivants s'écartent-ils en exploration ?
  - (d) la phase de capture : territoire laissé à la cible, part des pas en tenaille.
- **Les coûts.** Tokens de prompt, de pensée et de message par décision ; relances par décision et
  tentatives non comptées ; latence, temps mural, dollars.
- **La revendication.**
  - Le schéma est un protocole conçu par nous, et le budget est imposé par le schéma (SPEC § 3).
  - A3 n'est pas le bras JSON complet : c'est le point bas de la courbe.
  - Le papier fixe la longueur du message latent, mais ne borne son témoin texte que par une
    coupure, sans comparaison à budget égal (§ Ce que fait le papier).

## Points ouverts (non traités ici)

- **Un point plafonné intermédiaire.** Le schéma d'A2, sous un budget de N = 64 positions annoncé
  au prompt, avec une troncature structurée par le harnais (on retire les derniers lieux des listes
  en gardant un JSON valide). Il faudrait alors changer le sens de `candidates` : un lieu absent
  serait « inconnu », et non plus « vu vide ». Coût estimé : environ 27 h et 13 $ de pod, comme A2.
  À envisager si A3 perd nettement face à A2, et si le crédit le permet.
- **A4.** Prévoir `m = 32` parmi les longueurs latentes, pour la comparaison à budget égal avec
  A3. Les points de la spec A2 (§ Points ouverts, A4) restent valables.
- **La phase de capture.** C'est le vrai chantier pour approcher P2, mais c'est une question de
  décision, hors du périmètre du canal.
