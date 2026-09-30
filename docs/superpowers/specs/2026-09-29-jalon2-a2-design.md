# Jalon 2, bras A2 — canal de message JSON non contraint

Date : 2026-09-29
Statut : validé pour implémentation. A3 (même canal sous budget) fera l'objet d'un
design séparé, sur la base de celui-ci.

## Contexte

Le bras A1 est terminé (`results/jalon2_a1v3.md`) :

- **Captures.** A1v3 capture 16 seeds sur 30, soit 53 %. R1 en capture 63 % et R2 83 %.
- **Comparaisons appariées.** A1v3 ne se distingue pas de R1 (p = 0,51). Il est
  significativement sous R2 (p = 0,02).
- **Marge d'A2.** R2 capture 11 des 14 seeds manquées par A1v3 : c'est ce qu'A2 peut
  espérer récupérer sur ce profil.
- **Échecs d'A1v3.** Le premier mode d'échec est la cible poursuivie sans être prise
  (6 seeds). Viennent ensuite la cible jamais trouvée (4) et la cible trouvée tard (4).
  Ce qui manque est la coordination du second poursuivant.

**Pourquoi A2 avant A4.** La préparation du jalon 3
(`results/jalon3_recursivemas_gemma4.md`) a établi deux choses :
- les liens de RecursiveMAS s'apprennent sur du texte ;
- le message latent occupe, dans le prompt du récepteur, l'emplacement du message
  textuel.

A2 fournit donc à la fois le témoin texte d'A4 et ses données d'entraînement.

## Objectif

Produire le point « JSON non contraint » de la courbe de Pareto de la SPEC (§3) :
- en abscisse, le volume de message mesuré en positions dans le contexte du récepteur ;
- en ordonnée, la performance (confinement à T, captures) ;
- à politique de décision identique à celle d'un témoin sans communication, A1bis.

## Non-objectifs

- **A3.** Le même canal sous budget imposé à la génération fera l'objet d'un design
  séparé.
- **A4 et la pile `transformers` bf16.** Le témoin texte sur cette pile reste au
  jalon 3.
- **Les règles du jeu.** Aucune modification des règles, de la croyance calculée par le
  harnais, de la cible ni du profil : le jalon 1 est figé.
- **Stratégie.** Aucune consigne de stratégie dans les prompts (répartition, tenaille).
  Le prompt dit à quoi sert chaque champ du message, pas quoi en faire.
- **Relais.** Pas de relais des informations reçues dans le message émis (voir Points
  ouverts).

## Principe : le harnais ne lit jamais le message

Le message arrive tel quel dans le prompt du récepteur, et c'est le LLM récepteur qui
l'exploite. Le harnais ne s'en sert jamais pour calculer quoi que ce soit : ni croyance,
ni perception, ni fusion.

Trois raisons :
1. **Seul le canal change entre A2 et A4.** A4 injectera au même endroit des vecteurs
   que le harnais ne peut pas décoder. Si le harnais fusionnait les croyances en A2, les
   deux bras n'auraient plus la même politique de décision (SPEC §3 : « seul ce canal
   est remplacé d'un bras à l'autre »).
2. **L'unité de mesure resterait vide.** La SPEC compte les positions occupées dans le
   contexte du récepteur. Un message consommé par le harnais n'en occupe aucune.
3. **On mesurerait le harnais, pas le canal.** Une fusion par le harnais redonnerait
   R2, qu'on connaît déjà.

Conséquences :
- **La croyance reste celle de R1 dans les trois bras.** Le harnais calcule pour chaque
  poursuivant sa croyance individuelle, exactement comme R1.
- **Le LLM n'a pas de mémoire.** Chaque appel reste un tour isolé. L'information du
  coéquipier ne s'accumule donc pas d'un pas à l'autre : elle n'existe que dans le
  message du pas en cours.

## Le témoin A1bis

La perception d'A1v3 est égocentrique : des issues et des distances. Le LLM n'y connaît
ni sa position ni la carte, et un message comme « cible en (4,3) » ou « j'ai vidé le
nord » serait inexploitable par le coéquipier. A2 ajoute donc à la perception un
référentiel commun, les lieux nommés.

La SPEC (§5, jalon 2) exige une observation stable entre les bras. A1 est donc relancé
avec la même perception qu'A2, sous le nom d'**A1bis** :
- même perception qu'A2 ;
- le prompt système d'A1v3, plus le paragraphe sur les lieux ;
- aucun message.

Les deux comparaisons se lisent alors ainsi :
- **A1bis → A2** mesure l'effet du canal, et rien d'autre ;
- **A1v3 → A1bis** mesure l'effet du référentiel seul.

## Protocole d'un pas (A2)

**Ce que reçoit le poursuivant `i` au pas `t`**, dans son prompt utilisateur :
1. sa perception (§ Perception commune) ;
2. à la fin du prompt, sous l'en-tête fixe « Message de ton coéquipier (écrit au pas
   précédent) : », le message écrit par son coéquipier au pas `t-1`.

Au premier pas, ou si le coéquipier n'a pas pu écrire de message (repli), le bloc
indique « Aucun message reçu. ».

**Ce qu'il renvoie**, en un seul appel : `move(direction, reasoning, message)`.
- **Ce qui est transmis.** Seul `message` part vers le coéquipier.
- **Ce qui reste privé.** La pensée (`reasoning_content`) et la justification
  (`reasoning`) ne sont que journalisées.

**Pourquoi un seul appel.** Un second appel dédié au message ajouterait une deuxième
phase de pensée, environ 446 tokens par décision en A1v3. Il doublerait presque le
temps d'une campagne, déjà d'environ 26 h en local.

**Délai.** Les deux appels d'un même pas ne dépendent que des messages du pas
précédent. C'est le délai d'un pas de la SPEC, celui de R2 avec `comm_delay=1`.

**Contrôles figés, repris d'A1v3.**
- Même modèle : `google/gemma-4-12b` Q6_K.
- Mêmes réglages : `temperature=0`, 2 relances.
- Même profil réduit : `max_steps=60 size=15 n_loops=1 min_loop_len=6
  min_spawn_dist=6`.
- Mêmes seeds : 0 à 29.

**Budget de génération (révisé le 2026-09-30).** `max_tokens` vaut 4 000 pour A1bis
comme pour A2. A1 garde les 1 600 d'A1v3. Le message se prend sur ce même budget que
la pensée. Les mesures (`results/jalon2_a2_budget.md`) :
- **A2** consomme de 2 600 à 3 300 tokens par appel, dont 2 300 à 3 000 de pensée : le
  modèle rédige et relit la liste des lieux avant de l'écrire. Avec 1 600, les trois
  appels mesurés ont tous été coupés (`finish_reason=length`).
- **A1bis** consomme environ 450 tokens par appel, comme A1v3. Le plafond commun ne le
  gêne donc pas, et le contrôle « même budget de raisonnement » reste respecté.

Ce surcoût de pensée d'A2 est un résultat en soi, à rapporter : c'est ce que coûte au
texte la sérialisation d'un ensemble.

## Référentiel de lieux

Les noms ne dépendent que de la carte. Ils sont donc identiques pour les deux
poursuivants et dans tous les bras.

- **Carrefours.** Ce sont les cases libres qui ont au moins 3 voisines libres. Elles
  sont nommées `K1`, `K2`… dans l'ordre de lecture : du nord au sud, puis de l'ouest à
  l'est.
- **Couloirs.** Ce sont les composantes connexes des autres cases, nommées `C1`, `C2`…
  dans l'ordre de lecture de leur première case.
- **Sens de lecture d'un couloir.** Chaque couloir est orienté depuis son extrémité
  voisine du carrefour de plus petit nom ; une extrémité en cul-de-sac vient toujours
  en dernier. Si les deux extrémités touchent le même carrefour, on part de celle dont
  la case vient la première dans l'ordre de lecture.
- **Tronçons.** Un couloir de plus de 5 cases est coupé en `ceil(n/5)` tronçons
  consécutifs et équilibrés (tailles égales à une case près, les plus longs d'abord),
  suffixés `a`, `b`, `c`… dans le sens de lecture : `C7a`, `C7b`… Un couloir de 5 cases
  ou moins garde son nom sans suffixe.
- **Case précise.** Une case se désigne par son tronçon et son rang dans le sens de
  lecture, à partir de 1 : `C6b.3`. Une case de carrefour se désigne par le nom du
  carrefour.

Mesures sur les 30 cartes du profil réduit :

| | Médiane | 90e centile | Maximum |
|---|---|---|---|
| Longueur des couloirs, en cases | 5 | 16 | 48 (seed 5) |

| | Moyenne | Minimum | Maximum |
|---|---|---|---|
| Lieux par carte (carrefours et tronçons) | 30,6 | 26 | 35 |

Avec un couloir entier comme lieu, on n'aurait que 19,8 lieux en moyenne, mais un seul
lieu pourrait couvrir jusqu'à 48 cases : trop grossier pour décrire un ensemble
candidat. Les tronçons de 5 cases correspondent à peu près à la portée du champ de
vision (`view_size=5`).

## Perception commune (A1bis et A2)

La perception reste celle d'A1v3, texte compris : l'annonce de la cible, les quatre
lignes de direction et le total des candidates. Elle reçoit quatre ajouts :

1. **Une ligne de position, en tête.**
   - Si mon lieu contient encore des cases candidates : « Tu es en C2a.3 (ton lieu C2a
     contient des candidates : 12 %). »
   - Sinon : « Tu es en C2a.3. »
2. **La case de la cible**, dans l'annonce quand elle est visible : « Cible visible en
   C6b.3, à 3 pas par les couloirs (à vol d'oiseau : …). »
3. **Sous chaque issue praticable**, une ligne « lieux par là : » qui liste les lieux
   dont la case la plus proche s'atteint au plus court par cette issue.
   - Chaque lieu est donné par son nom et sa distance.
   - S'il contient au moins une case candidate, on ajoute sa part de probabilité.
     Exemple : `K1 à 3, C1 à 4 (2 %), C4a à 4 (moins de 1 %)`.
   - Le tri se fait par distance, puis dans l'ordre des noms.
   - Un lieu aussi proche par deux issues apparaît sous les deux.
   - Le lieu où l'on se trouve n'y figure pas.
4. **Les noms des cases candidates.** Quand l'ensemble candidat est petit
   (`<= track_threshold`), chaque case de la liste est précédée de son nom :
   « C6b.3 à 3 pas par SUD (100 %) ».

Les parts de probabilité viennent de la même carte qu'en A1v3
(`belief.diffuse`/`observe_prob`), sommée sur les cases du lieu.

Exemple réel, seed 5, pas 8, poursuivant 0 sous R1 (liste raccourcie) :

```
Tu es en C2a.3.
Cible non visible.
Directions :
- NORD : praticable ; 75 cases candidates au plus court par là (97 % de la probabilité), la plus proche à 5 pas
  lieux par là : K1 à 3, C1 à 4 (2 %), C4a à 4 (moins de 1 %), C4b à 8 (3 %), K3 à 11 (2 %), … C7j à 66 (5 %)
- SUD : praticable ; 10 cases candidates au plus court par là (3 % de la probabilité), la plus proche à 6 pas
  lieux par là : C2b à 3 (moins de 1 %), C2c à 7 (moins de 1 %), K2 à 11 (1 %), C3 à 12 (moins de 1 %), C5 à 12 (1 %)
- EST : mur
- OUEST : mur
Ensemble candidat total : 85 case(s).
```

**Pourquoi ranger les lieux sous les issues.**
- Le récepteur lit un nom dans le message, le retrouve sous une de ses issues, et sait
  par où c'est et à combien de pas. C'est une correspondance de noms, sans calcul de
  chemin : l'intersection d'ensembles que la SPEC juge à la portée d'un LLM.
- L'ordre des tronçons (`C7a` puis `C7b`) et les distances suffisent à voir si la cible
  est entre soi et son coéquipier.

**Écart avec l'autre forme essayée.** Une table d'une ligne par lieu, avec la topologie
de chaque couloir, coûtait 700 à 780 tokens.

**Coût.** Mesuré avec le tokenizer de gemma-4 sur deux cartes (seeds 0 et 5) :
- perception : 321 et 386 tokens, contre environ 106 au format A1v3 ;
- prompt complet : environ 850 tokens, contre environ 600.

**Prompt système.** Celui d'A1bis et celui d'A2 gagnent un paragraphe qui explique les
noms (carrefours, couloirs, tronçons, rang d'une case, sens de lecture). Il précise
aussi qu'un lieu sans pourcentage ne contient aucune case candidate.

## Le message

### Schéma

L'outil `move` porte, en A2 seulement, un argument `message`. C'est un objet dont les
cinq champs sont obligatoires :

| Champ | Type | Contenu | Rôle annoncé |
|---|---|---|---|
| `moi` | chaîne | ma case, par ex. `"C2a.3"` | se répartir la recherche, préparer la tenaille |
| `cible` | chaîne ou `null` | la case de la cible si je la vois à ce pas | la localiser |
| `candidates` | objet `lieu → nombre` | les lieux où la cible peut être d'après ma perception, avec leur probabilité en pourcentage entier, par ex. `{"C7a": 7, "C9": 6}` | l'ensemble candidat pondéré |
| `intention` | liste de chaînes | les prochains lieux que je compte traverser, dans l'ordre | l'intention sur N pas |
| `je_couvre` | chaîne ou `null` | le lieu que je bloque ou garde | l'issue couverte |

Ces trois derniers champs sont le minimum exigé par la SPEC (§3) : ensemble candidat
pondéré, intention sur N pas, issue couverte.

**Aucune limite de longueur.** L'émetteur liste autant de lieux qu'il le veut, et
`intention` n'a pas de taille maximale. C'est ce qui distingue A2 d'A3.

**Sémantique**, énoncée dans le prompt système d'A2 :
- `candidates` se lit d'après la perception de l'émetteur, pas d'après le message qu'il
  a reçu.
- Un lieu absent de `candidates` a été vu vide par l'émetteur au pas précédent : la
  cible n'y est presque sûrement pas.

Le prompt système donne un exemple de message.

**Revendication à annoncer.** Ce schéma est un protocole conçu par nous. A2 mesure donc
« le texte à charge utile fixée », pas « des agents qui découvrent quoi se dire »
(SPEC §3). Le rapport le dit.

### Validation

- **L'appel est relancé** si les arguments ne sont pas du JSON lisible, si un champ du
  message manque, ou si un champ a le mauvais type :
  - une valeur de `candidates` qui n'est pas un nombre ;
  - `intention` qui n'est pas une liste de chaînes ;
  - `moi` qui n'est pas une chaîne ;
  - `cible` ou `je_couvre` qui n'est ni une chaîne ni `null`.

  La règle est celle d'A1 : 2 relances, puis repli sur `STAY` sans message. Le
  coéquipier reçoit alors « Aucun message reçu. ».
- **Les noms ne sont pas filtrés.** Un nom qui n'existe pas sur la carte (`C99`,
  `C4b.9`) est transmis tel quel, comme un bruit du canal. Il est compté dans le
  journal (`unknown_names`).
- **Le harnais ne corrige jamais le contenu.**

### Insertion chez le récepteur

Le message validé est réécrit en JSON compact :
`json.dumps(message, ensure_ascii=False, separators=(",", ":"))`.
- L'ordre des clés est celui de l'émetteur.
- Seuls les espaces changent par rapport à ce qu'il a écrit.

### Mesure

- **`message_tokens`.** C'est le nombre de tokens du message réécrit, tel qu'il est
  inséré chez le récepteur, compté avec le tokenizer de `google/gemma-4-12B-it` :
  bibliothèque `tokenizers`, fichier `tokenizer.json` seul, sans les poids, sans token
  spécial. L'en-tête fixe n'est pas compté.
- **`thinking_tokens`.** C'est le coût de raisonnement, compté séparément : les tokens
  de la pensée (`reasoning_content`), avec le même tokenizer. Communication et
  raisonnement sont ainsi séparés, comme l'exige la SPEC (§5).
- **Ordre de grandeur,** mesuré sur deux messages écrits à la main :

  | Situation | Tokens |
  |---|---|
  | Cible localisée | 51 |
  | 11 lieux candidats | 164 |
  | Les 30 lieux listés (estimation) | environ 250 |

## Architecture du code

Branche `jalon2-a1`. Le harnais A1 reste utilisable tel quel (`--arm A1`), avec
exactement la perception d'A1v3.

```
chase/llm/
  places.py    # nouveau : lieux d'une carte (noms, cases, nom de chaque case)
  message.py   # nouveau : schéma de `message`, validation, réécriture, comptage, noms inconnus
  ceiling.py   # nouveau : plafond mécanique du protocole (P2) et écriture mécanique d'un message
  prompts.py   # build_perception(..., places=None) ; prompts système A1, A1bis, A2
  client.py    # outil `move` avec `message` en A2 ; LLMCallResult étendu
  policy.py    # LLMPursuers(..., arm="A1" | "A1bis" | "A2") ; boîtes de réception/envoi
  logging.py   # StepLog / EpisodeLLMStats étendus
scripts/run_llm.py       # --arm ; traces complètes
scripts/ceiling_a2.py    # nouveau : R1, R2 et P2 sur 200 seeds (§ Validation, étape 1)
scripts/situations_a2.py # nouveau : situations témoins contre le modèle local (étape 2)
requirements.txt         # + tokenizers
```

P2 vit dans `chase/llm/` et non dans `chase/policies.py` : il dépend des lieux, et
`make_policy` ne doit pas importer le paquet LLM.

### `places.py`

- **Rôle.** `Places.from_graph(g: MazeGraph)` applique les règles du § Référentiel de
  lieux.
- **Ce qu'il expose :**
  - la liste ordonnée des lieux, chacun avec son nom et ses indices de cases ;
  - le nom de chaque case (`cell_name[i]`) ;
  - l'ensemble des noms valides, lieux et cases, pour repérer les noms inconnus.
- **Quand il est calculé.** Une fois par épisode, dans `LLMPursuers.reset`.
- **Dépendances.** Aucune, sinon le graphe.

### `message.py`

- `MESSAGE_SCHEMA` : sous-schéma JSON du paramètre `message` de l'outil.
- `validate(obj) -> str | None` : `None` si le message est valide, sinon la raison.
- `render(msg) -> str` : le JSON compact.
- `count_tokens(text) -> int` : le tokenizer de gemma-4, chargé une fois et partagé
  entre les threads.
- `unknown_names(msg, places) -> list[str]` : les noms absents de la carte.

### `ceiling.py`

- `write_message(...)` rédige mécaniquement un message à partir de la croyance
  individuelle et de la position d'un poursuivant, au grain des lieux :
  - `moi` : sa case ;
  - `cible` : la case de la cible s'il la voit ;
  - `candidates` : tous les lieux qui contiennent une candidate, avec leur part de
    probabilité arrondie ;
  - `intention` : les lieux successifs des 5 prochaines cases de son chemin vers son
    objectif ;
  - `je_couvre` : toujours `null`, P2 n'ayant pas de notion de lieu gardé.
- `ProtocolPursuers` est la politique scriptée P2 (§ Validation, étape 1). Elle réutilise
  la décision de R2 (`GreedyPursuers._act_one`, répartition de Voronoï et tenaille).
  Elle ne connaît de son coéquipier que ce que `write_message` transmet.

### `prompts.py`

- `build_perception(pos, g, belief, prob, target_seen, track_threshold, places=None)`.
  - Avec `places=None`, la sortie est identique au caractère près au format A1v3.
  - Avec `places`, on obtient les quatre ajouts du § Perception commune.
- `system_prompt(arm)` renvoie :
  - `A1` : le `SYSTEM_PROMPT` actuel, inchangé ;
  - `A1bis` : celui-ci, plus le paragraphe sur les lieux ;
  - `A2` : le même, avec la phrase sur le coéquipier remplacée (« avec qui tu échanges
    un message à chaque pas »), plus le paragraphe sur le canal (champs, sémantique,
    exemple). La consigne de sortie demande `direction`, `reasoning` et `message`.
- `message_block(msg | None) -> str` : l'en-tête fixe, suivi du message réécrit ou de
  « Aucun message reçu. ».

### `client.py`

- `decide(system_prompt, user_prompt, with_message=False)`. L'outil `move` ne déclare
  le paramètre `message`, obligatoire, que si `with_message`.
- `LLMCallResult` gagne deux champs :
  - `message: dict | None`, le message validé ;
  - `raw_arguments: str`, la chaîne brute des arguments de l'appel d'outil. C'est celle
    dont A4 aura besoin pour situer les tokens du message.
- Un message invalide déclenche une relance, comme un appel d'outil invalide en A1.

### `policy.py`

- **Constructeur.** `LLMPursuers(cfg, llm_cfg, client=None, arm="A1")`, et `name = arm`.
- **`reset`.**
  - Calcule les lieux, sauf en A1.
  - Vide les boîtes de réception : `inbox = [None, None]`.
- **`update`.** Inchangé : croyance et carte de probabilité individuelles, comme R1.
- **`act`.** Pour chaque poursuivant `i` :
  1. il construit sa perception ;
  2. en A2, il ajoute `message_block(inbox[i])` ;
  3. il appelle le client ;
  4. il range son message dans `outbox[1 - i]`, ou `None` en cas de repli.

  **Après les deux appels seulement**, on fait `inbox = outbox`. Le poursuivant 1 ne
  voit donc jamais le message que le poursuivant 0 vient d'écrire au même pas.

### `logging.py`

- **`StepLog` gagne :**
  - `user_prompt` : le prompt utilisateur complet, perception plus message reçu ;
  - `message_in` et `message_out` : les messages réécrits, ou `None` ;
  - `raw_arguments` ;
  - `thinking_tokens` ;
  - `unknown_names`.

  `message_tokens` vaut le compte de `message_out`, soit 0 en A1 et A1bis. Il est
  compté à l'émission, et le texte compté est exactement celui que le récepteur lira
  au pas suivant. Le dernier message d'un épisode n'est jamais lu, mais il est compté
  quand même : c'est un coût payé par l'émetteur.
- **`EpisodeLLMStats` gagne :**
  - `total_thinking_tokens` ;
  - `mean_message_tokens`, sur les messages effectivement émis ;
  - `messages_sent` ;
  - `unknown_name_count`.

### `scripts/run_llm.py`

- **Option `--arm {A1,A1bis,A2}`**, avec `A1` par défaut.
- **Journal.** Le journal JSONL garde un seul bras par fichier. La reprise refuse un
  journal écrit par un autre bras.
- **Trace, en A1bis et A2.** Chaque décision garde : la seed, le pas, le poursuivant,
  les positions vraies, le prompt utilisateur complet, les messages reçu et émis, les
  arguments bruts, la pensée et le coup.
- **Prompt système.** Il est écrit une fois dans `DIR/system_prompt.txt`.
- **Jeu de données d'A4.** Ces traces le constituent. Pour chaque message, on sait qui
  l'a écrit, à partir de quel prompt, et ce que le récepteur a joué au pas suivant.

## Flux de données (un pas, A2)

1. `runner`/`run_llm` construit les `Percept`. Rien ne change.
2. `LLMPursuers.update` fait évoluer la croyance individuelle de chaque poursuivant,
   comme R1.
3. Pour chaque poursuivant `i`, `LLMPursuers.act` :
   - construit la perception (lieux compris) ;
   - ajoute `message_block(inbox[i])` ;
   - appelle `client.decide(system_prompt("A2"), prompt, with_message=True)` ;
   - obtient le coup et le message ;
   - range le message dans `outbox[1 - i]` ;
   - journalise.
4. Après les deux poursuivants : `inbox = outbox`.
5. L'environnement applique les coups. Rien ne change.

## Gestion des erreurs

| Cas | Comportement |
|---|---|
| Serveur injoignable, délai dépassé | 2 relances, puis `STAY`, sans message, et `fallback=True` (comme A1) |
| Pas d'appel d'outil, ou direction hors énumération | idem |
| Message absent, champ manquant ou de mauvais type | idem |
| Nom de lieu ou de case inconnu dans un message valide | transmis tel quel, compté dans `unknown_names` |
| Réponse coupée (`finish_reason="length"`) sans appel d'outil complet | relance, puis repli ; le taux est suivi au pilote |
| Coéquipier en repli au pas précédent | le récepteur lit « Aucun message reçu. » |

Aucune exception n'interrompt un épisode ni une campagne.

## Tests (faux client, sans réseau)

- **Lieux** (`tests/test_llm_places.py`), sur plusieurs seeds :
  - chaque case libre a exactement un nom de case, et les noms sont uniques ;
  - aucun tronçon ne dépasse 5 cases ;
  - les tronçons d'un couloir se suivent dans le sens de lecture ;
  - les carrefours ont au moins 3 voisines ;
  - les noms ne dépendent que de la seed.
- **Perception** (`tests/test_llm_prompts.py`, étendu) :
  - sans lieux, la sortie est identique au format A1v3 (non-régression) ;
  - avec lieux, chaque lieu apparaît sous chaque issue qui y mène au plus court, et
    seulement sous celles-là ;
  - le pourcentage n'apparaît que pour un lieu qui contient une candidate ;
  - la position et la cible sont nommées.
- **Message** (`tests/test_llm_message.py`) :
  - la validation couvre un cas par type d'erreur ;
  - la réécriture est compacte et garde l'ordre des clés ;
  - le comptage marche sur un exemple fixe ;
  - un nom inconnu est relevé mais le message reste valide.
- **Canal** (`tests/test_llm_policy.py`, étendu) :
  - un message émis par le poursuivant 0 au pas `t` apparaît dans le prompt du
    poursuivant 1 au pas `t+1`, et ni dans celui du poursuivant 1 au pas `t`, ni dans
    celui du poursuivant 0 ;
  - le premier pas et le pas qui suit un repli donnent « Aucun message reçu. » ;
  - A1bis n'a aucun bloc de message, et A1 garde exactement la perception d'A1v3 ;
  - la croyance individuelle reste identique à celle de R1 dans les trois bras.
- **Client** (`tests/test_llm_client.py`, étendu) :
  - le paramètre `message` n'est déclaré qu'avec `with_message=True` ;
  - un message invalide déclenche une relance, puis le repli ;
  - `raw_arguments` est conservé.
- **Plafond** (`tests/test_llm_ceiling.py`) :
  - `write_message` produit des messages valides, sans nom inconnu ;
  - `ProtocolPursuers` garde la vraie position de la cible dans la croyance de chaque
    poursuivant, à chaque pas (même invariant que `tests/test_chase.py`).

## Validation avant les campagnes

Même principe qu'A1 : vérifier à coût nul avant de dépenser des heures de LLM.
L'essai mécanique d'A1 prédisait correctement la campagne : 60 % pour le suiveur de
l'issue la plus probable, contre 53 % mesurés.

### 1. Plafond mécanique du protocole (P2), sans LLM

`ProtocolPursuers` joue avec la décision de R2, mais ne connaît de son coéquipier que
ce que le schéma transporte, au pas précédent :
- **sa position** : la case `moi`, qui remplace la position connue de R2 ;
- **sa croyance, au grain des lieux.** On prend l'union des cases de ses lieux
  `candidates`, telle qu'elle était au pas `t-1`. On la propage d'un pas, puis on
  l'intersecte avec la croyance individuelle du pas `t` (et la carte de probabilité est
  restreinte à ce support, puis renormalisée) ;
- **la cible.** Si le message donne `cible`, c'est cette case, et non l'union des
  lieux, qui est propagée d'un pas puis intersectée.

Pas d'accumulation d'un pas à l'autre : chaque poursuivant garde sa croyance
individuelle, et l'intersection n'est refaite que pour décider. C'est exactement ce
qu'un LLM sans mémoire peut faire avec le message.

On l'évalue sur 200 seeds, avec R1 et R2 sur les mêmes seeds.

**Porte.** Si P2 perd plus de la moitié de l'écart R1 → R2 en taux de capture, le
protocole est en cause, pas le LLM. On revoit alors le grain avant tout appel au
modèle :
- des tronçons de 3 cases ;
- ou des cases désignées par intervalles.

### 2. Situations témoins (environ 1 h en local)

Dix situations tirées de parties de R2, choisies pour que le message change le bon
coup :
- la cible n'est vue que par le coéquipier ;
- l'issue la plus probable mène vers des lieux que le coéquipier a déjà vidés ;
- la cible se trouve entre les deux poursuivants.

Chaque situation est présentée au modèle local deux fois :
- avec la perception A1bis seule ;
- en A2, avec un message écrit par `write_message`.

On relève :
- si le coup A2 rejoint celui de P2 plus souvent que le coup A1bis ;
- si le message émis est valide ;
- si `candidates` est complet par rapport à la croyance de l'émetteur.

Il n'y a pas de seuil figé. Les résultats sont consignés et décident du lancement du
pilote.

### 3. Pilote A2, seeds 0 à 3, sur RunPod

On surveille : les replis et leurs causes (`attempt_errors`), les messages invalides,
les noms inconnus, les positions par message, le taux de `finish_reason="length"`, les
tokens de pensée et la latence.

**Contexte par créneau (révisé le 2026-09-30).** Les prompts réels, mesurés avec le
tokenizer, sont environ deux fois plus longs que prévu :
- **A1bis** : environ 740 tokens de partie fixe (système et outil), jusqu'à environ
  1 400 avec la perception ;
- **A2** : environ 1 150 tokens de partie fixe (système et outil au schéma allégé), et
  de 1 600 à 2 500 avec la perception et le message.

Avec 4 000 tokens de complétion, il faut donc au moins 6 500 tokens par créneau. Le pod
charge **8 créneaux de 8 192 tokens** (contexte total 65 536, `scripts/pod/setup.sh`).
Le même réglage sert à A1bis.

## Campagnes

**Sur RunPod, pour A2 comme pour A1bis (décision du 2026-09-30).**
- **Pourquoi pas en local.** Sur la carte de 12 Go, un appel A2 prend de 6,5 à
  8,5 minutes : environ 8 jours pour A2, et environ 10 avec A1bis. Le modèle y déborde
  aussi en mémoire partagée dès qu'un autre programme occupe la VRAM (voir
  `results/jalon2_a2_budget.md`).
- **Réglages du pod.** 8 créneaux de 8 192 tokens, `--concurrency 8`, `--timeout 900`,
  `--max-tokens 4000`, même profil, seeds 0 à 29, `--trace`.
- **Durées estimées**, sur la base d'environ 120 tokens/s au total mesurés sur RunPod en
  septembre : environ 20 h pour A2, et environ 3 h pour A1bis.
- **Même pile pour les deux bras.** Le témoin et A2 tournent ainsi sur la même pile
  (CUDA). L'écart matériel avec A1v3, qui tournait sous Vulkan sur carte AMD, est à
  signaler dans le rapport.

Ordre :
1. **A2 d'abord.** Sorties : `results/jalon2_a2/{journal.jsonl,trace/,table.md,run.log}`.
2. **A1bis ensuite.** Sorties : `results/jalon2_a1bis/`, même structure.

Commandes : voir l'en-tête de `scripts/pod/deploy.sh`. `campaign.sh start` reprend un
run interrompu aux seeds manquantes. Un épisode où `fallback_count > 1` est suspect.

## Rapport `results/jalon2_a2.md`

- **Tableau.** On y met R0, R1, R2, P2, A1v3, A1bis et A2, avec :
  - le confinement à T, métrique principale ;
  - le confinement moyen ;
  - les captures, avec leur IC à 95 % ;
  - la médiane des pas jusqu'à la capture.
- **Comparaisons appariées.** Test de McNemar exact, bilatéral :
  - **A2 contre A1bis**, l'effet du canal (comparaison principale) ;
  - A2 contre R2 ;
  - A2 contre P2 ;
  - A1bis contre A1v3, l'effet du référentiel seul.
- **Point de Pareto d'A2.**
  - En abscisse : positions par message, moyenne et médiane, et total par épisode.
  - En ordonnée : confinement à T et captures.
  - C'est le premier point de la courbe qu'A3 balaiera.
- **Qualité des messages.**
  - Taux de messages valides, noms inconnus.
  - Complétude de `candidates` : la part de la vraie masse de l'émetteur couverte par
    les lieux listés.
  - Justesse de `moi` et de `cible`.
- **Usage des messages, lu dans les traces.**
  - Le récepteur évite-t-il les issues qui ne mènent qu'à des lieux vidés par son
    coéquipier ?
  - Rejoint-il la cible quand seul son coéquipier la voit ?
  - Les deux poursuivants se séparent-ils en exploration ?
- **Coûts.** Tokens de prompt, de pensée et de message par décision. Latence et temps
  mural, avec les mêmes réserves qu'en A1v3.
  - **Le surcoût de pensée d'A2 face à A1bis** est un résultat à part entière : le prix
    de la sérialisation.
  - **Les relances de chaque bras** doivent être rapportées. Les tokens des tentatives
    relancées ne sont pas comptés, et A2 a une cause de relance de plus : le message
    invalide.
- **Revendication.** Le schéma est un protocole conçu par nous (§ Le message).

## Points ouverts (non traités ici)

- **A3.** Budget imposé à la génération sur les mêmes champs. Il reste à choisir ce qui
  se tronque (nombre de lieux de `candidates`, longueur d'`intention`) et les valeurs de
  budget à balayer.
- **Relais.** Faire écrire à l'émetteur `candidates` comme l'intersection de sa
  perception et du message reçu. L'information s'accumulerait alors d'un pas à l'autre
  à travers les messages, au prix d'un calcul mental de plus. À envisager si A2 plafonne
  nettement sous P2.
- **A4.**
  - Choisir le coup cible de l'entraînement : celui de R2, de P2 ou d'A2.
  - Reconstituer le tour de l'émetteur pour en extraire les états cachés sur les
    tokens du message. Mise en garde (relecture finale, 2026-09-30) :
    `raw_arguments` est la réécriture JSON faite par LM Studio. Ce n'est pas ce que le
    modèle a généré, puisque gemma-4 écrit ses appels d'outil dans une syntaxe native
    (`<|tool_call>`). A4 devra donc reconstruire le tour avec le gabarit de chat, à
    partir de la pensée et des arguments décodés. Il a besoin pour cela de l'outil
    exact, écrit par `run_llm --trace` dans `DIR/tools.json`, et des `prompt_tokens`
    de chaque décision pour valider la reconstruction.
  - Relancer un bras texte sur la pile `transformers` bf16.
