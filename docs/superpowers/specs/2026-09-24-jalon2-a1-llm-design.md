# Jalon 2, bras A1 — harnais LLM sans communication

Date : 2026-09-24
Statut : validé pour implémentation (sous-projet 1 du jalon 2 ; A2/A3 feront l'objet
d'un design séparé une fois A1 validé contre le modèle local).

## Contexte

Le jalon 1 (simulateur + heuristiques scriptées R0/R1/R2) est terminé et figé
(`results/jalon1.md`, `README.md`). Le jalon 2 (SPEC §4) demande de brancher des
agents LLM sur l'environnement existant, sans retoucher les règles du jeu, pour
les bras A1 (communication interdite), A2 (JSON libre) et A3 (JSON sous budget).

Ce document couvre uniquement **A1**. C'est le sous-projet le plus risqué à
valider en premier : il n'ajoute que la boucle de décision LLM (perception →
action), sans la complexité du canal de message. A2/A3 réutiliseront cette base
et ajouteront le canal une fois qu'on sait que le modèle choisi tient la boucle
seule.

**Modèle** : servi par LM Studio headless (API compatible OpenAI), modèle
`google/gemma-4-12b` (GGUF, Q6_K), capacité `tool_use` confirmée (function
calling). **Décision du 2026-09-25 : hébergé sur un pod RunPod plutôt qu'en
local**, exactement comme anticipé ci-dessus — même client, même modèle, seul
`base_url` change (voir Configuration). Justification : ne pas monopoliser la
machine locale pour une campagne de ≥ 30 épisodes × plusieurs centaines
d'appels, et exploiter le crédit RunPod disponible.

- Pod `jhbk30ligdrozo` (RTX 4090 24 Go, secure cloud, 0,74 $/h, DC EU-RO-1).
- `base_url` : `https://jhbk30ligdrozo-1234.proxy.runpod.net/v1`.
- `model` (identifiant exact chargé côté serveur) : `gemma-4-12b-a1`.
- **Le pod n'a pas de coupure automatique** : `runpodctl` v2.12.0 n'expose
  aucun `--terminate-after`/`--stop-after` sur `pod create`/`pod update`, et
  l'API REST v2 n'a pas non plus de champ d'arrêt programmé. Il facture en
  continu tant qu'il n'est pas arrêté/supprimé manuellement
  (`pod-action` → `stop` ou `terminate`). À couper explicitement en fin de
  session de travail.
- Concurrence GPU : `lms load` supporte `--parallel N` (continuous batching
  llama.cpp), un seul serveur suffit — pas besoin de plusieurs instances.
  Chargé avec `--gpu max --context-length 81920 --parallel 32`, ce qui occupe
  ~89,7 % de la VRAM (22 Go / 24,5 Go) de façon stable. Vérifié en conditions
  réelles : 32 appels `move` forcés simultanés → 13,5 s (32/32 valides) contre
  ~80 s en séquentiel, soit ~5,9×. Le harnais LLM (`scripts/run_llm.py`) doit
  donc envoyer ses appels **en parallèle** (pool de ~32 requêtes concurrentes)
  vers ce **seul** `base_url`, pas de round-robin entre plusieurs endpoints.
- **Écart avec l'API OpenAI standard** : LM Studio rejette la forme objet de
  `tool_choice` (`{"type":"function","function":{"name":"move"}}` →
  `400 Invalid tool_choice type: 'object'`). Utiliser la forme chaîne
  `tool_choice: "required"` à la place — équivalent ici puisqu'un seul outil
  (`move`) est exposé, mais **le code client (`chase/llm/client.py`) et son
  test (`test_decide_forces_the_move_tool`) doivent être écrits contre cette
  forme, pas celle de la spec OpenAI générique.**
- Le modèle émet un champ `reasoning_content` (chaîne de pensée) avant l'appel
  d'outil, qui consomme une partie du budget `max_tokens` : mesuré à 99/142
  tokens de complétion sur l'appel de vérification. Sous le budget par défaut
  de 150 (§ Configuration), mais à surveiller au pilote (Task 7) avec de vrais
  prompts de perception, plus longs que le test de vérification.

## Objectif du sous-projet

Produire un bras A1 fonctionnel et mesurable : une politique `PursuerPolicy`
pilotée par le LLM local, sans aucun canal de communication entre poursuivants
(chacun ignore tout de son coéquipier, position comprise — même contrainte que
R1), avec journalisation du coût de raisonnement séparé du coût de
communication (nul par construction pour A1, mais le schéma de log doit rester
stable pour A2/A3).

## Non-objectifs

- Le canal de message JSON (A2/A3) : hors périmètre, design séparé à venir.
- Toute modification des règles du jeu, de `chase/env.py`, `chase/belief.py`,
  `chase/graph.py`, `chase/target.py` : jalon 1 figé (SPEC §6).
- Comparaison chiffrée A1 vs R1/R2 dans ce document : viendra avec le rapport
  du jalon 2, pas avec ce design.
- Support d'un fournisseur LLM fondamentalement différent (API Anthropic,
  OpenAI hébergé, vLLM/TGI) : le client reste écrit contre l'API compatible
  OpenAI de LM Studio ; changer de fournisseur serait un changement de
  client ultérieur, pas traité ici. Héberger ce même serveur LM Studio sur un
  pod RunPod loué plutôt qu'en local n'est qu'un changement d'adresse
  (`base_url`/`model`), déjà traité ci-dessus (§ Modèle), pas un changement de
  fournisseur.

## Architecture

Nouveau paquet `chase/llm/`, à côté de `chase/policies.py`. Rien d'autre dans
`chase/` ne change.

```
chase/llm/
  __init__.py
  client.py    # appel LM Studio (tool calling forcé), retries, mesure tokens/latence
  prompts.py   # construction du prompt système + résumé de perception par direction
  policy.py    # LLMPursuers(PursuerPolicy), croyance individuelle, boucle de décision
  logging.py   # structures de log par pas / par épisode
```

`LLMPursuers` implémente la même interface que `GreedyPursuers`
(`reset`, `update`, `act`, `beliefs`) et se substitue à elle dans
`chase/runner.py::run_episode` sans changement du runner — `run_episode`
prend déjà une `PursuerPolicy` en paramètre.

### `chase/llm/client.py`

- SDK `openai` pointé sur `base_url` du pod RunPod (voir § Modèle ; `api_key`
  factice, requis par le SDK mais ignoré par LM Studio).
- Un seul outil exposé : `move(direction: enum[STAY,NORTH,SOUTH,EAST,WEST], reasoning: string)`.
  `tool_choice` forcé sur cette fonction : le modèle ne peut pas répondre en
  texte libre hors du schéma. **Forme exacte : `tool_choice="required"`**
  (chaîne, pas l'objet `{"type":"function","function":{"name":"move"}}` de la
  spec OpenAI générique — LM Studio le rejette avec `400`, voir § Modèle).
  Équivalent ici puisqu'un seul outil est déclaré.
- Le SDK `openai` est thread-safe pour des appels concurrents (chaque
  `chat.completions.create` est indépendant) : c'est ce qui permet la
  parallélisation décrite en § Exécution parallèle de la campagne, sans
  changement à `client.py` lui-même — la concurrence se pilote depuis
  `policy.py` (deux poursuivants) et `scripts/run_llm.py` (plusieurs épisodes).
- `temperature=0` par défaut (reproductibilité à seed égale — nécessaire pour
  que les comparaisons entre bras et entre runs restent interprétables).
- Budget de raisonnement : `max_tokens` sur la réponse complète (outil +
  `reasoning`), valeur par défaut 150 tokens, **identique pour tous les
  poursuivants et tous les bras LLM** (A1/A2/A3) — c'est le contrôle figé
  « même budget de raisonnement individuel » de la SPEC §4. Valeur ajustable
  après le pilote si 150 s'avère trop court pour un tool call valide.
- Retries : jusqu'à 2 tentatives sur erreur réseau/timeout ou réponse qui ne
  contient pas d'appel d'outil valide (JSON malformé, direction hors enum).
  Après épuisement des retries : repli sur `STAY`, épisode non interrompu,
  échec journalisé (pas d'exception qui ferait perdre l'épisode).
- Chaque appel est un **tour isolé** : pas d'historique de conversation
  accumulé au fil de l'épisode. Le prompt (système + perception du pas
  courant) est reconstruit à chaque appel. Le coût par appel reste donc borné
  quelle que soit la longueur de l'épisode — propriété nécessaire vu le
  contexte de 13056 tokens.
- Retourne une structure `LLMCallResult` : direction choisie, texte de
  raisonnement, tokens prompt/complétion, latence, nombre de retries,
  indicateur de repli.

### `chase/llm/prompts.py`

- **Prompt système** (statique, un seul texte figé pour A1) : rappelle les
  règles pertinentes pour la décision (un poursuivant par appel, un pas =
  une direction ou immobile, capture par occupation ou croisement de case,
  pas de communication avec le coéquipier), la sémantique des 5 actions, et
  le format de sortie attendu (appel de l'outil `move` uniquement).
- **Résumé de perception** (par pas, par poursuivant) — format « résumé
  structuré par direction » retenu pendant le brainstorming, pas de grille
  ASCII :
  - pour chacune des 4 directions cardinales : praticable ou non (mur), et ce
    qui y est visible (case libre, cible visible à telle distance, densité de
    croyance — nombre de cases candidates visibles dans cette direction) ;
  - si la cible est visible : sa position relative exacte ;
  - taille de l'ensemble candidat courant, et sa localisation résumée
    (liste explicite des cases si `len(cand) <= track_threshold`, sinon
    décompte + direction dominante, en réutilisant les mêmes seuils que
    `chase/policies.py` pour rester comparable).
  - La croyance utilisée est la **croyance individuelle** du poursuivant
    (mêmes fonctions `chase/belief.py::propagate`/`observe` que R1, appliquées
    indépendamment par poursuivant — aucune fusion, aucune position de
    coéquipier transmise : c'est ce qui fait de ce bras un bras « sans
    communication »).
- Le prompt est un texte structuré (listes de champs nommés), pas une prose
  narrative ni une grille — objectif : lisible par un humain qui relit un log,
  sans ambiguïté spatiale pour un modèle de 12B.

### `chase/llm/policy.py`

- `LLMPursuers(cfg: ChaseConfig, llm_cfg: LLMConfig)` implémente
  `PursuerPolicy`.
- `reset` : réinitialise une croyance individuelle par poursuivant (comme
  `GreedyPursuers(fused=False)`).
- `update(percepts)` : propage puis observe la croyance individuelle de
  chaque poursuivant avec `belief.propagate`/`belief.observe`, exactement
  comme la branche `fused=False` de `GreedyPursuers.update`.
- `act(percepts)` : pour chaque poursuivant, construit le prompt via
  `prompts.py`, appelle `client.py`, journalise le résultat, retourne la
  liste de `Move`. Reste **séquentiel** sur les 2 poursuivants (inchangé) : la
  parallélisation se fait au niveau des épisodes, pas des poursuivants (voir
  § Exécution parallèle de la campagne) — inutile de complexifier `act()` et
  la synchronisation des `step_logs`/tests pour un gain borné à 2× quand le
  parallélisme inter-épisodes donne déjà le facteur mesuré (~5-6× à 32
  requêtes simultanées) sans toucher à l'ordonnancement interne d'un épisode.
- `beliefs()` : renvoie les croyances individuelles (réutilisé par les tests
  d'invariant existants, sur le modèle de `tests/test_chase.py`).

### `chase/llm/logging.py`

Structures de log pensées pour rester **stables entre A1/A2/A3** (le champ de
coût de communication existe déjà, à 0 pour A1) :

- Par pas et par poursuivant : `prompt_tokens`, `completion_tokens`,
  `latency_ms`, `move` choisi, `reasoning` (texte tronqué à N caractères pour
  le log), `retries`, `fallback` (bool), `message_tokens` (0 pour A1).
- Par épisode : agrégats (somme/moyenne des champs ci-dessus) + les champs
  déjà produits par `EpisodeResult` (`captured`, `steps`, `confinement`,
  `mean_confinement`) + `wall_time_s` de l'épisode.

Pas de nouvelle classe de résultat : on étend `EpisodeResult` (ou on
l'enveloppe) plutôt que de dupliquer les champs de `chase/runner.py`.

## Exécution parallèle de la campagne

Objectif : maximiser les épisodes traités par heure de pod louée (0,74 $/h),
donc le débit vers le serveur LM Studio à 32 créneaux concurrents (§ Modèle),
plutôt que de laisser le GPU idle entre deux appels séquentiels.

Un seul niveau de parallélisme, au niveau des épisodes
(`scripts/run_llm.py`) : plusieurs épisodes tournent en parallèle, chacun
dans son propre thread avec son propre `ChaseEnv` (aucun état partagé entre
épisodes, seeds différentes) et sa propre boucle séquentielle (2 appels
poursuivants par pas, `act()` inchangé — voir § `policy.py`).
`ThreadPoolExecutor` avec un nombre de workers configurable (`--concurrency`,
défaut à fixer au pilote, ordre de grandeur 16–32 : chaque épisode ne
consomme qu'un créneau à la fois puisque `act()` reste séquentiel).

Pas de parallélisme intra-épisode (entre les 2 poursuivants d'un même pas) :
inutile pour saturer les 32 créneaux du serveur, et ça éviterait de
complexifier `act()`/ses tests (ordonnancement des `step_logs`, thread-safety
du faux client) pour un gain marginal que le parallélisme inter-épisodes
couvre déjà.

Un seul `base_url` (le pod), pas de round-robin entre plusieurs endpoints —
confirmé au provisioning : un unique serveur avec `--parallel 32` sature déjà
~90 % de la VRAM disponible, pas besoin de plusieurs instances LM Studio.

Le calcul des métriques par épisode (`EpisodeLLMStats`) ne change pas : chaque
épisode reste agrégé indépendamment, la parallélisation ne touche que
l'ordonnancement, pas le contenu des logs.

## Configuration

Deux dataclasses séparées, pour ne pas mélanger règles du jeu et
configuration du client LLM :

- `ChaseConfig` (existant, inchangé) : un **profil dédié plus petit** pour le
  jalon 2 (grille et/ou `max_steps` réduits par rapport au jalon 1), défini
  après le pilote de mesure de latence (voir Plan de validation). Les valeurs
  retenues seront figées et versionnées de la même façon que le jalon 1
  (README + résultats), pas laissées en paramètre libre.
- `LLMConfig` (nouveau, `chase/llm/client.py` ou `chase/llm/config.py`) :
  `base_url`, `model`, `temperature`, `max_tokens` (budget de raisonnement),
  `timeout_s`, `max_retries`. Valeurs par défaut correspondant au pod RunPod
  en place (§ Modèle) : `base_url="https://jhbk30ligdrozo-1234.proxy.runpod.net/v1"`,
  `model="gemma-4-12b-a1"`.

`requirements.txt` gagne une dépendance : `openai` (client compatible avec
l'API exposée par LM Studio).

## Flux de données (un pas, un poursuivant)

1. `runner.py` construit le `Percept` (position, masque de visibilité,
   cible vue ou non) — inchangé.
2. `LLMPursuers.update` propage puis observe la croyance individuelle du
   poursuivant avec les fonctions de `belief.py`.
3. `LLMPursuers.act` → `prompts.py` traduit `Percept` + croyance individuelle
   en résumé structuré par direction.
4. `client.py` envoie le prompt système (statique) + le résumé (par pas) à
   LM Studio, avec l'outil `move` forcé.
5. Réponse valide → `Move` extrait, log rempli. Réponse invalide/erreur après
   retries → repli `STAY`, log marqué `fallback=True`.
6. `runner.py` applique le mouvement comme pour R0/R1/R2 — aucun changement
   à l'environnement.

## Gestion des erreurs

| Cas | Comportement |
|---|---|
| Timeout réseau / serveur LM Studio indisponible | Retry (jusqu'à 2), puis `STAY` + log `fallback=True` |
| Réponse sans appel d'outil, ou direction hors enum | Retry (jusqu'à 2), puis `STAY` + log `fallback=True` |
| `reasoning` dépasse le budget de tokens | Troncature à la journalisation uniquement ; n'affecte pas le choix d'action déjà décodé |
| Échecs répétés sur tout un épisode (serveur down) | L'épisode continue en `STAY` systématique plutôt que de crasher tout un run de calibration — visible dans les logs agrégés (taux de `fallback` élevé), pas une erreur silencieuse |

Aucune exception ne doit interrompre `evaluate()` : un poursuivant qui
échoue à chaque pas dégrade le score (visible dans les métriques), il ne doit
pas faire perdre les épisodes déjà calculés dans un run parallèle.

## Tests

- **Unitaires (`tests/test_llm_policy.py`)** : faux client LLM déterministe
  (implémente la même interface que `client.py`, renvoie des `LLMCallResult`
  en boîte) — pas d'appel réseau, pas de dépendance à LM Studio en cours
  d'exécution, pas de coût d'inférence.
  - La croyance individuelle produite par `LLMPursuers.update` est identique
    à celle de `GreedyPursuers(fused=False)` sur la même séquence de percepts
    (réutilisation directe de `belief.py`, doit être un test de non-régression
    fort).
  - Une direction retournée par le faux client se traduit bien en le `Move`
    attendu, y compris `STAY`.
  - Le repli sur `STAY` se déclenche bien après épuisement des retries sur
    une réponse invalide simulée, et le log correspondant a `fallback=True`.
  - Les champs de log attendus sont tous présents et du bon type.
- **Validation contre le vrai modèle (script séparé, pas dans `pytest`)** :
  un script pilote (`scripts/run_llm.py --pilot`, ou option dédiée) qui lance
  1 à 3 épisodes réels contre le pod RunPod, mesure le temps par appel et par
  épisode (séquentiel puis à la concurrence cible), et sert à fixer le profil
  `ChaseConfig` et `--concurrency` du jalon 2 avant de lancer la campagne
  complète (≥ 30 épisodes par point, comme les autres bras).

## Plan de validation (avant la campagne complète)

1. Pilote 1 à 3 épisodes **séquentiels** sur le profil de config par défaut du
   jalon 1 (ou un profil réduit de départ), mesure du temps réel par
   appel/épisode contre le pod RunPod.
2. Ajustement du profil `ChaseConfig` du jalon 2 (grille/`max_steps`) pour que
   la campagne complète (≥ 30 épisodes, comme R0/R1/R2) reste dans un temps
   raisonnable.
3. Ajustement éventuel du budget de raisonnement (`max_tokens`) si 150 s'avère
   trop court pour obtenir un appel d'outil valide de façon fiable — à
   surveiller en particulier à cause de `reasoning_content` (§ Modèle).
4. Pilote **concurrent** (`--concurrency`, quelques épisodes en parallèle) :
   confirmer que le débit augmente bien comme mesuré au provisioning (~5-6× à
   32 requêtes simultanées) sans dégrader le taux de `fallback`, et choisir la
   valeur de `--concurrency` retenue pour la campagne complète.
5. Une fois les paramètres stables (config jeu + LLM + concurrence) : campagne
   complète A1, résultats et paramètres figés et versionnés (même traitement
   que `results/jalon1.md`).
6. **Couper le pod** (`pod-action` → `stop` ou `terminate`) une fois la
   campagne terminée ou en fin de session de travail — pas de coupure
   automatique (§ Modèle).

## Points ouverts pour A2/A3 (non traités ici)

- Schéma du message JSON et son budget (A3) — SPEC §3 impose au minimum :
  ensemble candidat pondéré, intention de déplacement sur N pas, issue
  couverte.
- Unité de mesure du coût de communication : nombre de positions occupées
  dans le contexte du récepteur (SPEC §3) — à instrumenter au moment d'ajouter
  le canal, pas maintenant puisque A1 n'en a pas.
- Est-ce qu'un deuxième outil (`send_message`) s'ajoute à l'appel existant, ou
  un appel séparé après la décision de mouvement ? À trancher avec la mesure
  réelle de `reasoning`/latence obtenue sur A1.
