# Jalon 3 (préparation) : RecursiveMAS accepte-t-il gemma-4-12b ?

Vérification du 2026-09-29, avant de concevoir A2.

- **Pourquoi maintenant.** La SPEC impose le même modèle de base pour tous les
  bras, et le bras A4 (RecursiveMAS) exige les poids du modèle. Changer de
  modèle maintenant coûte une campagne A1 ; le faire après A2 en coûterait deux.
- **Ce qui a été examiné.** Le dépôt
  [RecursiveMAS/RecursiveMAS](https://github.com/RecursiveMAS/RecursiveMAS),
  commit `cbfcaab` du 2026-09-28, et la config de
  [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)
  (Apache 2.0, non soumis à autorisation d'accès).

## Verdict

**Oui, à trois conditions :**
1. `transformers` 5.10 ou plus récent ;
2. un correctif du rendu du gabarit de chat ;
3. un GPU d'environ 48 Go.

Aucune ne remet en cause le choix du modèle. La carte locale (12 Go, AMD) est
exclue pour A4.

## Ce que fait RecursiveMAS

Cela tranche la question ouverte du jalon 3 de la SPEC : **les liens sont
appris** (cas « encodeur ou projecteur appris »), et le modèle reste gelé.

- **Lien interne**, un par agent : un MLP résiduel à deux couches, avec
  normalisation, de la taille de l'état caché. Il apprend à transformer
  l'état caché de dernière couche en plongement du token suivant (perte
  cosinus), pour que l'agent puisse relire ses propres états.
- **Lien externe**, entre deux agents. Il est appris par entropie croisée sur
  la réponse du récepteur, le gradient traversant le récepteur gelé.
- **Le message** est la suite des états cachés de l'émetteur sur les tokens
  de sa réponse. Il passe par le lien interne puis le lien externe, est
  tronqué à `max_latent_tokens` vecteurs (80 par défaut), et est injecté à la
  place d'un marqueur dans le prompt du récepteur.
- **Le volume du message est donc un nombre de positions** dans le contexte
  du récepteur : c'est l'unité de mesure de la SPEC. `max_latent_tokens` est
  la molette de la courbe de Pareto.
- **L'entraînement demande du texte :** celui du message de l'émetteur, et la
  réponse attendue du récepteur. Pour le jeu, ce serait les messages JSON du
  bras A2 et un coup cible (celui de R2, ou celui d'A2, à décider). **A2 doit
  donc venir avant A4, et lui fournira ses données.**
- **Aucun des quatre schémas publiés ne correspond au jeu.** Ce sont
  séquentiel (3 agents en cycle), mélange, distillation et délibération ; le
  jeu demande 2 agents symétriques qui échangent à chaque pas. Il faudra
  écrire ce schéma, sur le modèle de `train/outer/sequential.py`.

## Compatibilité avec gemma-4-12b

| Point | Résultat |
|---|---|
| Classe chargée par `AutoModelForCausalLM` | `Gemma4UnifiedForConditionalGeneration`, multimodal (texte, image, audio) |
| Version de `transformers` | `gemma4_unified` est absent de la 5.3.0, épinglée par RecursiveMAS, et présent à partir de la 5.10.0. Testé avec la 5.17.0 : rien n'a cassé, car le dépôt n'utilise que `AutoModelForCausalLM`, `AutoTokenizer` et les ordonnanceurs |
| Entrée par vecteurs (`inputs_embeds`), sans `token_type_ids` | oui ; le cas particulier Gemma 3 du dépôt n'a pas lieu d'être |
| Plongements par couche (comme Gemma 3n) | absents (`hidden_size_per_layer_input = 0`) |
| Génération latente avec cache (vecteur réinjecté comme entrée suivante) | oui (`DynamicCache`) |
| **Gabarit de chat** | **incompatible en l'état** (détail ci-dessous) |

Le problème de gabarit : en génération, le gabarit de Gemma 4 ajoute un bloc
de pensée vide (`<|channel>thought\n<channel|>`) après `<|turn>model\n`. Le
rendu qui contient déjà la réponse ne l'a pas. Or RecursiveMAS suppose que le
prompt est un préfixe du texte complet, pour savoir où commencent les tokens
de réponse.

- **Boucle externe** (`common.py::build_stage_with_slot`) : aucun token de
  réponse n'est étiqueté, et la perte vaut `nan`.
- **Boucle interne** (`data.py`, ligne 281) : les premiers tokens de chaque
  réponse sont retirés de la perte, sans aucune erreur.
- **Correctif :** construire le texte complet comme prompt + réponse + fin de
  tour. Il tient en quelques lignes, et aligne l'entraînement sur ce que le
  modèle voit en génération.

## Test de fumée

[`scripts/recursivemas_smoke.py`](../scripts/recursivemas_smoke.py) : même
classe et même config que le 12B, 48 couches ramenées à 6, largeur 3840
ramenée à 64, poids aléatoires, vrai tokenizer et vrai gabarit, sur CPU. Il
fait passer le code d'entraînement du dépôt (`train/model.py`,
`train/outer/common.py`).

| Étape | Sans correctif | Avec `--patch-template` |
|---|---|---|
| 1. Chargement par `AutoModelForCausalLM` | `Gemma4UnifiedForConditionalGeneration` | idem |
| 2. Boucle interne : gradient sur le lien, modèle gelé | oui | oui |
| 3. Boucle externe : 8 vecteurs injectés, tokens de réponse étiquetés | **0, perte `nan`** | 4, perte 12,54 (= ln 262 144 à peu près : poids aléatoires), gradient sur le lien, récepteur gelé |
| 4. Génération latente, 3 pas avec cache | — | oui |

Le test ne dit rien de la qualité des liens : il vérifie que le chemin de code
passe.

## Mémoire

- **Le modèle :** 11,96 milliards de paramètres, vision et audio compris, soit
  environ 24 Go en bf16 par copie.
- **Deux agents :** le dépôt charge un modèle par agent, soit 48 Go pour
  2 poursuivants. Les deux sont gelés et identiques : une seule copie
  partagée suffit, au prix d'une petite modification.
- **Les liens :** environ 30 M de paramètres pour un lien interne, 74 M pour
  un lien externe (3840 → 7680 → 3840, plus le résiduel). Avec Adam en fp32,
  cela fait moins de 2 Go.
- **GPU nécessaire :** 48 Go est confortable, pour l'entraînement (avec
  checkpointing) comme pour l'inférence. 40 Go est possible. 24 à 32 Go ne
  suffisent pas sans quantification. Le pic de 15,29 Go annoncé par le papier
  concerne des agents de 1 à 4 milliards de paramètres.

## Conséquences pour la suite

1. **gemma-4-12b peut rester le modèle de base de tous les bras.** Le papier
   utilise des modèles de 1 à 9 milliards de paramètres, parmi lesquels
   Gemma 3 4B.
2. **« Même modèle de base » demandera un bras témoin.** A1 et A2 tournent en
   Q6_K sous llama.cpp ; A4 tournera en bf16 sous `transformers`. Il faudra
   rejouer au moins un bras texte avec la même pile qu'A4 et sur le même GPU,
   par exemple en réécrivant le client LLM au-dessus de `transformers`.
3. **A2 d'abord.** Ses messages JSON fournissent le texte d'entraînement du
   canal latent. Le choix du coup cible (R2 ou A2) est à trancher dans le
   design d'A4.
