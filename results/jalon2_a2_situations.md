# Jalon 2, bras A2 — situations témoins

Validation de l'étape 2 de la spec, faite le 2026-09-30.

**Matériel.** Pod RunPod `otdj10kjtl7a4g` : A40 48 Go, Secure, CA-MTL-1. LM Studio avec le moteur
CUDA, 8 créneaux de 8 192 tokens. Code `882e0e4`.

**Situations.** Elles viennent de parties de P2 jouées sur les seeds 30 à 129, hors campagne.
Chaque situation est présentée au modèle deux fois :
- en A1bis : la perception seule ;
- en A2 : la même perception, plus le message du coéquipier écrit par P2.

```bash
python -m scripts.situations_a2 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --base-url http://127.0.0.1:1234/v1 --timeout 900 --max-tokens 4000 --concurrency 8 \
    --out /workspace/results/situations_a2.jsonl
```

Données : [`jalon2_a2_situations.jsonl`](jalon2_a2_situations.jsonl), et le journal
[`jalon2_a2_situations.log`](jalon2_a2_situations.log).

## Bilan

| Famille | A2 = P2 | A1bis = P2 | Messages valides | Couverture des candidates |
|---|---|---|---|---|
| Cible vue par le seul coéquipier | 3 / 4 | 4 / 4 | 3 / 4 | 69 % |
| Issue la plus probable déjà vidée par le coéquipier | 0 / 3 | 0 / 3 | 3 / 3 | 93 % |
| Cible entre les deux poursuivants (tenaille) | 3 / 3 | 3 / 3 | 3 / 3 | 100 % |

- **A2 joue comme A1bis dans les 9 situations où il n'y a pas de repli.** Le seul écart est la
  seed 30 (vue par le coéquipier) : A2 y reste sur place après 3 tentatives coupées, alors
  qu'A1bis joue le coup de P2.
- **Le message est lu, mais pas exploité par intersection.** À la seed 30 (issue vidée), par
  exemple, la pensée d'A2 relève la position et l'intention du coéquipier. Elle résume ses
  candidates en « Mostly in the southern/central part of the map », puis suit sa propre
  probabilité (89 % au sud). Elle ne déduit jamais qu'un lieu absent du message a été vu vide.
  Dans les trois situations de cette famille, les coups d'A2 et d'A1bis sont identiques.
- **Les familles « vue par le coéquipier » et « tenaille » discriminent mal.** Le coup de P2 y
  coïncide avec celui qu'A1bis déduit de sa seule perception.

## Coût et fiabilité d'A2

| | A1bis | A2 |
|---|---|---|
| Tokens de complétion par décision (appel retenu) | 262 à 1 074 | 508 à 3 996 |
| Tentatives coupées à 4 000 tokens (`finish_reason=length`) | 0 | 5 sur 16 |
| Tentatives rejetées : `message` passé comme chaîne JSON et non comme objet | — | 2 sur 16 |
| Décisions en repli | 0 / 10 | 1 / 10 |
| Latence par appel, 8 créneaux occupés | — | 29 à 194 s |

- **Le budget de 4 000 tokens ne suffit pas toujours.** La pensée d'A2 va de 378 à 3 755 tokens, et
  certaines tentatives dépassent le plafond. Or, à température 0, une relance reproduit souvent la
  coupure.
- **Deux relances sur seize** viennent du `message` écrit sous forme de chaîne JSON. La validation
  stricte le refuse, alors que le contenu était correct.
