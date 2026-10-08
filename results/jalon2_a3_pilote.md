# Jalon 2, bras A3 — pilote sur RunPod, seeds 0 à 7

**Quand.** Le 2026-10-08, de 15h34 à 17h47 UTC.

**Où.**
- Pod `0jj7z07rmajhuv` : A40 48 Go, Secure, CA-MTL-1, 0,49 $/h.
  - C'est la même carte et le même data center que la campagne A2.
  - L'ancien pod `14rh3c8qgzdpmi` ne pouvait plus démarrer, faute de GPU libre sur sa machine.
- LM Studio, moteur `llama.cpp-linux-x86_64-nvidia-cuda12-avx2@2.41.0`, 8 créneaux de 8 192 tokens,
  13 911 Mo de VRAM. C'est la pile d'A2 (13 913 Mo).
- Code `1c0d5b2`.

```bash
campaign.sh start a3 --arm A3 --max-tokens 5600 --timeout 900 --first-seed 0 --episodes 8 \
    --concurrency 8 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 min_spawn_dist=6 \
    --trace /workspace/results/a3/trace
```

**Fichiers.** [`jalon2_a3/`](jalon2_a3/) contient `journal.jsonl`, `table.md`, `run.log` et `trace/`.
Le rapatriement a été vérifié par empreintes MD5. La campagne repartira de ce journal pour les
seeds 8 à 29.

## Résultats par seed

| Seed | A3 | A1bis | A2 | A2' | R1 | R2 | P3 | Positions par message (A3) |
|---|---|---|---|---|---|---|---|---|
| 0 | pas 21 | pas 17 | pas 24 | pas 20 | pas 35 | pas 22 | pas 22 | 23,1 |
| 1 | pas 19 | pas 19 | pas 21 | pas 32 | non | pas 27 | pas 27 | 23,2 |
| 2 | pas 27 | pas 27 | pas 27 | pas 27 | pas 30 | pas 30 | pas 30 | 23,4 |
| 3 | pas 47 | pas 58 | pas 47 | non | pas 39 | pas 39 | pas 39 | 21,6 |
| 4 | pas 31 | pas 33 | pas 31 | pas 31 | pas 33 | pas 33 | pas 33 | 21,8 |
| 5 | non | pas 25 | non | non | pas 27 | pas 35 | pas 35 | 24,2 |
| 6 | pas 38 | non | non | pas 38 | non | pas 56 | non | 22,8 |
| 7 | pas 50 | pas 38 | non | non | pas 55 | pas 22 | pas 26 | 20,8 |
| **Captures** | **7 / 8** | 7 / 8 | 5 / 8 | 5 / 8 | 6 / 8 | 8 / 8 | 7 / 8 | **22,5** |

Ces 8 seeds ne permettent aucune conclusion (McNemar : p = 0,50 contre A2, p = 1,00 contre A1bis).

- **La seed 5 est le goulot de la phase de capture.** A3 y voit la cible 77 fois, la rapproche à
  chaque fois, et ne la prend pas.

## Ce que montrent les traces

| | A1bis (30 seeds) | A2 (30 seeds) | A3, pilote (8 seeds) |
|---|---|---|---|
| Décisions | 2 714 | 2 732 | 586 |
| Replis | 0 | 68 (2,5 %) | **0** |
| Décisions relancées | 0,1 % | 16,2 % | **0,3 %** (2, réponses coupées à 5 600 tokens) |
| Messages rejetés (borne, champ en trop, type) | — | — | **0** |
| Tokens de prompt par décision | 1 152 | 1 661 | 1 431 |
| Tokens de pensée par décision | 492 | 2 528 | **1 060** (médiane 874) |
| Latence moyenne | 52 s | 158 s | 79 s |
| Positions par message | — | 138 | **22,5** (médiane 22, de 15 à 37) |
| Noms inconnus | — | 75 dans 36 messages | 7 dans 7 messages |
| Allers-retours | 30 % | 18 % | 18 % |
| Cible visible : coup qui la rapproche | 100 % | 100 % | 100 % |
| Cible cachée : vers l'issue la plus probable | 81 % | 79 % | 84 % |

Les chiffres d'A1bis et d'A2 viennent de `results/jalon2_a2.md` et de `results/jalon2_a2p.md`.

- **La borne est respectée sans effort.**
  - `intention` compte 1 lieu dans 76 messages, 2 dans 497 et 3 dans 13.
  - Aucun message n'a été rejeté.
- **Le message reste bien sous le budget annoncé dans la spec (36 positions), à une exception près.**
  - Le maximum, 37 positions, vient d'un nom de case plus long que dans l'exemple de la spec.
- **La pensée double par rapport à A1bis, mais reste 2,4 fois plus courte qu'en A2.** C'est
  donc bien la liste des lieux, et non le canal, qui faisait le coût d'A2.
- **L'ordre des clés est presque toujours alphabétique** : `cible`, `intention`, `moi` dans 545
  messages sur 586. L'ordre du prompt n'apparaît que dans 41 messages. Le harnais garde l'ordre de
  l'émetteur.
- **`cible` est renseignée dans 149 messages.** Sa véracité, inventée ou relayée, sera mesurée au
  rapport.

## Règle d'arrêt (spec A3, § Validation, étape 2)

| Critère | Seuil | Pilote |
|---|---|---|
| Décisions finies en repli | 5 % au plus | 0 % |
| Décisions relancées pour message invalide | 10 % au plus | 0 % |

**Verdict : la règle est remplie, la campagne peut être lancée.**

## Coût et durée

- **Le pilote a pris 2 h 13** : 8 épisodes en parallèle, de 58 min à 2 h 13 chacun.
- **Le pod a tourné 6 h 10 au total, soit environ 3,02 $.**
  - Le déploiement a pris environ 10 min.
  - Le pod est ensuite resté **inactif 3 h 45 après le pilote**, environ 1,85 $ perdus.
  - **La cause est la surveillance locale.** Elle cherchait le processus avec un `pgrep -f`
    lancé dans un `bash -c` dont la ligne de commande contenait le motif : elle se trouvait
    elle-même. Il faut surveiller avec `campaign.sh status`, qui ne se détecte pas lui-même.
- **Le pod est en pause depuis 21h35 UTC.**
- **Estimation de la campagne, seeds 8 à 29 :**
  - 22 épisodes, d'environ 1 h 35 en moyenne, sur 8 créneaux : environ 5 h à 6 h 30 de pod,
    soit 2,50 à 3,20 $ ;
  - il faut y ajouter environ 10 min de redéploiement si le pod ne peut pas redémarrer sur sa
    machine.
