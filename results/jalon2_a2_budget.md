# Jalon 2, bras A2 — budget de génération et réglage du serveur

Diagnostic du 2026-09-29 et du 2026-09-30, avant les situations témoins. Deux causes indépendantes
ont fait échouer les premiers appels A2. Elles fixent le budget et le lieu des campagnes.

## 1. Mémoire du GPU en local

Au premier essai des situations témoins, le 2026-09-29, presque tous les appels se sont repliés
sur `STAY`. Le journal de LM Studio en donne la raison.

- **Délai dépassé.** Les requêtes parties à 14h03:54 ont été coupées à 14h10:34, soit exactement
  les 400 s du délai client. Les générations ont été interrompues vers 254 tokens.
- **Débit effondré.** On a mesuré environ 4 tokens/s au total, contre 15,4 tokens/s le matin
  même, au rejeu d'A1v3.
- **Cause : la carte de 12 Go était partagée.** Le modèle était chargé avec 12 288 tokens de
  contexte et 4 créneaux. D'autres programmes occupaient 3,7 Go de VRAM : League of Legends et son
  client, l'émulateur MEmu, l'affichage Windows, Steam, Discord. Le modèle débordait alors de
  6,7 Go en mémoire partagée.
- **Les programmes fermés, le débit revient** à environ 14 tokens/s au total, sur 3 créneaux.
  Mais 4,4 Go restent en mémoire partagée.
- **Fin de l'essai.** LM Studio a ensuite été fermé en cours d'essai : le modèle a été déchargé
  à 14h11:23. Cet essai est sans valeur et n'est pas conservé.

## 2. Ce que coûte un appel A2

Une fois la carte libérée, 3 appels A2 simultanés ont été lancés, avec `max_tokens=1600` et
4 096 tokens par créneau. Les trois ont fini en `finish_reason=length`, avec 1 597 tokens de pensée
sur 1 600. Le modèle rédige tout le message dans sa pensée avant de l'écrire, et il se relit
parfois : « Let me double check the `candidates` list ». Il n'arrive donc jamais à l'appel d'outil.

Les mêmes 3 situations ont été rejouées sans coupure : un seul créneau de 8 192 tokens,
`max_tokens=4000`, pas de relance. Ce sont 3 situations « vue par le coéquipier » tirées de
parties de P2, sur les seeds 30 à 32 :

| Seed | Prompt | Complétion | dont pensée | Message | Lieux listés / lieux candidats de l'émetteur | Coup = P2 | Durée (1 créneau) |
|---|---|---|---|---|---|---|---|
| 30 | 1 666 | 3 308 | 3 050 | 177 | 22 / 28 | oui | 519 s |
| 31 | 1 701 | 2 589 | 2 292 | 233 | 35 / 36 | oui | 393 s |
| 32 | 1 612 | 2 872 | 2 675 | 124 | 15 / 25 | oui | 433 s |

- **Le message est utilisable.** Les trois messages sont valides, et les trois coups sont ceux
  de P2. Dans ces situations, seul le coéquipier voyait la cible : il fallait donc lire le message.
- **Le coût est de 2 600 à 3 300 tokens par appel**, contre environ 450 en A1v3, soit 6 à 7 fois
  plus. Presque tout est de la pensée : le modèle recopie et vérifie la liste des lieux. Ce
  surcoût est un résultat en soi : c'est le prix, pour le texte, de la sérialisation d'un
  ensemble, précisément ce que le canal latent prétend éviter.
- **La complétude varie** : un message liste de 60 % à 97 % des lieux candidats de son émetteur.
  Or un lieu absent vaut « vu vide ». Il faudra le mesurer sur la campagne.

## Décisions (utilisateur, 2026-09-30)

- **Budget.** Plafond de génération commun de **4 000 tokens** pour A1bis et A2. A1bis n'en
  consomme qu'environ 450 : le plafond ne le gêne pas, et le contrôle « même budget » est
  respecté. A1 garde 1 600, pour que la campagne A1v3 reste reproductible.
- **Serveur.** 8 créneaux de 8 192 tokens, soit un contexte de 65 536 : environ 2 500 tokens de
  prompt au plus, plus 4 000 de complétion.
- **Lieu.** Les deux campagnes, A2 et son témoin A1bis, tournent sur RunPod, avec la même pile.
  En local, un appel A2 prend de 6 à 9 minutes : une campagne durerait plus d'une semaine.
- **Estimation**, sur la base d'environ 120 tokens/s au total mesurés sur RunPod en
  septembre :
  - A2 : environ 9 millions de tokens, soit environ 20 h ;
  - A1bis : environ 1,4 million, soit environ 3 h.
