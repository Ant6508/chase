# Chase — spécification d'amorce

Environnement de poursuite coopérative en observabilité partielle, destiné à
mesurer le gain en coût de communication apporté par **RecursiveMAS**
(communication inter-agents en espace latent, arXiv 2604.25917) face à des
agents LLM communiquant en texte structuré.

Ce document est l'amorce du projet. Il fige les décisions déjà prises et
délimite ce qui reste ouvert. Il n'est pas un plan d'implémentation détaillé :
l'architecture du code est à la main de l'implémenteur, les règles du jeu et le
protocole de mesure ne le sont pas.

---

## 1. Objectif

Produire une **courbe de Pareto coût de communication × performance**, comparant
plusieurs canaux de communication à politique de décision identique.

Le résultat attendu n'est pas un chiffre unique mais une comparaison de courbes.
Un gain revendiqué sur un point isolé n'a aucune valeur.

### Pourquoi cet environnement

Deux conditions doivent être satisfaites simultanément, et c'est le seul motif
de son design :

1. **Ignorer l'information d'autrui doit punir immédiatement.** La perception et
   l'action d'un agent ne doivent pas être alignées : agir sans message doit
   revenir à agir aveugle.
2. **Le message utile doit être volumineux et continu.** Si l'information utile
   tient en un fait (« cible en B3 »), le texte la code en cinq tokens et le
   canal latent n'a aucune marge à gagner. L'avantage de RecursiveMAS n'est pas
   de compresser des faits mais de transporter ce que le texte sérialise mal :
   un ensemble, une distribution, un plan.

Un précédent démonstrateur (MOBA-lite) a été abandonné faute de satisfaire la
condition 1 : chaque agent y observait et agissait dans la même zone, donc une
politique gloutonne locale était déjà quasi optimale.

---

## 2. Environnement

### Base technique

Partir de **MultiGrid** (`github.com/ini/multigrid`), extension multi-agents de
MiniGrid, API Gymnasium compatible ParallelEnv de PettingZoo. On en hérite :

- la génération et le rendu de grilles,
- le champ de vision par agent **avec occultation par les murs**
  (`see_through_walls = False`),
- le pas simultané multi-agents.

Ne pas réimplémenter ces briques.

### Topologie

La topologie décide du résultat avant les agents. La théorie « cops and
robbers » établit que le nombre de poursuivants nécessaires à une capture
garantie dépend de la structure du graphe : un arbre se capture à un seul
poursuivant, un graphe riche en cycles peut rester inexpugnable à deux quel que
soit leur niveau de coordination.

Décisions figées :

- Labyrinthe **majoritairement arborescent**, avec un nombre **paramétrable et
  faible** de cycles ajoutés (`n_loops`, défaut 2).
- Couloirs de largeur 1.
- Grille carrée, défaut 25 × 25.
- Générateur seedé, carte identique entre bras expérimentaux.

`n_loops` est une molette de réglage de difficulté, pas un détail cosmétique.

### Agents

- **2 poursuivants** (extensible à 3 si le jalon 1 échoue, voir §6).
- **1 cible**, politique **scriptée**. Non apprise, non adversariale-optimale.
  Une cible apprise détruirait l'attribution des écarts entre bras.
- Cible **pas plus rapide** que les poursuivants (1 case par pas pour tous).
- Pas de temps discret, actions simultanées, **un message de délai** entre
  émission et réception.

Politique de la cible : fuite gloutonne loin du poursuivant visible le plus
proche, marche aléatoire biaisée vers les cycles en l'absence de poursuivant
visible. À figer tôt et à ne plus retoucher.

### Perception

**Décision importante, qui remplace un modèle antérieur à relèvements bruités.**

Chaque poursuivant reçoit son masque de visibilité MiniGrid : l'ensemble des
cases qu'il voit, murs compris, occultation appliquée. Il en déduit par
complément l'**ensemble des cases où la cible peut encore se trouver**.

Ce choix est délibéré :

- il est déjà implémenté, donc gratuit ;
- il est **ensembliste et exact**, pas probabiliste et bruité. Un LLM sait
  intersecter deux ensembles ; il échoue à propager une postérieure bayésienne
  multimodale à travers des occultations sur une dizaine de pas. Donner
  l'angle brut à l'agent reviendrait à faire échouer tous les bras
  identiquement et à ne rien mesurer ;
- il produit nativement le volume recherché : le contenu utile d'un message
  est un **sac de cases candidates**, pas un fait.

C'est la formulation classique de la poursuite avec visibilité sur graphe
(Guibas, LaValle et al.).

L'ensemble candidat se **propage** entre deux pas : toute case atteignable
depuis une case candidate redevient candidate, moins ce qui est observé vide.
C'est cette propagation qui rend l'information périssable et impose de
communiquer en continu.

Le bruit angulaire reste une extension possible, à n'activer que si les écarts
entre bras s'avèrent trop francs (§6).

### Objectif et score

**Ne pas utiliser la capture binaire comme métrique principale.** Elle produit
une falaise 0/1 qui masque les différences entre bras et rend le réglage
impossible.

Métrique principale : **confinement**, soit la taille de l'ensemble candidat à
l'instant `T`, normalisée par la taille de la grille libre. Score gradué,
continu, sensible au réglage.

Métriques secondaires : taux de capture, pas jusqu'à capture.

La capture reste définie et instrumentée : la cible est prise quand toutes ses
cases de sortie sont simultanément occupées ou observées.

---

## 3. Canal de communication et bras expérimentaux

L'agent raisonne pour lui, agit dans le jeu, et n'émet vers son coéquipier
qu'un message **distinct et explicitement délimité**. Sans cette séparation, le
raisonnement interne est compté comme de la communication et la mesure est
fausse. **Seul ce canal est remplacé d'un bras à l'autre.**

| Bras | Canal |
|------|-------|
| A0 | agent solo contrôlant les deux poursuivants (borne haute) |
| A1 | communication interdite |
| A2 | texte structuré (JSON) non contraint |
| A3 | texte structuré sous budget imposé à la génération |
| A4 | RecursiveMAS (canal latent) |

### Sur le JSON — revendication à annoncer honnêtement

Un schéma de message figé revient à concevoir soi-même le protocole de
communication. Ce n'est pas illégitime, mais cela change ce qui est démontré :
on ne montre plus « les agents découvrent quoi se dire », on montre « le canal
latent est moins cher que le JSON à charge utile fixée ». C'est un protocole
plus propre, à condition de l'écrire ainsi dans le rapport et de laisser au
bras JSON **assez de champs pour ne pas être estropié** (au minimum : ensemble
candidat pondéré, intention de déplacement sur N pas, issue couverte).

### Unité de mesure

**Nombre de positions occupées dans le contexte du récepteur.** « N tokens
contre un vecteur » n'est pas une comparaison, c'est le premier point qu'un
relecteur attaquera. Les tokens peuvent figurer en titre puisque c'est la
revendication du papier ; l'instrumentation compte des positions de contexte.

### Variable indépendante

Le **volume** de message est ce que balaie la courbe de Pareto, pas un
paramètre à fixer. La **fréquence** est fixée à un message par pas. Pas de
balayage 2D.

---

## 4. Jalon 1 — calibration sans LLM (bloquant)

**Rien d'autre ne doit être écrit avant que ce jalon ne soit franchi.** Aucun
LLM, aucun harnais de comptage, aucun RecursiveMAS. Quelques centaines de
lignes, exécution en secondes.

Trois politiques scriptées :

- **R0** — aléatoire uniforme.
- **R1** — gloutonne locale, sans communication : chaque poursuivant poursuit
  son propre ensemble candidat.
- **R2** — gloutonne à croyance parfaitement fusionnée : les deux poursuivants
  partagent l'intersection de leurs ensembles candidats et se répartissent les
  issues.

**Porte de sortie :** l'écart R1 → R2 doit être franc **mais pas caricatural**.

- Cible visée : R1 autour de **30–40 %** de taux de capture, R2 au-dessus de
  **85 %**.
- Si R2 échoue, **le labyrinthe est en cause, pas les agents** : réduire
  `n_loops`, réduire la grille, ou passer à 3 poursuivants.
- Si R1 réussit trop, réduire le rayon de vision ou augmenter la taille de la
  grille.
- Si l'écart est de type 0 % contre 90 %, la démonstration devient suspecte :
  remonter le rayon de vision.

Molettes de réglage, par ordre de préférence : rayon de vision, `n_loops`,
taille de grille, nombre de poursuivants.

Ce cycle de réglage coûte des secondes tant qu'aucun LLM n'est dans la boucle.
Il coûterait des heures et de l'argent après. C'est précisément pourquoi il
vient en premier.

**Livrable du jalon :** un tableau R0/R1/R2 sur ≥ 30 épisodes seedés, et les
paramètres retenus, figés et versionnés.

---

## 5. Jalons suivants

### Jalon 2 — harnais LLM, bras A1 à A3

- Actions **nommées sémantiquement**, jamais des identifiants numériques.
- Observation au format lisible, stable entre bras.
- Journalisation par épisode : coût de communication **séparé** du coût de
  raisonnement interne, confinement, taux de capture, pas jusqu'à capture,
  temps mural.
- Contrôles figés : même modèle de base, mêmes seeds, mêmes cartes, même espace
  d'action, même budget de raisonnement individuel, ≥ 30 épisodes par point.

### Jalon 3 — bras A4, RecursiveMAS

**Question ouverte à trancher par lecture du papier avant de chiffrer :**
comment RecursiveMAS produit-il ses vecteurs latents ?

- Directement depuis les états cachés → aucun entraînement, brancher et mesurer.
- Encodeur ou projecteur appris → il y a un apprentissage, mais il porte sur le
  **canal**, pas sur la politique de jeu.

La réponse change le chiffrage de plusieurs semaines.

### Jalon 4 — interprétabilité et visuel

**Sonde linéaire sur les vecteurs latents**, pour décoder l'ensemble candidat de
la cible. C'est à la fois la parade principale au reproche d'environnement
taillé sur mesure et le meilleur visuel du projet : afficher côte à côte
l'ensemble candidat vrai et celui décodé depuis le vecteur latent.

---

## 6. Risques identifiés et parades

| Risque | Parade |
|--------|--------|
| « L'environnement est taillé pour faire gagner le canal latent » | Ne pas caricaturer la baseline texte (§3). Publier la courbe entière, jamais un point. Sonde linéaire (jalon 4). |
| La topologie rend la capture impossible ou triviale | Jalon 1 bloquant. `n_loops` et rayon de vision comme molettes. |
| Le LLM ne fait pas la fusion d'information | Perception ensembliste, pas de relèvements bruités (§2). |
| La métrique binaire masque les écarts | Confinement comme métrique principale. |
| Le raisonnement interne est compté comme communication | Canal de message explicitement délimité (§3). |
| Dérive des règles en cours de projet | Règles figées ici, versionnées, non retouchées après le jalon 1. |

---

## 7. Hors périmètre

- Classes ou capacités hétérogènes entre poursuivants. Chaque asymétrie est un
  facteur de confusion : un gain en environnement hétérogène ne dirait pas si
  le latent compresse mieux ou gère mieux l'hétérogénéité des rôles.
- Cible apprise ou adversariale.
- Plusieurs cibles simultanées.
- Rendu 3D. Le spectaculaire d'une démo vient de la lisibilité du contraste
  entre deux runs, pas de la richesse du moteur.

---

## 8. Instruction à l'implémenteur

Écrire le simulateur et les trois heuristiques du jalon 1, **et rien d'autre**.
Revenir avec le tableau R0/R1/R2 avant d'ouvrir le moindre fichier lié aux LLM.
