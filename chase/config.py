"""Paramètres du jeu. Les valeurs par défaut sont celles retenues au jalon 1.

Toute modification après le jalon 1 est une dérive de règles (SPEC §6) :
changer une valeur ici impose de refaire la calibration et de la versionner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ChaseConfig:
    # Topologie
    size: int = 27               # grille carrée, impaire (cellules du labyrinthe aux coordonnées impaires)
    n_loops: int = 3             # cycles ajoutés à l'arbre couvrant
    min_loop_len: int = 12       # longueur minimale (en cases) d'un cycle ajouté

    # Agents
    n_pursuers: int = 2
    view_size: int = 5           # champ de vision MiniGrid (carré orienté, impair, >= 3)
    min_spawn_dist: int = 12     # distance de chemin minimale cible <-> chaque poursuivant au départ

    # Épisode
    max_steps: int = 250         # T : instant de mesure du confinement

    # Cible scriptée (figée après le jalon 1)
    target_memory: int = 6       # nombre de pas pendant lesquels la cible fuit un poursuivant perdu de vue
    target_cycle_bias: float = 0.5  # biais de la marche aléatoire vers les cycles

    # Poursuivants R1 / R2
    comm_delay: int = 1          # R2 : délai (en pas) du canal de fusion ; 1 = conditions des bras LLM
    track_threshold: int = 8     # |croyance| sous laquelle on passe de l'exploration à la poursuite

    def replace(self, **kwargs) -> "ChaseConfig":
        return replace(self, **kwargs)

    def as_dict(self) -> dict:
        return asdict(self)
