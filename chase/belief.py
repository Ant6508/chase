"""Ensemble candidat de la cible (poursuite avec visibilité sur graphe).

Une croyance est un masque booléen `[x, y]` : True là où la cible peut encore
se trouver compte tenu de tout ce qui a été observé.
"""

from __future__ import annotations

import numpy as np


def propagate(belief: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Un pas de temps : toute case atteignable en un pas redevient candidate."""
    out = belief.copy()
    out[1:, :] |= belief[:-1, :]
    out[:-1, :] |= belief[1:, :]
    out[:, 1:] |= belief[:, :-1]
    out[:, :-1] |= belief[:, 1:]
    return out & free


def observe(belief: np.ndarray, visible: np.ndarray,
            target_seen: tuple[int, int] | None) -> np.ndarray:
    """Intègre une observation : cible vue -> une seule case, sinon on retire le vu."""
    if target_seen is not None:
        out = np.zeros_like(belief)
        out[target_seen] = True
        return out
    return belief & ~visible


def diffuse(prob: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Un pas de marche aléatoire paresseuse : chaque case répartit sa masse
    uniformément entre elle-même et ses voisines libres. Le support du résultat
    est exactement `propagate(support(prob))`."""
    deg = np.ones(free.shape)
    deg[1:, :] += free[:-1, :]
    deg[:-1, :] += free[1:, :]
    deg[:, 1:] += free[:, :-1]
    deg[:, :-1] += free[:, 1:]
    share = np.where(free, prob / deg, 0.0)
    out = share.copy()
    out[1:, :] += share[:-1, :]
    out[:-1, :] += share[1:, :]
    out[:, 1:] += share[:, :-1]
    out[:, :-1] += share[:, 1:]
    return out * free


def observe_prob(prob: np.ndarray, visible: np.ndarray,
                 target_seen: tuple[int, int] | None) -> np.ndarray:
    """Pendant probabiliste de `observe`, renormalisé."""
    if target_seen is not None:
        out = np.zeros_like(prob)
        out[target_seen] = 1.0
        return out
    out = prob * ~visible
    total = out.sum()
    return out / total if total > 0 else out
