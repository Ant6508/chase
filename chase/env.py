"""Environnement de poursuite construit sur MultiGrid.

Hérité de MultiGrid : grille, rendu, pas simultané, masque de visibilité
MiniGrid avec occultation par les murs (`see_through_walls=False`).
Ajouté ici : labyrinthe seedé, déplacements absolus, règle de capture,
projection des masques de visibilité en coordonnées monde, croyance d'équipe
(qui sert au calcul du confinement).

Agents : indices `0 .. n_pursuers-1` pour les poursuivants, `n_pursuers` pour la cible.
"""

from __future__ import annotations

import numpy as np
from multigrid.base import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.agent import AgentState
from multigrid.core.world_object import Wall
from multigrid.utils.obs import gen_obs_grid_vis_mask, get_view_exts

from .belief import observe, propagate
from .config import ChaseConfig
from .graph import MazeGraph
from .maze import generate_maze
from .moves import MOVE_DELTAS, MOVE_TO_DIR, Move


def episode_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Flux aléatoires indépendants par usage : la carte et le placement ne
    dépendent que de la seed, jamais de la politique évaluée."""
    return {name: np.random.default_rng([seed, k])
            for k, name in enumerate(("map", "target", "pursuers"))}


class ChaseEnv(MultiGridEnv):

    def __init__(self, cfg: ChaseConfig = ChaseConfig(), **kwargs):
        self.cfg = cfg
        super().__init__(
            agents=cfg.n_pursuers + 1,
            grid_size=cfg.size,
            max_steps=cfg.max_steps,
            see_through_walls=False,
            agent_view_size=cfg.view_size,
            allow_agent_overlap=True,
            **kwargs,
        )
        self.target_id = cfg.n_pursuers
        self._seed = 0
        self._view_maps = self._build_view_maps(cfg.view_size)

    # ------------------------------------------------------------------ carte

    def reset(self, seed: int = 0, **kwargs):
        self._seed = seed
        self.captured = False
        out = super().reset(seed=seed, **kwargs)
        vis = self.visibility()
        self.team_belief = self._team_update(self.free.copy(), vis)
        return out

    def _gen_grid(self, width: int, height: int):
        cfg = self.cfg
        rng = episode_rngs(self._seed)["map"]
        self.free = generate_maze(cfg.size, cfg.n_loops, cfg.min_loop_len, rng)
        self.graph = MazeGraph(self.free)
        self.grid = Grid(width, height)
        for x, y in np.argwhere(~self.free):
            self.grid.set(int(x), int(y), Wall())

        # Poursuivants sur des cases distinctes, cible loin de chacun d'eux.
        g = self.graph
        pursuers = rng.choice(g.n, size=cfg.n_pursuers, replace=False)
        far = np.all(g.dist[pursuers] >= cfg.min_spawn_dist, axis=0)
        choices = np.flatnonzero(far)
        if len(choices) == 0:
            raise ValueError("min_spawn_dist trop grand pour cette carte")
        target = rng.choice(choices)
        for agent, cell in zip(self.agents, [*pursuers, target]):
            agent.state.pos = g.cells[cell]
            agent.state.dir = int(rng.integers(4))

    # -------------------------------------------------------------- dynamique

    def handle_actions(self, actions):
        before = [tuple(a.state.pos) for a in self.agents]
        for i, move in actions.items():
            move = Move(move)
            if move == Move.STAY:
                continue
            agent = self.agents[i]
            agent.state.dir = MOVE_TO_DIR[move]
            dx, dy = MOVE_DELTAS[move]
            x, y = agent.state.pos
            if self.free[x + dx, y + dy]:
                agent.state.pos = (x + dx, y + dy)
        after = [tuple(a.state.pos) for a in self.agents]

        t = self.target_id
        for p in range(self.cfg.n_pursuers):
            same_cell = after[p] == after[t]
            crossed = after[p] == before[t] and after[t] == before[p]
            if same_cell or crossed:
                self.captured = True
        return {i: 0 for i in range(self.num_agents)}

    def step_moves(self, moves: list[Move]) -> list[np.ndarray]:
        """Applique un pas simultané, met à jour la croyance d'équipe et renvoie
        la visibilité de chaque agent après le pas."""
        self.step(dict(enumerate(moves)))
        vis = self.visibility()
        self.team_belief = self._team_update(propagate(self.team_belief, self.free), vis)
        return vis

    @property
    def done(self) -> bool:
        return self.captured or self.step_count >= self.max_steps

    # ------------------------------------------------------------- perception

    @staticmethod
    def _build_view_maps(view: int):
        """Pour chaque orientation, correspondance case de vue -> décalage monde,
        reproduisant la rotation de `multigrid.utils.obs.gen_obs_grid`."""
        i, j = np.meshgrid(np.arange(view), np.arange(view), indexing="ij")
        i, j = i.ravel(), j.ravel()
        maps = []
        for rot in range(4):
            if rot == 0:
                ir, jr = i, j
            elif rot == 1:
                ir, jr = j, view - i - 1
            elif rot == 2:
                ir, jr = view - i - 1, view - j - 1
            else:
                ir, jr = view - j - 1, i
            maps.append((i, j, ir, jr))
        return maps

    def visibility(self) -> list[np.ndarray]:
        """Masque de visibilité MiniGrid (occultation comprise) de chaque agent,
        exprimé en coordonnées monde `[x, y]`.

        Poursuivants : champ orienté MiniGrid. Cible : union des quatre
        orientations (vision à 360°, occultation comprise). Une cible aveugle
        dans son dos se ferait prendre par n'importe quel poursuivant qui la
        suit, sans coordination, ce qui viderait la mesure de son sens.
        """
        states = np.array(self.agent_states.view(np.ndarray))
        out = self._world_masks(states, self.cfg.view_size)
        target = self.target_id
        for d in range(4):
            states[target, AgentState.DIR] = d
            out[target] |= self._world_masks(states, self.cfg.view_size)[target]
        return out

    def _world_masks(self, states: np.ndarray, view: int) -> list[np.ndarray]:
        maps = self._view_maps if view == self.cfg.view_size else self._build_view_maps(view)
        dirs = states[:, AgentState.DIR]
        masks = gen_obs_grid_vis_mask(self.grid.state, states, view)
        tops = get_view_exts(dirs, states[:, AgentState.POS], view)
        out = []
        for a in range(self.num_agents):
            i, j, ir, jr = maps[(int(dirs[a]) + 1) % 4]
            x, y = tops[a, 0] + i, tops[a, 1] + j
            inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            seen = masks[a][ir, jr] & inside
            world = np.zeros((self.width, self.height), dtype=bool)
            world[x[seen], y[seen]] = True
            out.append(world)
        return out

    def target_seen_by(self, visible: np.ndarray) -> tuple[int, int] | None:
        pos = self.target_pos
        return pos if visible[pos] else None

    def _team_update(self, belief: np.ndarray, vis: list[np.ndarray]) -> np.ndarray:
        for p in range(self.cfg.n_pursuers):
            belief = observe(belief, vis[p], self.target_seen_by(vis[p]))
        return belief

    # ------------------------------------------------------------- accesseurs

    @property
    def target_pos(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.agents[self.target_id].state.pos)

    def pursuer_pos(self, p: int) -> tuple[int, int]:
        return tuple(int(v) for v in self.agents[p].state.pos)

    @property
    def confinement(self) -> float:
        """Taille de l'ensemble candidat d'équipe normalisée par les cases libres.
        Vaut 0 après capture."""
        if self.captured:
            return 0.0
        return float(self.team_belief.sum()) / float(self.free.sum())
