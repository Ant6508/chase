import numpy as np
import pytest

from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.graph import MazeGraph
from chase.maze import generate_maze
from chase.policies import make_policy
from chase.runner import run_episode

CFG = ChaseConfig(max_steps=120)


def _edges(free):
    return int((free[1:, :] & free[:-1, :]).sum() + (free[:, 1:] & free[:, :-1]).sum())


@pytest.mark.parametrize("n_loops", [0, 1, 2, 4])
def test_maze_is_tree_plus_loops(n_loops):
    free = generate_maze(25, n_loops, 12, np.random.default_rng(3))
    # Graphe connexe dont le nombre cyclomatique vaut exactement n_loops.
    assert _edges(free) - free.sum() + 1 == n_loops
    g = MazeGraph(free)
    assert (g.dist < np.iinfo(np.int32).max).all()
    assert (g.on_cycle.sum() == 0) == (n_loops == 0)


def test_map_depends_only_on_seed():
    maps = []
    for name in ("R0", "R2"):
        env = ChaseEnv(CFG)
        run_episode(CFG, make_policy(name, CFG), seed=7, env=env)
        maps.append(env.free.copy())
    assert (maps[0] == maps[1]).all()


def test_visibility_is_occluded_and_includes_self():
    env = ChaseEnv(CFG)
    env.reset(seed=1)
    v = CFG.view_size
    for a, vis in enumerate(env.visibility()):
        pos = tuple(env.agents[a].state.pos)
        assert vis[pos]
        # Occultation : on ne voit pas au travers des murs, donc le vu est
        # strictement plus petit que la zone de vue (carré orienté pour un
        # poursuivant, union des quatre orientations pour la cible).
        area = v * v if a < CFG.n_pursuers else (2 * v - 1) ** 2
        assert vis.sum() < area


@pytest.mark.parametrize("name", ["R1", "R2"])
@pytest.mark.parametrize("delay", [0, 1])
def test_beliefs_always_contain_target(name, delay):
    """Invariant de la poursuite ensembliste : la vraie position de la cible
    appartient toujours à toute croyance, individuelle ou d'équipe."""
    cfg = CFG.replace(comm_delay=delay)

    def check(env, policy):
        t = env.target_pos
        if env.captured:
            return
        assert env.team_belief[t]
        for b in policy.beliefs():
            assert b[t]

    for seed in range(5):
        run_episode(cfg, make_policy(name, cfg), seed, on_step=check)


def test_deterministic():
    a = run_episode(CFG, make_policy("R2", CFG), seed=4)
    b = run_episode(CFG, make_policy("R2", CFG), seed=4)
    assert a == b
