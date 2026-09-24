"""Boucle d'épisode et agrégation des métriques du jalon 1."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import ChaseConfig
from .env import ChaseEnv, episode_rngs
from .policies import Percept, PursuerPolicy, make_policy
from .target import ScriptedTarget


@dataclass
class EpisodeResult:
    seed: int
    captured: bool
    steps: int
    confinement: float       # à l'instant T (0 si capture)
    mean_confinement: float  # moyenne sur l'épisode, pas capturés comptés à 0


def _percepts(env: ChaseEnv, vis: list[np.ndarray]) -> list[Percept]:
    return [Percept(env.pursuer_pos(p), vis[p], env.target_seen_by(vis[p]))
            for p in range(env.cfg.n_pursuers)]


def run_episode(cfg: ChaseConfig, policy: PursuerPolicy, seed: int,
                env: ChaseEnv | None = None,
                on_step: Callable[[ChaseEnv, PursuerPolicy], None] | None = None) -> EpisodeResult:
    env = env or ChaseEnv(cfg)
    env.reset(seed=seed)
    rngs = episode_rngs(seed)
    policy.reset(env.graph, rngs["pursuers"])
    target = ScriptedTarget(env.graph, rngs["target"], cfg.target_memory, cfg.target_cycle_bias)

    vis = env.visibility()
    policy.update(_percepts(env, vis))
    history = [env.confinement]
    if on_step:
        on_step(env, policy)
    while not env.done:
        moves = policy.act(_percepts(env, vis))
        pursuers = [env.pursuer_pos(p) for p in range(cfg.n_pursuers)]
        moves.append(target.act(env.target_pos, vis[env.target_id], pursuers))
        vis = env.step_moves(moves)
        policy.update(_percepts(env, vis))
        history.append(env.confinement)
        if on_step:
            on_step(env, policy)

    # Après capture, le confinement reste nul jusqu'à T.
    history += [0.0] * (cfg.max_steps + 1 - len(history))
    return EpisodeResult(seed, env.captured, env.step_count, env.confinement,
                         float(np.mean(history)))


def _run_chunk(cfg: ChaseConfig, policy_name: str, seeds: list[int]) -> list[EpisodeResult]:
    env = ChaseEnv(cfg)
    return [run_episode(cfg, make_policy(policy_name, cfg), s, env) for s in seeds]


def evaluate(cfg: ChaseConfig, policy_name: str, seeds: range,
             workers: int | None = None) -> dict:
    """Évalue une politique sur des seeds. Les épisodes sont indépendants et
    déterministes : le parallélisme ne change pas les résultats."""
    workers = workers or os.cpu_count() or 1
    seeds = list(seeds)
    if workers > 1 and len(seeds) > 1:
        chunks = [seeds[k::workers] for k in range(workers)]
        with ProcessPoolExecutor(workers) as pool:
            parts = pool.map(_run_chunk, [cfg] * workers, [policy_name] * workers, chunks)
        results = sorted((r for part in parts for r in part), key=lambda r: r.seed)
    else:
        results = _run_chunk(cfg, policy_name, seeds)
    captured = [r for r in results if r.captured]
    return {
        "policy": policy_name,
        "episodes": len(results),
        "capture_rate": len(captured) / len(results),
        "confinement": float(np.mean([r.confinement for r in results])),
        "mean_confinement": float(np.mean([r.mean_confinement for r in results])),
        "steps_to_capture": (float(np.median([r.steps for r in captured]))
                             if captured else float("nan")),
        "results": results,
    }
