"""Bras A1 (jalon 2) : pilote de mesure puis campagne contre LM Studio.

    python -m scripts.run_llm --pilot --episodes 1
    python -m scripts.run_llm --episodes 30 --out results/jalon2_a1.md
    python -m scripts.run_llm --pilot --episodes 2 --set max_steps=60 size=15
"""

from __future__ import annotations

import argparse
import json
import time

from chase.config import ChaseConfig
from chase.env import ChaseEnv, episode_rngs
from chase.llm.config import LLMConfig
from chase.llm.logging import EpisodeLLMStats
from chase.llm.policy import LLMPursuers
from chase.policies import Percept
from chase.target import ScriptedTarget


def _percepts(env: ChaseEnv, vis) -> list[Percept]:
    return [Percept(env.pursuer_pos(p), vis[p], env.target_seen_by(vis[p]))
            for p in range(env.cfg.n_pursuers)]


def run_llm_episode(cfg: ChaseConfig, llm_cfg: LLMConfig, seed: int):
    env = ChaseEnv(cfg)
    env.reset(seed=seed)
    rngs = episode_rngs(seed)
    policy = LLMPursuers(cfg, llm_cfg)
    policy.reset(env.graph, rngs["pursuers"])
    target = ScriptedTarget(env.graph, rngs["target"], cfg.target_memory, cfg.target_cycle_bias)

    t0 = time.monotonic()
    vis = env.visibility()
    policy.update(_percepts(env, vis))
    while not env.done:
        moves = policy.act(_percepts(env, vis))
        pursuers = [env.pursuer_pos(p) for p in range(cfg.n_pursuers)]
        moves.append(target.act(env.target_pos, vis[env.target_id], pursuers))
        vis = env.step_moves(moves)
        policy.update(_percepts(env, vis))
    wall_time_s = time.monotonic() - t0

    stats = EpisodeLLMStats.from_steps(policy.step_logs, wall_time_s)
    return env.captured, env.step_count, env.confinement, stats


def _parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        out[key] = json.loads(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], help="surcharges clé=valeur de ChaseConfig")
    parser.add_argument("--pilot", action="store_true",
                        help="affiche le temps par épisode/appel sans écrire de tableau")
    parser.add_argument("--out", help="écrit le tableau Markdown dans ce fichier")
    args = parser.parse_args()

    cfg = ChaseConfig().replace(**_parse_overrides(args.set))
    llm_cfg = LLMConfig()

    rows = []
    for seed in range(args.first_seed, args.first_seed + args.episodes):
        t0 = time.monotonic()
        captured, steps, confinement, stats = run_llm_episode(cfg, llm_cfg, seed)
        episode_wall_s = time.monotonic() - t0
        rows.append((seed, captured, steps, confinement, stats, episode_wall_s))
        print(f"seed {seed}: capturé={captured} pas={steps} "
              f"temps={episode_wall_s:.1f}s latence_moy={stats.mean_latency_ms:.0f}ms "
              f"replis={stats.fallback_count}", flush=True)

    if args.pilot:
        return

    lines = [
        f"Seeds {args.first_seed}..{args.first_seed + args.episodes - 1} "
        f"({args.episodes} épisodes), T = {cfg.max_steps}.",
        "",
        "| Seed | Capture | Pas | Confinement à T | Temps mural (s) | Latence moy. (ms) | Replis |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed, captured, steps, confinement, stats, wall_s in rows:
        lines.append(
            f"| {seed} | {'oui' if captured else 'non'} | {steps} | {confinement:.3f} "
            f"| {wall_s:.1f} | {stats.mean_latency_ms:.0f} | {stats.fallback_count} |")
    lines += ["", "Paramètres du jeu :", "", "```json", json.dumps(cfg.as_dict(), indent=2), "```"]
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
