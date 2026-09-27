"""Bras A1 (jalon 2) : pilote de mesure puis campagne contre LM Studio (pod RunPod).

    python -m scripts.run_llm --pilot --episodes 1
    python -m scripts.run_llm --pilot --episodes 16 --concurrency 16
    python -m scripts.run_llm --episodes 30 --concurrency 24 --out results/jalon2_a1.md
    python -m scripts.run_llm --pilot --episodes 2 --set max_steps=60 size=15

--concurrency > 1 exécute plusieurs épisodes en parallèle (threads), chacun
avec son propre ChaseEnv (aucun état partagé) — c'est le levier qui exploite
les créneaux concurrents du serveur LM Studio (voir design doc § Exécution
parallèle de la campagne). --concurrency 1 (défaut) reste séquentiel. La
progression par pas (--pilot) n'est affichée qu'en séquentiel : au-delà, les
lignes de plusieurs épisodes s'entrelaceraient sans rien apporter.

--journal écrit chaque épisode dans un fichier JSONL dès qu'il se termine ;
relancer la même commande avec le même journal ne rejoue que les seeds
manquantes (un run tué ne perd que ses épisodes en cours). --base-url vise un
autre serveur que celui de LLMConfig, par ex. http://127.0.0.1:1234/v1 quand
le script tourne sur le pod lui-même (voir scripts/pod/).

--trace DIR écrit, pour le diagnostic, un fichier DIR/seed_<n>.jsonl par
épisode : une ligne par décision (perception envoyée, pensée du modèle, coup,
positions vraies du poursuivant et de la cible), au fil de l'eau.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, replace
from typing import Callable

from chase.config import ChaseConfig
from chase.env import ChaseEnv, episode_rngs
from chase.llm.client import LLMClient
from chase.llm.config import LLMConfig
from chase.llm.journal import EpisodeJournal
from chase.llm.logging import EpisodeLLMStats, StepLog
from chase.llm.policy import LLMPursuers
from chase.policies import Percept
from chase.target import ScriptedTarget


def _percepts(env: ChaseEnv, vis) -> list[Percept]:
    return [Percept(env.pursuer_pos(p), vis[p], env.target_seen_by(vis[p]))
            for p in range(env.cfg.n_pursuers)]


def _trace_record(seed: int, step: int, target: tuple[int, int], log: StepLog) -> dict:
    return {"seed": seed, "step": step, "pursuer": log.pursuer,
            "pos": [int(v) for v in log.pos], "target": [int(v) for v in target],
            "move": log.move.name, "fallback": log.fallback, "retries": log.retries,
            "completion_tokens": log.completion_tokens, "perception": log.perception,
            "reasoning": log.reasoning, "thinking": log.thinking}


def run_llm_episode(cfg: ChaseConfig, llm_cfg: LLMConfig, seed: int,
                     on_step: Callable[[ChaseEnv, LLMPursuers], None] | None = None,
                     trace_path: str | None = None, client: LLMClient | None = None):
    env = ChaseEnv(cfg)
    env.reset(seed=seed)
    rngs = episode_rngs(seed)
    policy = LLMPursuers(cfg, llm_cfg, client=client)
    policy.reset(env.graph, rngs["pursuers"])
    target = ScriptedTarget(env.graph, rngs["target"], cfg.target_memory, cfg.target_cycle_bias)

    t0 = time.monotonic()
    vis = env.visibility()
    policy.update(_percepts(env, vis))
    if on_step:
        on_step(env, policy)
    with open(trace_path, "w", encoding="utf-8") if trace_path else nullcontext() as trace:
        while not env.done:
            step, target_pos = env.step_count, env.target_pos
            moves = policy.act(_percepts(env, vis))
            if trace:
                for log in policy.step_logs[-cfg.n_pursuers:]:
                    trace.write(json.dumps(_trace_record(seed, step, target_pos, log),
                                           ensure_ascii=False) + "\n")
                trace.flush()
            pursuers = [env.pursuer_pos(p) for p in range(cfg.n_pursuers)]
            moves.append(target.act(env.target_pos, vis[env.target_id], pursuers))
            vis = env.step_moves(moves)
            policy.update(_percepts(env, vis))
            if on_step:
                on_step(env, policy)
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
    parser.add_argument("--concurrency", type=int, default=1,
                        help="épisodes exécutés en parallèle (threads) ; 1 = séquentiel")
    parser.add_argument("--out", help="écrit le tableau Markdown dans ce fichier")
    parser.add_argument("--journal",
                        help="JSONL : un épisode par ligne dès qu'il se termine ; reprise si existant")
    parser.add_argument("--base-url", help="serveur LM Studio (défaut : LLMConfig.base_url)")
    parser.add_argument("--trace",
                        help="dossier : une trace JSONL pas à pas par seed, pour le diagnostic")
    parser.add_argument("--timeout", type=float,
                        help="délai par appel en secondes (défaut : LLMConfig.timeout_s) ; "
                             "à allonger sur un GPU lent")
    args = parser.parse_args()

    cfg = ChaseConfig().replace(**_parse_overrides(args.set))
    llm_cfg = LLMConfig()
    if args.base_url:
        llm_cfg = replace(llm_cfg, base_url=args.base_url)
    if args.timeout:
        llm_cfg = replace(llm_cfg, timeout_s=args.timeout)
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    if args.trace:
        os.makedirs(args.trace, exist_ok=True)

    journal = None
    done: dict[int, dict] = {}
    if args.journal:
        # l'adresse du serveur n'entre pas dans les paramètres : même modèle en local
        # ou sur le pod, un run peut reprendre de l'un à l'autre
        llm_params = {k: v for k, v in asdict(llm_cfg).items() if k != "base_url"}
        journal = EpisodeJournal(args.journal, {"game": cfg.as_dict(), "llm": llm_params})
        done = {s: r for s, r in journal.completed().items() if s in seeds}
        if done:
            print(f"reprise : {len(done)} épisode(s) déjà dans {args.journal}, "
                  f"{len(seeds) - len(done)} à jouer", flush=True)

    def _run_one(seed: int):
        t0 = time.monotonic()

        on_step = None
        if args.pilot and args.concurrency == 1:
            def on_step(env, policy, t0=t0):
                elapsed = time.monotonic() - t0
                print(f"  pas {env.step_count}/{cfg.max_steps} temps_écoulé={elapsed:.1f}s",
                      flush=True)

        trace_path = os.path.join(args.trace, f"seed_{seed}.jsonl") if args.trace else None
        captured, steps, confinement, stats = run_llm_episode(
            cfg, llm_cfg, seed, on_step=on_step, trace_path=trace_path)
        episode_wall_s = time.monotonic() - t0
        if journal:
            journal.append({"seed": seed, "captured": captured, "steps": steps,
                            "confinement": confinement, "wall_time_s": episode_wall_s,
                            "stats": asdict(stats)})
        print(f"seed {seed}: capturé={captured} pas={steps} "
              f"temps={episode_wall_s:.1f}s latence_moy={stats.mean_latency_ms:.0f}ms "
              f"replis={stats.fallback_count}", flush=True)
        return seed, captured, steps, confinement, stats, episode_wall_s

    rows = [(s, r["captured"], r["steps"], r["confinement"], EpisodeLLMStats(**r["stats"]),
             r["wall_time_s"]) for s, r in done.items()]
    todo = [seed for seed in seeds if seed not in done]
    if args.concurrency <= 1:
        for seed in todo:
            rows.append(_run_one(seed))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_run_one, seed) for seed in todo]
            for future in as_completed(futures):
                rows.append(future.result())
    # ni l'ordre de complétion ni celui de la reprise ne sont l'ordre des seeds
    rows.sort(key=lambda r: r[0])

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
