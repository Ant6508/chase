"""Jalon 1 : tableau R0/R1/R2 sur des épisodes seedés.

    python -m scripts.calibrate                      # paramètres figés, 100 seeds
    python -m scripts.calibrate --episodes 30 --set view_size=3 n_loops=4
    python -m scripts.calibrate --out results/jalon1.md
"""

from __future__ import annotations

import argparse
import json
import math
import time

from chase.config import ChaseConfig
from chase.runner import evaluate


def _parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        out[key] = json.loads(value)
    return out


def _ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], help="surcharges clé=valeur de ChaseConfig")
    parser.add_argument("--policies", default="R0,R1,R2")
    parser.add_argument("--out", help="écrit le tableau Markdown dans ce fichier")
    args = parser.parse_args()

    cfg = ChaseConfig().replace(**_parse_overrides(args.set))
    seeds = range(args.first_seed, args.first_seed + args.episodes)

    lines = [
        f"Seeds {seeds.start}..{seeds.stop - 1} ({len(seeds)} épisodes), T = {cfg.max_steps}.",
        "",
        "| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen | Pas jusqu'à capture (médiane) |",
        "|---|---|---|---|---|---|",
    ]
    for name in args.policies.split(","):
        t0 = time.time()
        r = evaluate(cfg, name, seeds)
        steps = "—" if math.isnan(r["steps_to_capture"]) else f"{r['steps_to_capture']:.0f}"
        lines.append(
            f"| {name} | {r['capture_rate']:.0%} | ± {_ci95(r['capture_rate'], len(seeds)):.0%} "
            f"| {r['confinement']:.3f} | {r['mean_confinement']:.3f} | {steps} |")
        print(f"{name} évalué en {time.time() - t0:.1f} s", flush=True)

    lines += ["", "Paramètres :", "", "```json", json.dumps(cfg.as_dict(), indent=2), "```"]
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
