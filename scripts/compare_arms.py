"""Comparaisons appariées entre bras (test de McNemar exact, bilatéral).

    python -m scripts.compare_arms --episodes 30 --set max_steps=60 size=15 n_loops=1 \\
        min_loop_len=6 min_spawn_dist=6 \\
        --journal A1v3=results/jalon2_a1v3_local/journal.jsonl \\
        --journal A1bis=results/jalon2_a1bis_local/journal.jsonl \\
        --journal A2=results/jalon2_a2_local/journal.jsonl \\
        --pairs A2:A1bis A2:R2 A2:P2 A1bis:A1v3 R1:R2

Les politiques scriptées (R1, R2, P2) sont rejouées sur les mêmes seeds ; les bras
LLM sont lus dans leurs journaux. Pour chaque paire : captures de chacun, seeds que
seul le premier capture, seeds que seul le second capture, p de McNemar.
"""

from __future__ import annotations

import argparse
import json
import math

from chase.config import ChaseConfig
from scripts.calibrate import _parse_overrides
from scripts.ceiling_a2 import POLICIES, evaluate


def mcnemar_p(b: int, c: int) -> float:
    """p bilatéral exact : sous H0, les b + c seeds discordantes se répartissent à pile ou face."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def captures_from_journal(path: str, seeds) -> dict[int, bool]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # ligne coupée par un kill
            if record["seed"] in seeds:
                out[record["seed"]] = record["captured"]
    missing = sorted(set(seeds) - set(out))
    if missing:
        raise ValueError(f"{path} : seeds absentes {missing}")
    return out


def compare(a: dict[int, bool], b: dict[int, bool]) -> dict:
    only_a = sorted(s for s in a if a[s] and not b[s])
    only_b = sorted(s for s in a if b[s] and not a[s])
    return {"a": sum(a.values()), "b": sum(b.values()), "only_a": only_a, "only_b": only_b,
            "p": mcnemar_p(len(only_a), len(only_b))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], help="surcharges clé=valeur de ChaseConfig")
    parser.add_argument("--journal", action="append", default=[], help="nom=chemin d'un journal LLM")
    parser.add_argument("--pairs", nargs="+", required=True, help="premier:second, ex. A2:A1bis")
    args = parser.parse_args()

    cfg = ChaseConfig().replace(**_parse_overrides(args.set))
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    arms = {name: captures_from_journal(path, seeds)
            for name, path in (j.split("=", 1) for j in args.journal)}
    for name in {n for pair in args.pairs for n in pair.split(":")} - set(arms):
        arms[name] = {s: r.captured for s, r in zip(seeds, evaluate(cfg, name, seeds)["results"])}

    print("| Paire | Captures | Seul le premier | Seul le second | p |")
    print("|---|---|---|---|---|")
    for pair in args.pairs:
        first, second = pair.split(":")
        r = compare(arms[first], arms[second])
        print(f"| {first} contre {second} | {r['a']} / {r['b']} "
              f"| {len(r['only_a'])} ({', '.join(map(str, r['only_a'])) or '—'}) "
              f"| {len(r['only_b'])} ({', '.join(map(str, r['only_b'])) or '—'}) "
              f"| {r['p']:.2f} |")


if __name__ == "__main__":
    main()
