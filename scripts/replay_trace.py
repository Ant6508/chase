"""Rejoue des décisions enregistrées (traces de run_llm.py --trace) contre le serveur actuel.

    python -m scripts.replay_trace results/jalon2_a1v3_local/trace \\
        --groups avant=4-16,18 après=17,19-29 -n 12 --base-url http://127.0.0.1:1234/v1 \\
        --timeout 400 --out results/jalon2_a1v3_local/rejeu.jsonl

Pour chaque groupe de seeds, tire n décisions (graine fixe) parmi celles qui ont
au moins deux issues praticables et ne sont pas des replis, renvoie au modèle la
perception exacte de la trace avec le prompt système actuel, et compare le coup
obtenu au coup enregistré, ainsi que la pensée (`thinking`) mot pour mot. Les
groupes sont entrelacés pour que leurs appels partagent les mêmes lots côté serveur.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from chase.llm.client import LMStudioClient
from chase.llm.config import LLMConfig
from chase.llm.logging import REASONING_LOG_CHARS
from chase.llm.prompts import ARMS, system_prompt
from scripts.analyze_trace import parse_exits


def parse_seeds(spec: str) -> list[int]:
    """« 4-16,18 » -> [4, 5, ..., 16, 18]."""
    seeds = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        seeds += range(int(lo), int(hi or lo) + 1)
    return seeds


def candidates(trace_dir: str, seeds: list[int]) -> list[dict]:
    out = []
    for seed in seeds:
        with open(f"{trace_dir}/seed_{seed}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r["fallback"] and len(parse_exits(r["perception"])) >= 2:
                    out.append(r)
    return out


def draw(trace_dir: str, groups: dict[str, list[int]], n: int, seed: int = 0) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    drawn = {name: rng.sample(candidates(trace_dir, seeds), n) for name, seeds in groups.items()}
    # entrelacés : a0, b0, a1, b1, ...
    return [(name, drawn[name][k]) for k in range(n) for name in groups]


def replay(client, name: str, rec: dict, arm: str = "A1") -> dict:
    # user_prompt (A1bis, A2) contient aussi le message reçu ; absent des traces d'A1v3
    user = rec.get("user_prompt") or rec["perception"]
    if arm == "A2":
        result = client.decide(system_prompt(arm), user, with_message=True)
    else:
        result = client.decide(system_prompt(arm), user)
    return {"group": name, "seed": rec["seed"], "step": rec["step"], "pursuer": rec["pursuer"],
            "recorded": rec["move"], "replayed": result.move.name,
            "same_move": result.move.name == rec["move"],
            "same_thinking": result.thinking == rec["thinking"],
            "same_reasoning": result.reasoning[:REASONING_LOG_CHARS] == rec["reasoning"],
            "fallback": result.fallback,
            "recorded_tokens": rec["completion_tokens"], "replayed_tokens": result.completion_tokens,
            "reasoning": result.reasoning[:REASONING_LOG_CHARS]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--groups", nargs="+", required=True, help="nom=seeds, ex. avant=4-16,18")
    parser.add_argument("-n", type=int, default=12, help="décisions tirées par groupe")
    parser.add_argument("--seed", type=int, default=0, help="graine du tirage")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-tokens", type=int,
                        help="budget de complétion par appel (défaut : LLMConfig.max_tokens)")
    parser.add_argument("--arm", choices=ARMS, default="A1",
                        help="bras dont on rejoue le prompt système (défaut : A1)")
    parser.add_argument("--out", required=True, help="JSONL : une ligne par décision rejouée")
    args = parser.parse_args()

    groups = {}
    for g in args.groups:
        name, spec = g.split("=", 1)
        groups[name] = parse_seeds(spec)
    cfg = LLMConfig()
    if args.base_url:
        cfg = replace(cfg, base_url=args.base_url)
    if args.timeout:
        cfg = replace(cfg, timeout_s=args.timeout)
    if args.max_tokens:
        cfg = replace(cfg, max_tokens=args.max_tokens)
    client = LMStudioClient(cfg)

    sample = draw(args.trace_dir, groups, args.n, args.seed)
    rows = []
    with open(args.out, "w", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for row in pool.map(lambda s: replay(client, *s, arm=args.arm), sample):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            print(f"{row['group']} seed {row['seed']} pas {row['step']} P{row['pursuer']} : "
                  f"{row['recorded']} -> {row['replayed']}", flush=True)

    for name in groups:
        g = [r for r in rows if r["group"] == name]
        print(f"{name} : même coup {sum(r['same_move'] for r in g)}/{len(g)}, "
              f"même pensée mot pour mot {sum(r['same_thinking'] for r in g)}/{len(g)}, "
              f"replis {sum(r['fallback'] for r in g)}", flush=True)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
