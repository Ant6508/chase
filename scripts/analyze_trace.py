"""Diagnostic des bras LLM : analyse des traces pas à pas de run_llm.py --trace.

    python -m scripts.analyze_trace results/a1/trace --set max_steps=60 size=15 n_loops=1 \\
        min_loop_len=6 min_spawn_dist=6 [--samples 1]

Par seed : immobilité, coups contre un mur, allers-retours, cases visitées.
Cible visible : part des coups qui rapprochent de la cible en distance de chemin,
et sens nord/sud quand la cible est dans l'axe. Cible cachée : part des coups
vers une issue annoncée avec des cases candidates, vers la plus proche, vers
la plus chargée. Les positions viennent de la trace (vraies positions), la
carte est régénérée depuis la seed, comme dans les épisodes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.graph import MazeGraph
from chase.moves import MOVE_DELTAS, Move

_EXIT = re.compile(r"- (NORD|SUD|EST|OUEST) : praticable ; (?:(\d+) cases? candidates? au plus "
                   r"court par là, la plus proche à (\d+) pas|aucune case candidate)")
_MOVES = {"NORD": "NORTH", "SUD": "SOUTH", "EST": "EAST", "OUEST": "WEST"}


def _exits(perception: str) -> dict[str, tuple[int, int | None]]:
    """Issues praticables annoncées : direction -> (cases candidates, distance de la plus proche)."""
    return {_MOVES[d]: (int(n), int(k)) if n else (0, None) for d, n, k in _EXIT.findall(perception)}


def summarize(records: list[dict], g: MazeGraph) -> Counter:
    c = Counter()
    trails: dict[int, list[tuple[int, int]]] = {}
    for r in records:
        pos, target, move = tuple(r["pos"]), tuple(r["target"]), r["move"]
        trails.setdefault(r["pursuer"], []).append(pos)
        c["décisions"] += 1
        c["replis"] += r["fallback"]
        c["tokens_complétion"] += r["completion_tokens"]
        nxt = pos
        if move == "STAY":
            c["immobile"] += 1
        else:
            dx, dy = MOVE_DELTAS[Move[move]]
            if g.free[pos[0] + dx, pos[1] + dy]:
                nxt = (pos[0] + dx, pos[1] + dy)
            else:
                c["contre_un_mur"] += 1

        if r["perception"].startswith("Cible visible"):
            c["cible_vue"] += 1
            before = g.dist[g.index[pos], g.index[target]]
            after = g.dist[g.index[nxt], g.index[target]]
            c["vue_rapproche"] += after < before
            c["vue_éloigne"] += after > before
            dx, dy = target[0] - pos[0], target[1] - pos[1]
            if dx == 0 and dy != 0:  # cible dans l'axe nord/sud (NORTH = dy négatif)
                c["vue_axe_ns"] += 1
                c["axe_ns_bon_sens"] += move == ("NORTH" if dy < 0 else "SOUTH")
                c["axe_ns_sens_inverse"] += move == ("SOUTH" if dy < 0 else "NORTH")
        else:
            c["cible_cachée"] += 1
            exits = _exits(r["perception"])
            with_cand = {m: v for m, v in exits.items() if v[0] > 0}
            if move in with_cand:
                c["cachée_vers_issue_avec_candidates"] += 1
                c["cachée_vers_plus_proche"] += move == min(with_cand, key=lambda m: with_cand[m][1])
                c["cachée_vers_plus_chargée"] += move == max(with_cand, key=lambda m: with_cand[m][0])
            elif move in exits:
                c["cachée_vers_issue_sans_candidate"] += 1

    for trail in trails.values():
        c["cases_visitées"] += len(set(trail))
        c["allers_retours"] += sum(1 for t in range(1, len(trail) - 1)
                                   if trail[t + 1] == trail[t - 1] != trail[t])
    return c


def _pct(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "—"


def _row(seed, c: Counter) -> str:
    n = c["décisions"]
    return (f"| {seed} | {n} | {_pct(c['immobile'], n)} | {_pct(c['contre_un_mur'], n)} "
            f"| {_pct(c['allers_retours'], n)} | {c['cases_visitées']} "
            f"| {_pct(c['vue_rapproche'], c['cible_vue'])} / {_pct(c['vue_éloigne'], c['cible_vue'])}"
            f" ({c['cible_vue']}) "
            f"| {c['axe_ns_bon_sens']} / {c['axe_ns_sens_inverse']} ({c['vue_axe_ns']}) "
            f"| {_pct(c['cachée_vers_issue_avec_candidates'], c['cible_cachée'])} "
            f"| {_pct(c['cachée_vers_issue_sans_candidate'], c['cible_cachée'])} "
            f"| {_pct(c['cachée_vers_plus_proche'], c['cible_cachée'])} "
            f"| {c['replis']} | {c['tokens_complétion'] / n:.0f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--set", nargs="*", default=[], help="surcharges clé=valeur de ChaseConfig")
    parser.add_argument("--samples", type=int, default=0,
                        help="nombre de décisions montrées en entier par seed (perception et pensée)")
    args = parser.parse_args()
    cfg = ChaseConfig().replace(**{k: json.loads(v) for k, v in (p.split("=", 1) for p in args.set)})

    print("| Seed | Décisions | Immobile | Contre un mur | Allers-retours | Cases visitées "
          "| Cible vue : rapproche / éloigne | Axe N/S : bon sens / inverse "
          "| Cachée : vers des candidates | Cachée : vers une issue vide | Cachée : vers la plus proche "
          "| Replis | Tokens/décision |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    total = Counter()
    shown = []
    paths = glob.glob(os.path.join(args.trace_dir, "seed_*.jsonl"))
    for path in sorted(paths, key=lambda p: int(re.search(r"seed_(\d+)", p).group(1))):
        records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        if not records:
            continue
        env = ChaseEnv(cfg)
        env.reset(seed=records[0]["seed"])
        c = summarize(records, env.graph)
        total += c
        print(_row(records[0]["seed"], c))
        shown += records[:: max(1, len(records) // args.samples)][:args.samples] if args.samples else []
    print(_row("total", total))

    for r in shown:
        print(f"\n### seed {r['seed']}, pas {r['step']}, poursuivant {r['pursuer']} en {tuple(r['pos'])}, "
              f"cible en {tuple(r['target'])} : {r['move']}\n{r['perception']}\n"
              f"[pensée] {r['thinking']}\n[justification] {r['reasoning']}")


if __name__ == "__main__":
    main()
