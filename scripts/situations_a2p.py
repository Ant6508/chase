"""Test d'A2' sur des situations tirées de la campagne A2 (results/jalon2_a2.md, § 6).

    python -m scripts.situations_a2p select --out results/jalon2_a2p_test/situations.jsonl
    python -m scripts.situations_a2p play results/jalon2_a2p_test/situations.jsonl \\
        --base-url http://127.0.0.1:1234/v1 --timeout 900 --max-tokens 5600 --concurrency 8 \\
        --out results/jalon2_a2p_test/replay.jsonl
    python -m scripts.situations_a2p report results/jalon2_a2p_test/situations.jsonl \\
        results/jalon2_a2p_test/replay.jsonl

Les situations sont des décisions d'A2 sans repli, cible cachée au récepteur, dont on
garde le prompt utilisateur exact (perception et message reçu). Deux familles :
- vide_morte : le coéquipier ne voyait pas la cible, et l'issue la plus probable pour le
  récepteur seul est « morte » : elle mène à des candidates de sa croyance, mais à aucune
  une fois celle-ci intersectée avec le message reçu, propagé d'un pas (comme P2). A2 l'a
  suivie 35 fois sur 44. On les prend toutes ;
- cible_vue : le coéquipier voyait la cible au pas précédent, le récepteur ni alors ni
  maintenant. Témoin : A2' ne doit pas y perdre ce qu'A2 fait déjà (rejoindre la cible).
Chaque situation est jouée deux fois, prompt système d'A2 puis d'A2', en entrelaçant les
appels pour que les deux bras partagent la même charge du serveur.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from chase.belief import diffuse, observe, observe_prob, propagate
from chase.config import ChaseConfig
from chase.env import ChaseEnv, episode_rngs
from chase.llm.client import LMStudioClient
from chase.llm.config import LLMConfig
from chase.llm.message import A2_SPEC, count_tokens
from chase.llm.places import Places
from chase.moves import Move
from chase.runner import _percepts
from chase.target import ScriptedTarget
from scripts.calibrate import _parse_overrides
from scripts.compare_arms import mcnemar_p

ARMS_PLAYED = ("A2", "A2p")
CAMPAIGN = {"max_steps": 60, "size": 15, "n_loops": 1, "min_loop_len": 6, "min_spawn_dist": 6}


def _cells_via(g, here: int, mask: np.ndarray) -> dict[int, np.ndarray]:
    """Voisin -> cases de `mask` dont un plus court chemin depuis `here` passe par lui."""
    cand = g.index[mask]
    return {n: cand[g.dist[n, cand] == g.dist[here, cand] - 1] for n in g.neighbors[here]}


def exit_moves(g, here: int, belief: np.ndarray, prob: np.ndarray, support: np.ndarray) -> dict:
    """Issues mortes (des candidates propres, aucune après intersection avec `support`
    propagé d'un pas), vivantes (des candidates après intersection), et la plus probable
    pour le récepteur seul."""
    own = _cells_via(g, here, belief)
    live = _cells_via(g, here, belief & propagate(support, g.free))
    name = {n: g.move_between(here, n).name for n in own}
    mass = {n: float(prob[g.xy[c, 0], g.xy[c, 1]].sum()) for n, c in own.items()}
    return {"dead": [name[n] for n in own if len(own[n]) and not len(live[n])],
            "alive": [name[n] for n in own if len(live[n])],
            "best": name[max(mass, key=mass.get)]}


def approach_moves(g, here: int, cell: tuple[int, int]) -> list[str]:
    """Coups qui rapprochent (en chemin) de `cell`."""
    c = int(g.index[cell])
    return [g.move_between(here, n).name for n in g.neighbors[here] if g.dist[n, c] < g.dist[here, c]]


def _support(g, places: Places, message: dict) -> np.ndarray:
    out = np.zeros(g.free.shape, dtype=bool)
    if message["cible"] is not None and message["cible"] in places.cell_by_name:
        cells = [places.cell_by_name[message["cible"]]]
    else:
        cells = [c for name in message["candidates"] if name in places.by_name
                 for c in places.by_name[name].cells]
    for c in cells:
        out[g.cells[c]] = True
    return out


def _episode_situations(path: str, cfg: ChaseConfig) -> list[dict]:
    """Rejoue un épisode d'A2 avec ses coups enregistrés (positions vérifiées contre la
    trace) et renvoie ses situations des deux familles."""
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by = {(r["step"], r["pursuer"]): r for r in recs}
    seed = recs[0]["seed"]
    env = ChaseEnv(cfg)
    env.reset(seed=seed)
    g = env.graph
    places = Places.from_graph(g)
    target = ScriptedTarget(g, episode_rngs(seed)["target"], cfg.target_memory, cfg.target_cycle_bias)
    free = g.free
    bel, prob, prev_seen = [None, None], [None, None], None
    out = []
    vis = env.visibility()
    step = 0
    while not env.done:
        per = _percepts(env, vis)
        for i, p in enumerate(per):
            first = bel[i] is None
            bel[i] = observe(free if first else propagate(bel[i], free), p.visible, p.target_seen)
            prob[i] = observe_prob(free / free.sum() if first else diffuse(prob[i], free),
                                   p.visible, p.target_seen)
        rs = [by[(step, i)] for i in range(2)]
        if any(tuple(rs[i]["pos"]) != per[i].pos for i in range(2)) or tuple(rs[0]["target"]) != env.target_pos:
            raise RuntimeError(f"seed {seed}, pas {step} : le rejeu s'écarte de la trace")
        for i in range(2):
            r = rs[i]
            if prev_seen is None or r["fallback"] or not r["message_in"] or per[i].target_seen:
                continue
            here = int(g.index[per[i].pos])
            base = {"seed": seed, "step": step, "pursuer": i, "recorded": r["move"],
                    "user_prompt": r["user_prompt"], "message_in": r["message_in"]}
            mate_saw = prev_seen[1 - i]
            if mate_saw is None:
                ex = exit_moves(g, here, bel[i], prob[i], _support(g, places, json.loads(r["message_in"])))
                if ex["dead"] and ex["alive"] and ex["best"] in ex["dead"]:
                    out.append({**base, "family": "vide_morte", **ex})
            elif prev_seen[i] is None:
                out.append({**base, "family": "cible_vue", "target_seen_at": list(mate_saw),
                            "approach": approach_moves(g, here, mate_saw)})
        prev_seen = [p.target_seen for p in per]
        moves = [Move[rs[0]["move"]], Move[rs[1]["move"]]]
        moves.append(target.act(env.target_pos, vis[env.target_id], [env.pursuer_pos(p) for p in range(2)]))
        vis = env.step_moves(moves)
        step += 1
    return out


def select(trace_dir: str, cfg: ChaseConfig, n_control: int, seed: int = 0) -> list[dict]:
    paths = sorted(glob.glob(f"{trace_dir}/seed_*.jsonl"), key=lambda p: int(re.search(r"seed_(\d+)", p).group(1)))
    found = [s for p in paths for s in _episode_situations(p, cfg)]
    dead = [s for s in found if s["family"] == "vide_morte"]
    seen = [s for s in found if s["family"] == "cible_vue"]
    chosen = dead + random.Random(seed).sample(seen, min(n_control, len(seen)))
    return [{"id": k, **s} for k, s in enumerate(chosen)]


def play(client, situations: list[dict], concurrency: int = 8, count=count_tokens,
         sink=None) -> list[dict]:
    """Chaque situation avec le prompt système d'A2 puis d'A2', appels entrelacés. `sink`
    reçoit chaque ligne dès qu'elle est prête, dans l'ordre (écriture au fil de l'eau)."""
    from chase.llm.prompts import system_prompt

    jobs = [(s, arm) for s in situations for arm in ARMS_PLAYED]

    def one(job):
        s, arm = job
        res = client.decide(system_prompt(arm), s["user_prompt"], message_spec=A2_SPEC)
        return {"id": s["id"], "arm": arm, "move": res.move.name, "fallback": res.fallback,
                "retries": res.retries, "finish_reason": res.finish_reason,
                "cut_attempts": sum("length" in e for e in res.attempt_errors),
                "attempt_errors": list(res.attempt_errors),
                "completion_tokens": res.completion_tokens, "thinking_tokens": count(res.thinking),
                "reasoning": res.reasoning, "thinking": res.thinking,
                "message": res.message}

    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for row in pool.map(one, jobs):
            rows.append(row)
            if sink:
                sink(row)
    return rows


def summarize(situations: list[dict], rows: list[dict]) -> dict:
    sit = {s["id"]: s for s in situations}
    by = {(r["id"], r["arm"]): r for r in rows}
    out = {}
    for family, key in (("vide_morte", "to_dead"), ("cible_vue", "approach")):
        ids = [i for i, s in sit.items() if s["family"] == family]

        def hit(r):
            s = sit[r["id"]]
            return r["move"] in (s["dead"] if family == "vide_morte" else s["approach"])

        fam = {}
        for arm in ARMS_PLAYED:
            rs = [by[(i, arm)] for i in ids if (i, arm) in by]
            ok = [r for r in rs if not r["fallback"]]
            fam[arm] = {key: (sum(hit(r) for r in ok), len(ok)), "fallbacks": len(rs) - len(ok),
                        "cut_attempts": sum(r["cut_attempts"] for r in rs),
                        "thinking_tokens": float(np.mean([r["thinking_tokens"] for r in ok])) if ok else float("nan")}
            if family == "vide_morte":
                fam[arm]["follows_best"] = (sum(r["move"] == sit[r["id"]]["best"] for r in ok), len(ok))
        pairs = Counter()
        for i in ids:
            a, b = by.get((i, "A2")), by.get((i, "A2p"))
            if a and b and not a["fallback"] and not b["fallback"]:
                pairs[{(True, False): "only_A2", (False, True): "only_A2p",
                       (True, True): "both", (False, False): "neither"}[(hit(a), hit(b))]] += 1
        fam["paired"] = {k: pairs[k] for k in ("only_A2", "only_A2p", "both", "neither")}
        out[family] = fam
    return out


def _rate(pair: tuple[int, int]) -> str:
    k, n = pair
    return f"{k / n:.0%} ({k}/{n})" if n else "—"


def report(situations: list[dict], rows: list[dict]) -> str:
    s = summarize(situations, rows)
    sit = {x["id"]: x for x in situations}
    same = [r for r in rows if r["arm"] == "A2" and not r["fallback"]]
    lines = [f"Situations : {sum(x['family'] == 'vide_morte' for x in situations)} vide_morte, "
             f"{sum(x['family'] == 'cible_vue' for x in situations)} cible_vue. "
             f"Rejeu d'A2 : même coup que la campagne dans {_rate((sum(r['move'] == sit[r['id']]['recorded'] for r in same), len(same)))}.",
             "", "| Famille | Bras | Mesure | Suit la plus probable (morte) | Replis | Tentatives coupées | Pensée moyenne |",
             "|---|---|---|---|---|---|---|"]
    for family, key, label in (("vide_morte", "to_dead", "vers une issue morte"),
                               ("cible_vue", "approach", "rapproche de la cible vue")):
        for arm in ARMS_PLAYED:
            f = s[family][arm]
            lines.append(f"| {family} | {arm} | {label} : {_rate(f[key])} | "
                         f"{_rate(f['follows_best']) if 'follows_best' in f else '—'} | {f['fallbacks']} | "
                         f"{f['cut_attempts']} | {f['thinking_tokens']:.0f} |")
    lines.append("")
    for family in ("vide_morte", "cible_vue"):
        p = s[family]["paired"]
        lines.append(f"{family}, apparié (sans replis) : seul A2 {p['only_A2']}, seul A2' {p['only_A2p']}, "
                     f"les deux {p['both']}, aucun {p['neither']} ; McNemar p = {mcnemar_p(p['only_A2'], p['only_A2p']):.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--trace", default="results/jalon2_a2/trace")
    sel.add_argument("--controls", type=int, default=16, help="situations cible_vue tirées")
    sel.add_argument("--seed", type=int, default=0)
    sel.add_argument("--set", nargs="*", default=[f"{k}={v}" for k, v in CAMPAIGN.items()])
    sel.add_argument("--out", required=True)
    pl = sub.add_parser("play")
    pl.add_argument("situations")
    pl.add_argument("--base-url")
    pl.add_argument("--timeout", type=float)
    pl.add_argument("--max-tokens", type=int)
    pl.add_argument("--concurrency", type=int, default=8)
    pl.add_argument("--out", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("situations")
    rep.add_argument("replay")
    args = parser.parse_args()

    if args.cmd == "select":
        cfg = ChaseConfig().replace(**_parse_overrides(args.set))
        chosen = select(args.trace, cfg, args.controls, args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            for s in chosen:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(Counter(s["family"] for s in chosen))
    elif args.cmd == "play":
        situations = [json.loads(line) for line in open(args.situations, encoding="utf-8")]
        cfg = LLMConfig()
        if args.base_url:
            cfg = replace(cfg, base_url=args.base_url)
        if args.timeout:
            cfg = replace(cfg, timeout_s=args.timeout)
        if args.max_tokens:
            cfg = replace(cfg, max_tokens=args.max_tokens)
        with open(args.out, "w", encoding="utf-8") as f:
            def sink(r):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"situation {r['id']} {r['arm']} : {r['move']}"
                      f"{' (repli)' if r['fallback'] else ''}", flush=True)
            rows = play(LMStudioClient(cfg), situations, args.concurrency, sink=sink)
        print(report(situations, rows))
        print("FIN", flush=True)
    else:
        situations = [json.loads(line) for line in open(args.situations, encoding="utf-8")]
        rows = [json.loads(line) for line in open(args.replay, encoding="utf-8")]
        print(report(situations, rows))


if __name__ == "__main__":
    main()
