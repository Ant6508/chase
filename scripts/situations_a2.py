"""Situations témoins du bras A2 contre le modèle local (§ Validation, étape 2).

    python -m scripts.situations_a2 --set max_steps=60 size=15 n_loops=1 min_loop_len=6 \\
        min_spawn_dist=6 --base-url http://127.0.0.1:1234/v1 --timeout 400 --max-tokens 4000 \\
        --out results/jalon2_a2_situations.jsonl

Les situations viennent de parties de P2 (décisions de R2, message A2 exact) sur les
seeds 30 à 129, hors des seeds de campagne. Au plus une par famille et par seed.
Trois familles, choisies pour que le message change le bon coup :
- vue_par_coéquipier : la cible n'est vue que par le coéquipier ;
- issue_vidée : personne ne voit la cible, et le coup de P2 n'est pas l'issue la plus
  probable d'après la seule perception du poursuivant ;
- tenaille : le poursuivant voit la cible, et elle se trouve sur un plus court chemin
  entre lui et son coéquipier.
Chacune est présentée deux fois au modèle : perception A1bis seule, puis A2 avec le
message du coéquipier écrit par P2. On compare les coups à celui de P2, et on contrôle
le message émis en A2 (validité, noms inconnus, part de la masse de l'émetteur couverte).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from chase.config import ChaseConfig
from chase.env import ChaseEnv
from chase.llm.ceiling import ProtocolPursuers
from chase.llm.client import LMStudioClient
from chase.llm.config import LLMConfig
from chase.llm.message import render, unknown_names
from chase.llm.places import Places
from chase.llm.prompts import build_perception, message_block, system_prompt
from chase.moves import Move
from chase.runner import run_episode
from scripts.analyze_trace import parse_exits
from scripts.calibrate import _parse_overrides
from scripts.replay_trace import parse_seeds

FAMILIES = {"vue_par_coéquipier": 4, "issue_vidée": 3, "tenaille": 3}


class _Recorder(ProtocolPursuers):
    """P2 qui garde, pour chaque décision, de quoi reconstruire les prompts A1bis et A2."""

    def reset(self, graph, rng):
        super().reset(graph, rng)
        self.decisions: list[dict] = []
        self._t = 0

    def act(self, percepts):
        before = []
        for i, p in enumerate(percepts):
            perception = build_perception(p.pos, self.g, self._own[i], self._own_p[i],
                                          p.target_seen, self.cfg.track_threshold, self.places)
            before.append((self._inbox[i], perception))
        moves = super().act(percepts)
        for i, p in enumerate(percepts):
            inbox, perception = before[i]
            received = None if inbox is None else render(inbox)
            self.decisions.append({
                "step": self._t, "pursuer": i, "pos": p.pos, "target_seen": p.target_seen,
                "inbox": inbox, "move": moves[i], "own_message": self.sent[i],
                "a1bis_prompt": perception,
                "a2_prompt": f"{perception}\n\n{message_block(received)}",
            })
        self._t += 1
        return moves


def _family(d: dict, g, places: Places) -> str | None:
    m = d["inbox"]
    if m is None:
        return None
    if d["target_seen"] is None and m["cible"] is not None:
        return "vue_par_coéquipier"
    if d["target_seen"] is None:
        exits = {e: v for e, v in parse_exits(d["a1bis_prompt"]).items()
                 if v[0] > 0 and v[2] is not None}
        if exits and d["move"] != Move.STAY:
            best = max(exits, key=lambda e: exits[e][2])
            if best != d["move"].name:
                return "issue_vidée"
        return None
    here, t = g.index[d["pos"]], g.index[d["target_seen"]]
    mate = places.cell_by_name[m["moi"]]
    if t not in (here, mate) and g.dist[here, mate] == g.dist[here, t] + g.dist[t, mate]:
        return "tenaille"
    return None


def collect(cfg: ChaseConfig, seeds, quotas: dict[str, int] = FAMILIES) -> list[dict]:
    found: dict[str, list] = {f: [] for f in quotas}
    for seed in seeds:
        env = ChaseEnv(cfg)
        policy = _Recorder(cfg)
        run_episode(cfg, policy, seed, env=env)
        used: set[str] = set()
        for d in policy.decisions:
            family = _family(d, env.graph, policy.places)
            if family is None or family in used or len(found[family]) >= quotas[family]:
                continue
            used.add(family)
            found[family].append({
                "seed": seed, "step": d["step"], "pursuer": d["pursuer"], "family": family,
                "p2_move": d["move"].name, "a1bis_prompt": d["a1bis_prompt"],
                "a2_prompt": d["a2_prompt"], "own_message": d["own_message"]})
        if all(len(found[f]) >= q for f, q in quotas.items()):
            break
    return [s for f in quotas for s in found[f]]


class _Names:
    """Adaptateur : unknown_names attend un objet qui a un attribut `names`."""

    def __init__(self, names: frozenset[str]):
        self.names = names


def ask(client, s: dict, names: frozenset[str]) -> dict:
    a1bis = client.decide(system_prompt("A1bis"), s["a1bis_prompt"])
    a2 = client.decide(system_prompt("A2"), s["a2_prompt"], with_message=True)
    own = s["own_message"]["candidates"]
    listed = a2.message["candidates"] if a2.message is not None else {}
    # part de la vraie masse de l'émetteur couverte par les lieux qu'il a listés
    coverage = sum(v for k, v in own.items() if k in listed) / max(1, sum(own.values()))
    return {**{k: s[k] for k in ("seed", "step", "pursuer", "family", "p2_move")},
            "a1bis_move": a1bis.move.name, "a2_move": a2.move.name,
            "a1bis_fallback": a1bis.fallback, "a2_fallback": a2.fallback,
            "a2_message_valid": a2.message is not None, "a2_message": a2.message,
            "unknown_names": (unknown_names(a2.message, _Names(names))
                              if a2.message is not None else []),
            "coverage": coverage,
            "a1bis_thinking": a1bis.thinking, "a2_thinking": a2.thinking,
            # diagnostic de chaque appel (le repli y est lisible dans `reasoning`)
            **{f"{arm}_{k}": getattr(r, k) for arm, r in (("a1bis", a1bis), ("a2", a2))
               for k in ("reasoning", "prompt_tokens", "completion_tokens", "finish_reason",
                         "attempt_errors", "latency_ms")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[], help="surcharges clé=valeur de ChaseConfig")
    parser.add_argument("--seeds", default="30-129", help="seeds des parties de P2, ex. 30-129")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-tokens", type=int,
                        help="budget de complétion par appel (défaut : LLMConfig.max_tokens)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", required=True, help="JSONL : une ligne par situation")
    args = parser.parse_args()

    cfg = ChaseConfig().replace(**_parse_overrides(args.set))
    situations = collect(cfg, parse_seeds(args.seeds))
    print(f"{len(situations)} situations : "
          + ", ".join(f"{f} {sum(s['family'] == f for s in situations)}" for f in FAMILIES),
          flush=True)
    names = {}
    for s in situations:
        if s["seed"] not in names:
            env = ChaseEnv(cfg)
            env.reset(seed=s["seed"])
            names[s["seed"]] = Places.from_graph(env.graph).names

    llm_cfg = LLMConfig()
    if args.base_url:
        llm_cfg = replace(llm_cfg, base_url=args.base_url)
    if args.timeout:
        llm_cfg = replace(llm_cfg, timeout_s=args.timeout)
    if args.max_tokens:
        llm_cfg = replace(llm_cfg, max_tokens=args.max_tokens)
    client = LMStudioClient(llm_cfg)

    rows = []
    with open(args.out, "w", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for row in pool.map(lambda s: ask(client, s, names[s["seed"]]), situations):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            print(f"{row['family']} seed {row['seed']} pas {row['step']} P{row['pursuer']} : "
                  f"P2 {row['p2_move']}, A1bis {row['a1bis_move']}, A2 {row['a2_move']}", flush=True)

    for f in FAMILIES:
        g = [r for r in rows if r["family"] == f]
        if not g:
            continue
        print(f"{f} : A2 = P2 {sum(r['a2_move'] == r['p2_move'] for r in g)}/{len(g)}, "
              f"A1bis = P2 {sum(r['a1bis_move'] == r['p2_move'] for r in g)}/{len(g)}, "
              f"messages valides {sum(r['a2_message_valid'] for r in g)}/{len(g)}, "
              f"couverture moyenne {sum(r['coverage'] for r in g) / len(g):.0%}", flush=True)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
