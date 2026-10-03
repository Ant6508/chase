"""Échos de `cible` dans A2p : suites de pas où la cible circule entre les deux sans être vue.

Depuis la racine du dépôt (rapport : results/jalon2_a2p.md, § 5.2) :

    PYTHONPATH=. python results/jalon2_a2p/echo_runs.py
"""
import glob
import importlib.util
import json
import re

spec = importlib.util.spec_from_file_location("mu", "results/jalon2_a2/msg_usage.py")
mu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mu)

from chase.env import ChaseEnv, episode_rngs
from chase.llm.places import Places
from chase.moves import Move
from chase.runner import _percepts
from chase.target import ScriptedTarget

CFG = mu.CFG
runs = []
examples = []
for arm in ("A2p",):
    for path in sorted(glob.glob(f"results/jalon2_{arm.lower()}/trace/seed_*.jsonl"),
                       key=lambda p: int(re.search(r"seed_(\d+)", p).group(1))):
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        by = {(r["step"], r["pursuer"]): r for r in recs}
        seed = recs[0]["seed"]
        env = ChaseEnv(CFG)
        env.reset(seed=seed)
        rngs = episode_rngs(seed)
        g = env.graph
        places = Places.from_graph(g)
        target = ScriptedTarget(g, rngs["target"], CFG.target_memory, CFG.target_cycle_bias)
        vis = env.visibility()
        step = 0
        cur = None  # (début, cellule relayée)
        while not env.done:
            per = _percepts(env, vis)
            seen = [p.target_seen for p in per]
            rs = [by[(step, i)] for i in range(2)]
            relayed = None
            if seen[0] is None and seen[1] is None:
                for i in range(2):
                    r = rs[i]
                    if r["message_out"] and r["message_in"]:
                        m, rin = json.loads(r["message_out"]), json.loads(r["message_in"])
                        if m["cible"] is not None and m["cible"] == rin.get("cible"):
                            relayed = m["cible"]
                            # le récepteur de ce relais au pas suivant : va-t-il vers la case relayée ?
                            nxt_r = by.get((step + 1, 1 - i))
                            if nxt_r is not None:
                                examples.append((seed, step + 1, 1 - i, relayed, nxt_r))
            if relayed is not None:
                if cur is None:
                    cur = [step, step, relayed, int(g.index[env.target_pos])]
                cur[1] = step
            elif cur is not None:
                runs.append((seed, cur[0], cur[1] - cur[0] + 1, cur[2]))
                cur = None
            moves = [Move[rs[0]["move"]], Move[rs[1]["move"]]]
            pursuers = [env.pursuer_pos(p) for p in range(2)]
            moves.append(target.act(env.target_pos, vis[env.target_id], pursuers))
            vis = env.step_moves(moves)
            step += 1
        if cur is not None:
            runs.append((seed, cur[0], cur[1] - cur[0] + 1, cur[2]))

lens = sorted((r[2] for r in runs), reverse=True)
print("suites d'écho (aucun ne voit la cible, une cible relayée circule) :", len(runs), "suites,",
      sum(lens), "pas ; longueurs :", lens[:15])
print("plus longues :", sorted(runs, key=lambda r: -r[2])[:8])
# exemples : pensée du récepteur d'un relais en écho
for seed, step, i, cell, r in examples:
    if seed in (12, 20, 23) and r["thinking"] and cell in (r["message_in"] or ""):
        print("-----", seed, step, i, cell, r["move"])
        print("IN :", r["message_in"])
        print("OUT:", r["message_out"])
        print("REASONING:", r["reasoning"])
        print("THINK:", r["thinking"][:1500].replace("\n", " "))
        break
