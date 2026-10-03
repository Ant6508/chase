"""Décomposition des messages faux (A2p contre A2) : relais, écho, omissions.

Depuis la racine du dépôt (rapport : results/jalon2_a2p.md, § 5.2) :

    PYTHONPATH=. python results/jalon2_a2p/msg_truth.py
"""
import glob
import importlib.util
import json
import re
from collections import Counter

spec = importlib.util.spec_from_file_location("mu", "results/jalon2_a2/msg_usage.py")
mu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mu)

from chase.env import ChaseEnv, episode_rngs
from chase.llm.places import Places
from chase.moves import Move
from chase.runner import _percepts
from chase.target import ScriptedTarget

CFG = mu.CFG


def run(arm):
    c = Counter()
    per_seed = {}
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
        prev_seen = [None, None]
        ps = per_seed.setdefault(seed, Counter())
        while not env.done:
            per = _percepts(env, vis)
            rs = [by[(step, i)] for i in range(2)]
            seen = [p.target_seen for p in per]
            for i in range(2):
                r = rs[i]
                if r["message_out"] is None or seen[i] is not None:
                    continue
                m = json.loads(r["message_out"])
                rin = json.loads(r["message_in"]) if r["message_in"] else None
                true = bool(mu.support(g, places, m)[env.target_pos])
                c["n"] += 1
                c["vrai"] += true
                ps["n"] += 1
                ps["faux"] += not true
                if m["cible"] is not None:
                    relay = rin is not None and rin.get("cible") == m["cible"]
                    kind = "relais" if relay else "cible_autre"
                    if relay:
                        # écho : la cible reçue était elle-même un relais (le coéquipier ne la voyait pas)
                        j = 1 - i
                        echo = prev_seen[j] is None
                        c["relais_echo"] += echo
                        c["relais_echo_faux"] += echo and not true
                else:
                    kind = "candidates"
                    listed = set(m["candidates"]) & set(places.by_name)
                    if rin is not None and rin.get("cible") is None:
                        rc = set(rin.get("candidates", {})) & set(places.by_name)
                        c["cand_recu_cands"] += 1
                        c["cand_sous_ens_recu"] += listed <= rc
                        c["cand_sous_ens_recu_faux"] += (listed <= rc) and not true
                    elif rin is not None:
                        c["cand_recu_cible"] += 1
                    else:
                        c["cand_sans_recu"] += 1
                c[f"{kind}_n"] += 1
                c[f"{kind}_faux"] += not true
            moves = [Move[rs[0]["move"]], Move[rs[1]["move"]]]
            pursuers = [env.pursuer_pos(p) for p in range(2)]
            moves.append(target.act(env.target_pos, vis[env.target_id], pursuers))
            vis = env.step_moves(moves)
            prev_seen = seen
            step += 1
    return c, per_seed


for arm in ("A2p", "A2"):
    c, per_seed = run(arm)
    print("=====", arm)
    print(f"messages cible cachée {c['n']}, faux {c['n'] - c['vrai']} ({(c['n'] - c['vrai']) / c['n']:.1%})")
    for k in ("relais", "cible_autre", "candidates"):
        if c[f"{k}_n"]:
            print(f"  {k}: {c[f'{k}_n']} messages, faux {c[f'{k}_faux']} ({c[f'{k}_faux'] / c[f'{k}_n']:.1%})")
    print(f"  relais en écho (le coéquipier ne la voyait pas non plus) : {c['relais_echo']}, dont faux {c['relais_echo_faux']}")
    print(f"  candidates : reçu des candidates {c['cand_recu_cands']}, sous-ensemble du reçu {c['cand_sous_ens_recu']} "
          f"(dont faux {c['cand_sous_ens_recu_faux']}) ; reçu une cible {c['cand_recu_cible']} ; rien reçu {c['cand_sans_recu']}")
    print("  faux par seed :", {s: f"{v['faux']}/{v['n']}" for s, v in sorted(per_seed.items())})
