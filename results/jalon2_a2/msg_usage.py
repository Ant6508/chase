"""Qualité et usage des messages A2, comparés au témoin A1bis, par rejeu des traces.

Chaque épisode est rejoué dans l'environnement avec les coups enregistrés (la cible
scriptée retombe sur les mêmes coups, vérifié pas à pas contre la trace). On recalcule
à chaque pas la croyance individuelle de chaque poursuivant (celle de LLMPursuers), puis :

Qualité (A2) : justesse de `moi` et de `cible`, couverture par `candidates` de la vraie
masse de l'émetteur, lieux listés sans candidate, usage de `je_couvre`.

Usage (A2 contre A1bis) :
(a) issues mortes : cible cachée au récepteur ; une issue « morte » a des candidates dans
    sa croyance, mais aucune une fois intersectée avec ce que savait le coéquipier au pas
    précédent ; une issue « vivante » en garde. Dans les situations à au moins une issue
    morte et une vivante, part des coups vers une issue morte. Le savoir du coéquipier est
    pris soit dans le message réellement reçu (A2), soit dans le message mécanique
    (write_message, P2) rédigé depuis sa vraie croyance (A1bis et A2).
(b) le coéquipier voyait la cible au pas précédent, pas le récepteur : part des coups qui
    rapprochent de la case où elle était vue.
(c) exploration (aucun ne voit la cible) : distance de chemin moyenne entre les deux
    poursuivants, et part des pas où ils sont à 3 cases ou moins l'un de l'autre.

(a) est séparé selon ce que savait le coéquipier : « cible » s'il la voyait, « vide » sinon
(seuls ses lieux vus vides écartent alors une issue). Tests : permutation appariée par seed.

Depuis la racine du dépôt (rapport : results/jalon2_a2.md) :

    PYTHONPATH=. python results/jalon2_a2/msg_usage.py

Écrit aussi examples.jsonl (décisions d'A2 citées en exemple) dans le dossier courant.
"""
from __future__ import annotations

import glob
import json
import re
import statistics

from collections import Counter

import numpy as np

from chase.belief import diffuse, observe, observe_prob, propagate
from chase.config import ChaseConfig
from chase.env import ChaseEnv, episode_rngs
from chase.llm.ceiling import ProtocolPursuers, write_message
from chase.llm.places import Places
from chase.moves import Move
from chase.policies import make_policy
from chase.runner import _percepts, run_episode
from chase.target import ScriptedTarget

CFG = ChaseConfig().replace(max_steps=60, size=15, n_loops=1, min_loop_len=6, min_spawn_dist=6)


def support(g, places, message):
    out = np.zeros(g.free.shape, dtype=bool)
    if message.get("cible") is not None and message["cible"] in places.cell_by_name:
        out[g.cells[places.cell_by_name[message["cible"]]]] = True
        return out
    for name in message.get("candidates", {}):
        if name in places.by_name:
            for c in places.by_name[name].cells:
                out[g.cells[c]] = True
    return out


def exits_cells(g, here, mask):
    """Voisin -> cases de `mask` dont un plus court chemin depuis `here` passe par lui."""
    cand = g.index[mask]
    out = {}
    for n in g.neighbors[here]:
        out[n] = cand[g.dist[n, cand] == g.dist[here, cand] - 1]
    return out


def next_cell(g, here, move):
    if move == "STAY":
        return here
    m = Move[move]
    return next((n for n in g.neighbors[here] if g.move_between(here, n) == m), here)


def replay(path, c: Counter, lists: dict, arm: str, per_seed: dict, examples: list):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by = {(r["step"], r["pursuer"]): r for r in recs}
    seed = recs[0]["seed"]
    env = ChaseEnv(CFG)
    env.reset(seed=seed)
    rngs = episode_rngs(seed)
    g = env.graph
    places = Places.from_graph(g)
    target = ScriptedTarget(g, rngs["target"], CFG.target_memory, CFG.target_cycle_bias)
    free = g.free
    uniform = free / free.sum()
    bel, prob = [None, None], [None, None]
    prev = None  # (croyances, probas, cible vue, positions) au pas précédent
    vis = env.visibility()
    step = 0
    while not env.done:
        per = _percepts(env, vis)
        for i, p in enumerate(per):
            first = bel[i] is None
            bel[i] = observe(free if first else propagate(bel[i], free), p.visible, p.target_seen)
            prob[i] = observe_prob(uniform if first else diffuse(prob[i], free), p.visible, p.target_seen)
        rs = [by.get((step, i)) for i in range(2)]
        if None in rs:
            raise RuntimeError(f"seed {seed} : trace incomplète au pas {step}")
        for i in range(2):
            assert tuple(rs[i]["pos"]) == per[i].pos, (seed, step, i)
            assert tuple(rs[i]["target"]) == env.target_pos, (seed, step, "cible")
        here = [int(g.index[p.pos]) for p in per]
        seen = [p.target_seen for p in per]

        # --- qualité des messages émis (A2)
        for i in range(2):
            r = rs[i]
            if arm != "A2":
                continue
            c["décisions"] += 1
            if r["message_out"] is None:
                c["sans_message"] += 1
                continue
            m = json.loads(r["message_out"])
            c["messages"] += 1
            c["moi_exact"] += m["moi"] == places.cell_name[here[i]]
            c["moi_bon_lieu"] += m["moi"].split(".")[0] == places.places[places.place_of[here[i]]].name
            if seen[i] is not None:
                c["cible_vue"] += 1
                tname = places.cell_name[int(g.index[seen[i]])]
                if m["cible"] is None:
                    c["cible_vue_null"] += 1
                elif m["cible"] == tname:
                    c["cible_vue_exacte"] += 1
                elif m["cible"].split(".")[0] == tname.split(".")[0]:
                    c["cible_vue_bon_lieu"] += 1
                else:
                    c["cible_vue_fausse"] += 1
            else:
                c["cible_cachée"] += 1
                c["cible_cachée_inventée"] += m["cible"] is not None
                listed = set(m["candidates"]) & set(places.by_name)
                cover = np.zeros(free.shape, dtype=bool)
                for name in listed:
                    for cc in places.by_name[name].cells:
                        cover[g.cells[cc]] = True
                lists["couverture_masse"].append(float(prob[i][cover].sum()))
                lists["couverture_cases"].append(float((bel[i] & cover).sum() / bel[i].sum()))
                true_places = {pl.name for pl in places.places
                               if bel[i][g.xy[np.array(pl.cells), 0], g.xy[np.array(pl.cells), 1]].any()}
                lists["lieux_vrais"].append(len(true_places))
                lists["lieux_listés"].append(len(listed))
                lists["lieux_listés_vides"].append(len(listed - true_places))
                lists["lieux_omis"].append(len(true_places - listed))
            c["je_couvre"] += m.get("je_couvre") is not None
            lists["intention_len"].append(len(m.get("intention") or []))

        # --- usage (A2 et A1bis), décisions au pas t, savoir du coéquipier au pas t-1
        if prev is not None:
            pbel, pprob, pseen, phere, ppos = prev
            for i in range(2):
                j = 1 - i
                r = rs[i]
                if r["fallback"]:
                    continue
                nxt = next_cell(g, here[i], r["move"])
                if seen[i] is None:
                    sources = {"méca": write_message(g, places, ppos[j], pbel[j], pprob[j],
                                                     pseen[j], [])}
                    if arm == "A2" and r["message_in"]:
                        sources["reçu"] = json.loads(r["message_in"])
                    # « vide » : le coéquipier ne voyait pas la cible, seuls ses lieux vus vides
                    # (absents de `candidates`) permettent d'écarter une issue
                    kind = "cible" if pseen[j] is not None else "vide"
                    for base, msg in sources.items():
                        src = f"{base}_{kind}"
                        inter = bel[i] & propagate(support(g, places, msg), free)
                        own = exits_cells(g, here[i], bel[i])
                        live = exits_cells(g, here[i], inter)
                        dead = [n for n in own if len(own[n]) and not len(live[n])]
                        alive = [n for n in live if len(live[n])]
                        if dead and alive:
                            c[f"a_{src}_situations"] += 1
                            c[f"a_{src}_vers_morte"] += nxt in dead
                            c[f"a_{src}_vers_vivante"] += nxt in alive
                            ps = per_seed.setdefault(seed, Counter())
                            ps[f"a_{src}_n"] += 1
                            ps[f"a_{src}_k"] += nxt in dead
                            # l'issue la plus probable pour le récepteur seul est-elle morte ?
                            mass = {n: float(prob[i][tuple(g.xy[own[n]].T)].sum()) if len(own[n]) else 0.0 for n in own}
                            best = max(mass, key=mass.get)
                            if best in dead:
                                c[f"a_{src}_best_morte"] += 1
                                c[f"a_{src}_best_morte_suivie"] += nxt == best
                                ps[f"abest_{src}_n"] += 1
                                ps[f"abest_{src}_k"] += nxt == best
                                if arm == "A2" and src == "reçu_vide":
                                    examples.append(("a" if nxt != best else "a_suivie", seed, step, i, r))
                    # (b) seul le coéquipier voyait la cible au pas précédent
                    if pseen[j] is not None and pseen[i] is None:
                        tc = int(g.index[pseen[j]])
                        c["b_situations"] += 1
                        c["b_rapproche"] += g.dist[nxt, tc] < g.dist[here[i], tc]
                        ps = per_seed.setdefault(seed, Counter())
                        ps["b_n"] += 1
                        ps["b_k"] += g.dist[nxt, tc] < g.dist[here[i], tc]
                        if arm == "A2":
                            examples.append(("b", seed, step, i, r))
                        if arm == "A2" and r["message_in"]:
                            c["b_message_cible"] += json.loads(r["message_in"])["cible"] is not None
        if seen[0] is None and seen[1] is None:
            d = g.dist[here[0], here[1]]
            lists["c_distance"].append(float(d))
            c["c_pas"] += 1
            c["c_proches"] += d <= 3
            ps = per_seed.setdefault(seed, Counter())
            ps["c_n"] += 1
            ps["c_k"] += d <= 3

        prev = ([b.copy() for b in bel], [q.copy() for q in prob], list(seen), list(here),
                [per[0].pos, per[1].pos])
        moves = [Move[rs[0]["move"]], Move[rs[1]["move"]]]
        pursuers = [env.pursuer_pos(p) for p in range(2)]
        moves.append(target.act(env.target_pos, vis[env.target_id], pursuers))
        vis = env.step_moves(moves)
        step += 1
    c["épisodes"] += 1


def scripted_separation(name):
    lists = []
    for seed in range(30):
        pol = ProtocolPursuers(CFG) if name == "P2" else make_policy(name, CFG)

        def hook(env, policy):
            vis = env.visibility()
            if all(env.target_seen_by(vis[p]) is None for p in range(2)):
                g = env.graph
                lists.append(g.dist[g.index[env.pursuer_pos(0)], g.index[env.pursuer_pos(1)]])
        run_episode(CFG, pol, seed, on_step=hook)
    return lists


def pct(a, b):
    return f"{a / b:.1%} ({a}/{b})" if b else "—"


def perm_test(sa: dict, sb: dict, key: str, draws: int = 100_000, rng_seed: int = 0) -> tuple[float, float, float]:
    """Test de permutation apparié par seed : sous H0, les comptes (k, n) des deux bras sont
    échangeables à l'intérieur d'une seed. Statistique : écart des taux poolés."""
    seeds = sorted(set(sa) | set(sb))
    ka = np.array([sa.get(s, Counter())[f"{key}_k"] for s in seeds], float)
    na = np.array([sa.get(s, Counter())[f"{key}_n"] for s in seeds], float)
    kb = np.array([sb.get(s, Counter())[f"{key}_k"] for s in seeds], float)
    nb = np.array([sb.get(s, Counter())[f"{key}_n"] for s in seeds], float)
    obs = ka.sum() / na.sum() - kb.sum() / nb.sum()
    rng = np.random.default_rng(rng_seed)
    flip = rng.integers(0, 2, size=(draws, len(seeds))).astype(bool)
    KA = np.where(flip, kb, ka).sum(1); NA = np.where(flip, nb, na).sum(1)
    KB = np.where(flip, ka, kb).sum(1); NB = np.where(flip, na, nb).sum(1)
    stat = KA / NA - KB / NB
    return ka.sum() / na.sum(), kb.sum() / nb.sum(), float((np.abs(stat) >= abs(obs) - 1e-12).mean())


def main():
    seeds_of = {}
    all_examples = []
    for arm, d in (("A2", "results/jalon2_a2/trace"), ("A1bis", "results/jalon2_a1bis/trace")):
        c, lists = Counter(), {k: [] for k in ("couverture_masse", "couverture_cases", "lieux_vrais",
                                                "lieux_listés", "lieux_listés_vides", "lieux_omis",
                                                "intention_len", "c_distance")}
        per_seed, examples = {}, []
        for p in sorted(glob.glob(f"{d}/seed_*.jsonl"), key=lambda p: int(re.search(r"seed_(\d+)", p).group(1))):
            replay(p, c, lists, arm, per_seed, examples)
        seeds_of[arm] = per_seed
        all_examples += examples
        print(f"===== {arm} : {c['épisodes']} épisodes rejoués (positions conformes à la trace)")
        if arm == "A2":
            print("messages valides", pct(c["messages"], c["décisions"]))
            print("moi exact", pct(c["moi_exact"], c["messages"]), "; bon lieu", pct(c["moi_bon_lieu"], c["messages"]))
            print("cible vue :", c["cible_vue"], "; exacte", pct(c["cible_vue_exacte"], c["cible_vue"]),
                  "; bon lieu, mauvais rang", pct(c["cible_vue_bon_lieu"], c["cible_vue"]),
                  "; fausse", pct(c["cible_vue_fausse"], c["cible_vue"]), "; null", pct(c["cible_vue_null"], c["cible_vue"]))
            print("cible cachée :", c["cible_cachée"], "; cible inventée", pct(c["cible_cachée_inventée"], c["cible_cachée"]))
            cm = lists["couverture_masse"]
            print(f"couverture de la masse : moyenne {statistics.mean(cm):.1%}, médiane {statistics.median(cm):.1%}, "
                  f"complète (>= 99,5 %) {sum(x >= 0.995 for x in cm) / len(cm):.1%}, < 90 % {sum(x < 0.9 for x in cm) / len(cm):.1%}, "
                  f"< 50 % {sum(x < 0.5 for x in cm) / len(cm):.1%}")
            print(f"couverture des cases : moyenne {statistics.mean(lists['couverture_cases']):.1%}")
            for k in ("lieux_vrais", "lieux_listés", "lieux_listés_vides", "lieux_omis"):
                print(f"{k} : moyenne {statistics.mean(lists[k]):.1f}, médiane {statistics.median(lists[k])}")
            print("je_couvre renseigné", pct(c["je_couvre"], c["messages"]),
                  f"; intention : {statistics.mean(lists['intention_len']):.1f} lieux en moyenne")
        print("(b) seul le coéquipier voyait la cible : rapproche", pct(c["b_rapproche"], c["b_situations"]),
              *(["; message avec cible", pct(c["b_message_cible"], c["b_situations"])] if arm == "A2" else []))
        dist = lists["c_distance"]
        print(f"(c) exploration : {c['c_pas']} pas, distance moyenne {statistics.mean(dist):.1f}, "
              f"médiane {statistics.median(dist)}, à 3 cases ou moins {pct(c['c_proches'], c['c_pas'])}")
        for src in ("reçu_cible", "reçu_vide", "méca_cible", "méca_vide"):
            if c[f"a_{src}_situations"]:
                print(f"(a) [{src}] situations {c[f'a_{src}_situations']} : vers une morte "
                      f"{pct(c[f'a_{src}_vers_morte'], c[f'a_{src}_situations'])} ; plus probable morte "
                      f"{c[f'a_{src}_best_morte']} fois, suivie "
                      f"{pct(c[f'a_{src}_best_morte_suivie'], c[f'a_{src}_best_morte'])}")
    a2, a1 = seeds_of["A2"], seeds_of["A1bis"]
    for key, label in (("a_méca_cible", "(a) cible vue par le coéquipier : vers une issue morte"),
                       ("a_méca_vide", "(a) lieux vus vides seulement : vers une issue morte"),
                       ("abest_méca_cible", "(a) cible vue : suit la plus probable quand elle est morte"),
                       ("abest_méca_vide", "(a) vus vides : suit la plus probable quand elle est morte"),
                       ("b", "(b) rapproche de la cible vue par le coéquipier"), ("c", "(c) à 3 cases ou moins en exploration")):
        ra, rb, pv = perm_test(a2, a1, key)
        print(f"{label} : A2 {ra:.1%} contre A1bis {rb:.1%}, p = {pv:.4f} (permutation appariée par seed)")
    for name in ("R1", "R2", "P2"):
        dist = scripted_separation(name)
        print(f"(c) {name} : {len(dist)} pas, distance moyenne {statistics.mean(dist):.1f}, "
              f"médiane {statistics.median(dist)}, à 3 cases ou moins {sum(x <= 3 for x in dist) / len(dist):.1%}")
    with open("examples.jsonl", "w", encoding="utf-8") as f:
        for kind, seed, step, i, r in all_examples:
            f.write(json.dumps({"kind": kind, "seed": seed, "step": step, "pursuer": i, "move": r["move"],
                                "message_in": r["message_in"], "perception": r["perception"],
                                "thinking": r["thinking"], "reasoning": r["reasoning"]}, ensure_ascii=False))
            f.write(chr(10))
    print(len(all_examples), "exemples écrits")


if __name__ == "__main__":
    main()
