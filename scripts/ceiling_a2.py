"""Plafonds mécaniques P2 (message A2) et P3 (message A3) contre R1 et R2, sans LLM.

    python -m scripts.ceiling_a2 --episodes 200 --set max_steps=60 size=15 n_loops=1 \\
        min_loop_len=6 min_spawn_dist=6 --out results/jalon2_a2_plafond.md

P2 et P3 jouent comme R2, mais ne savent de leur coéquipier que ce que transporte le
message de leur bras (chase/llm/ceiling.py). Porte d'A2 : P2 doit garder au moins la
moitié de l'écart de captures entre R1 et R2. Sinon, le grain du protocole est à revoir
avant tout appel au LLM (docs/superpowers/specs/2026-09-29-jalon2-a2-design.md,
§ Validation, étape 1). P3 n'a pas de porte : on rapporte la part de l'écart qu'il garde
(docs/superpowers/specs/2026-10-08-jalon2-a3-design.md, § Validation, étape 1).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from chase.config import ChaseConfig
from chase.llm.ceiling import ProtocolPursuers
from chase.llm.message import A3_SPEC
from chase.policies import GreedyPursuers
from chase.runner import run_episode
from scripts.calibrate import _ci95, _parse_overrides

POLICIES = {
    "R1": lambda cfg: GreedyPursuers(cfg, fused=False),
    "R2": lambda cfg: GreedyPursuers(cfg, fused=True),
    "P2": ProtocolPursuers,
    "P3": lambda cfg: ProtocolPursuers(cfg, spec=A3_SPEC),
}


def evaluate(cfg: ChaseConfig, name: str, seeds) -> dict:
    """Épisodes joués à la suite, dans l'ordre des seeds (quelques secondes pour 200)."""
    results = [run_episode(cfg, POLICIES[name](cfg), s) for s in seeds]
    captured = [r for r in results if r.captured]
    return {
        "results": results,
        "capture_rate": len(captured) / len(results),
        "confinement": float(np.mean([r.confinement for r in results])),
        "mean_confinement": float(np.mean([r.mean_confinement for r in results])),
        "steps_to_capture": (float(np.median([r.steps for r in captured]))
                             if captured else float("nan")),
    }


def kept_share(rates: dict[str, float], name: str) -> float:
    """Part de l'écart de captures R1 -> R2 que garde `name` ; nan sans écart positif."""
    gap = rates["R2"] - rates["R1"]
    if gap <= 0:
        return float("nan")
    return (rates[name] - rates["R1"]) / gap


def gate(rates: dict[str, float]) -> tuple[float, bool]:
    """Part de l'écart de captures R1 -> R2 que P2 conserve, et la porte (au moins 50 %).
    Sans écart positif, la porte n'est pas évaluable, donc pas franchie."""
    kept = kept_share(rates, "P2")
    if math.isnan(kept):
        return kept, False
    return kept, kept >= 0.5 - 1e-9  # tolérance à l'arrondi flottant près de la frontière


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], help="surcharges clé=valeur de ChaseConfig")
    parser.add_argument("--out", help="écrit le tableau Markdown dans ce fichier")
    args = parser.parse_args()

    cfg = ChaseConfig().replace(**_parse_overrides(args.set))
    seeds = range(args.first_seed, args.first_seed + args.episodes)
    lines = [
        f"Seeds {seeds.start}..{seeds.stop - 1} ({len(seeds)} épisodes), T = {cfg.max_steps}.",
        "",
        "| Politique | Capture | IC 95 % | Confinement à T | Confinement moyen "
        "| Pas jusqu'à capture (médiane) |",
        "|---|---|---|---|---|---|",
    ]
    rates = {}
    for name in POLICIES:
        t0 = time.time()
        r = evaluate(cfg, name, seeds)
        rates[name] = r["capture_rate"]
        steps = "—" if math.isnan(r["steps_to_capture"]) else f"{r['steps_to_capture']:.0f}"
        lines.append(
            f"| {name} | {r['capture_rate']:.0%} | ± {_ci95(r['capture_rate'], len(seeds)):.0%} "
            f"| {r['confinement']:.3f} | {r['mean_confinement']:.3f} | {steps} |")
        print(f"{name} évalué en {time.time() - t0:.1f} s", flush=True)

    kept, ok = gate(rates)
    verdict = "franchie" if ok else "non franchie"
    kept_txt = "non évaluable" if math.isnan(kept) else f"{kept:.0%}"
    p3 = kept_share(rates, "P3")
    p3_txt = "non évaluable" if math.isnan(p3) else f"{p3:.0%}"
    lines += ["", f"Écart de captures R1 -> R2 : {rates['R2'] - rates['R1']:+.0%}. "
                  f"P2 en garde {kept_txt}. Porte (au moins 50 %) : **{verdict}**.",
              "", f"P3 (message A3 : moi, cible, intention) en garde {p3_txt}, sans porte.",
              "", "Paramètres :", "", "```json", json.dumps(cfg.as_dict(), indent=2), "```"]
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
