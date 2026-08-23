"""Same cut, with and without damage softening. Does it change anything?"""

import pathlib
import sys

import KratosMultiphysics as KM
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from run import CuttingRun

results = {}
for soften in (False, True):
    params = KM.Parameters((HERE / "ProjectParameters.json").read_text(encoding="utf-8"))
    run = CuttingRun(KM.Model(), params, soften=soften)
    run.Run()
    last = run.samples[-1]
    forces = np.array(run.tool.force_history)
    steady = forces[len(forces) // 2 :]
    initiation = np.array(list(run.damage.initiation.values()))
    damage = np.array(list(run.damage.damage.values()))
    label = "softened" if soften else "baseline"
    results[label] = {
        "eps_p_p95": float(np.percentile(last["eps_p"], 95)),
        "eps_p_max": float(last["eps_p"].max()),
        "T_p95": float(np.percentile(last["temperature"], 95)),
        "Fc": float(steady[:, 0].mean()),
        "initiated": int((initiation >= 1.0).sum()),
        "damaged": int((damage > 0).sum()),
        "damage_max": float(damage.max()),
        "points": len(last["eps_p"]),
    }
    print(f"@@@ {label} done", flush=True)

print("\n@@@ === softening comparison ===")
keys = ["points", "initiated", "damaged", "damage_max", "eps_p_p95", "eps_p_max", "T_p95", "Fc"]
print(f"@@@ {'metric':12s} {'baseline':>14s} {'softened':>14s}")
for k in keys:
    b, s = results["baseline"][k], results["softened"][k]
    print(f"@@@ {k:12s} {b:14.4g} {s:14.4g}")
