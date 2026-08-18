"""Copper block driven into a clamped grid boundary, with JC damage attached.

Not a validation case — a working vertical slice: MPM + Johnson-Cook thermo-
plasticity + quinney's damage model + separation. Run `build_case.py` first.

    uv run python bench/jc_compression/build_case.py
    uv run python bench/jc_compression/run.py

Takes roughly a minute per threshold on 12 CPU threads. Deliberately kept out
of the unit suite, which must stay fast.
"""

import pathlib

import KratosMultiphysics as KM
import numpy as np
from KratosMultiphysics.MPMApplication.mpm_analysis import MpmAnalysis

from quinney.kratos_adapter.damage_process import JohnsonCookDamageProcess
from quinney.materials.ofhc_copper import OFHC_COPPER_DAMAGE

HERE = pathlib.Path(__file__).parent


class WithDamage(MpmAnalysis):
    def __init__(self, model, params, threshold):
        super().__init__(model, params)
        self._threshold = threshold
        self.damage_process = None
        self.separated_history = []

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        part = self.model["MPM_Material"]
        if self.damage_process is None:
            self.damage_process = JohnsonCookDamageProcess(
                part, OFHC_COPPER_DAMAGE, separation_threshold=self._threshold
            )
        self.damage_process.ExecuteFinalizeSolutionStep()
        d = np.array(list(self.damage_process.damage.values()))
        self.separated_history.append(int((d >= self._threshold).sum()))


def run(threshold):
    params = KM.Parameters((HERE / "ProjectParameters.json").read_text(encoding="utf-8"))
    a = WithDamage(KM.Model(), params, threshold)
    a.Run()
    d = np.array(list(a.damage_process.damage.values()))
    return d, a.separated_history


for threshold in (1.0, 0.05):
    d, hist = run(threshold)
    print(f"\n@@@ === separation_threshold = {threshold} ===")
    print(f"@@@ points tracked: {d.size}")
    print(f"@@@ damage  max {d.max():.6f}  mean {d.mean():.6f}  >0: {int((d > 0).sum())}")
    print(f"@@@ separated at end: {hist[-1]}")
    first = next((i for i, v in enumerate(hist) if v > 0), None)
    print(f"@@@ first separation at step: {first}")
