"""Run the friction case at two coefficients and compare lateral spread."""

import pathlib
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as MPM
import numpy as np
from KratosMultiphysics.MPMApplication.mpm_analysis import MpmAnalysis

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_case import BODY_X0, BODY_X1


class Run(MpmAnalysis):
    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()

    def spread(self):
        part = self.model["MPM_Material"]
        info = part.ProcessInfo
        raw = [e.CalculateOnIntegrationPoints(MPM.MP_COORD, info)[0] for e in part.Elements]
        x = np.array([c[0] for c in raw])
        return x.max() - x.min(), len(x)


for name in ("frictionless", "frictional"):
    params = KM.Parameters((HERE / f"ProjectParameters_{name}.json").read_text(encoding="utf-8"))
    run = Run(KM.Model(), params)
    run.Run()
    width, n = run.spread()
    print(
        f"@@@ {name:13s} points {n}  width {width * 1e3:.4f} mm "
        f"(initial {(BODY_X1 - BODY_X0) * 1e3:.3f} mm)",
        flush=True,
    )
