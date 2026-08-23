"""Run the orthogonal cutting case and dump telemetry.

    uv run python bench/orthogonal_cutting/build_case.py
    uv run python bench/orthogonal_cutting/run.py

Expect several minutes. Results land in telemetry.npz next to this file.
"""

import pathlib

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as MPM
import numpy as np
from build_case import CUTTING_SPEED_M_S, TIME_STEP_S
from KratosMultiphysics.MPMApplication.mpm_analysis import MpmAnalysis

from quinney.kratos_adapter.damage_process import JohnsonCookDamageProcess
from quinney.kratos_adapter.rigid_tool import RigidToolProcess
from quinney.kratos_adapter.softening import DamageSofteningProcess
from quinney.materials.ofhc_copper import (
    FRACTURE_ENERGY_J_M2,
    OFHC_COPPER_DAMAGE,
    YIELD_STRESS_PA,
)

HERE = pathlib.Path(__file__).parent

SAMPLE_EVERY = 25  # steps between full field snapshots
DAMAGE_EVERY = 5  # steps between damage updates; differencing makes this safe


class CuttingRun(MpmAnalysis):
    def __init__(self, model, parameters, soften=True):
        super().__init__(model, parameters)
        self.soften = soften
        self.softening = None
        self.tool = None
        self.damage = None
        self.samples = []
        self._step = 0

    def _attach(self):
        workpiece = self.model["MPM_Material.Parts_Workpiece"]
        tool = self.model["MPM_Material.Parts_Tool"]
        self.tool = RigidToolProcess(tool, (-CUTTING_SPEED_M_S, 0.0))
        self.damage = JohnsonCookDamageProcess(
            workpiece, OFHC_COPPER_DAMAGE, FRACTURE_ENERGY_J_M2, YIELD_STRESS_PA
        )
        if self.soften:
            self.softening = DamageSofteningProcess(workpiece)
        print(f"@@@ workpiece points {workpiece.NumberOfElements()}", flush=True)
        print(f"@@@ tool points {tool.NumberOfElements()}", flush=True)

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        if self.tool is None:
            self._attach()

        self._step += 1
        self.tool.ExecuteFinalizeSolutionStep()
        if self._step % DAMAGE_EVERY == 0:
            self.damage.ExecuteFinalizeSolutionStep()
            if self.softening is not None:
                self.softening.apply(self.damage.damage)
        if self._step % SAMPLE_EVERY == 0:
            self._sample()

    def _sample(self):
        part = self.model["MPM_Material.Parts_Workpiece"]
        info = part.ProcessInfo
        elements = list(part.Elements)
        if not elements:
            return

        def scalar(var):
            return np.array(
                [e.CalculateOnIntegrationPoints(var, info)[0] for e in elements], dtype=float
            )

        raw = [e.CalculateOnIntegrationPoints(MPM.MP_COORD, info)[0] for e in elements]
        coords = np.array([[c[0], c[1]] for c in raw], dtype=float)
        self.samples.append(
            {
                "step": self._step,
                "coords": coords,
                "eps_p": scalar(MPM.MP_EQUIVALENT_PLASTIC_STRAIN),
                "temperature": scalar(MPM.MP_TEMPERATURE),
                "rate": scalar(MPM.MP_EQUIVALENT_PLASTIC_STRAIN_RATE),
            }
        )
        latest = self.samples[-1]
        print(
            f"@@@ step {self._step:5d}  points {len(elements):4d}  "
            f"eps_p max {latest['eps_p'].max():7.3f}  "
            f"T max {latest['temperature'].max():7.1f}K  "
            f"Fc {self.tool.force_history[-1][0]:9.1f}N",
            flush=True,
        )


def main():
    parameters = KM.Parameters((HERE / "ProjectParameters.json").read_text(encoding="utf-8"))
    run = CuttingRun(KM.Model(), parameters)
    run.Run()

    forces = np.array(
        run.force_history if hasattr(run, "force_history") else run.tool.force_history
    )
    last = run.samples[-1]
    np.savez_compressed(
        HERE / "telemetry.npz",
        force=forces,
        time_step_s=TIME_STEP_S,
        final_coords=last["coords"],
        final_eps_p=last["eps_p"],
        final_temperature=last["temperature"],
        final_rate=last["rate"],
        damage=np.array(list(run.damage.damage.values())),
    )

    # Steady state: skip the approach and the initial transient.
    steady = forces[len(forces) // 2 :]
    print("\n@@@ === cutting summary ===")
    print(f"@@@ steps                {len(forces)}")
    print(
        f"@@@ cutting force  Fx    mean {steady[:, 0].mean():9.1f} N  std {steady[:, 0].std():8.1f}"
    )
    print(
        f"@@@ thrust force   Fy    mean {steady[:, 1].mean():9.1f} N  std {steady[:, 1].std():8.1f}"
    )
    print(f"@@@ eps_p max            {last['eps_p'].max():.3f}")
    print(f"@@@ temperature max      {last['temperature'].max():.1f} K")
    print(f"@@@ strain rate max      {last['rate'].max():.3e} /s")
    d = np.array(list(run.damage.damage.values()))
    print(f"@@@ damage max           {d.max():.4f}   points > 0: {int((d > 0).sum())}")

    # Chip evidence: material lifted above the original workpiece top surface.
    from build_case import WORKPIECE_Y1

    lifted = last["coords"][:, 1] > WORKPIECE_Y1 + 1e-6
    print(f"@@@ points above original surface: {int(lifted.sum())} of {len(lifted)}")
    if lifted.any():
        print(
            f"@@@ max height above surface: {(last['coords'][lifted, 1].max() - WORKPIECE_Y1) * 1e6:.1f} um"
        )


if __name__ == "__main__":
    main()
