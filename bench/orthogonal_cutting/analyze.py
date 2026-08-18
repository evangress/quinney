"""Read telemetry.npz and report the §5 diagnostics this case can support."""

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_case import CELL, DEPTH_OF_CUT, TOOL_Y0, WORKPIECE_Y1


def main():
    data = np.load(HERE / "telemetry.npz")
    coords, eps, temperature = data["final_coords"], data["final_eps_p"], data["final_temperature"]
    force, time_step_s = data["force"], float(data["time_step_s"])

    print(f"points {len(coords)}")
    print(
        f"eps_p  median {np.median(eps):.2f}  p95 {np.percentile(eps, 95):.2f}  max {eps.max():.2f}"
    )
    print(f"       above 5: {(eps > 5).sum()}   above 15: {(eps > 15).sum()}")

    lifted = coords[:, 1] > WORKPIECE_Y1 + 1e-6
    if lifted.sum() > 5:
        chip_thickness = coords[lifted, 1].max() - TOOL_Y0
        print(f"\nchip points {lifted.sum()}   thickness {chip_thickness * 1e6:.0f} um")
        print(f"chip thickness ratio r = t0/tc = {DEPTH_OF_CUT / chip_thickness:.3f}")

    # Band orientation: principal axis of the high-plastic-strain cloud.
    hot = eps > np.percentile(eps, 90)
    centred = coords[hot] - coords[hot].mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centred.T))
    major = eigenvectors[:, np.argmax(eigenvalues)]
    orientation = np.degrees(np.arctan2(abs(major[1]), abs(major[0])))
    elongation = np.sqrt(eigenvalues.max() / max(eigenvalues.min(), 1e-30))
    height_cells = (coords[hot, 1].max() - coords[hot, 1].min()) / CELL
    print(
        f"\nshear zone: {orientation:.1f} deg, elongation {elongation:.1f}:1, {height_cells:.1f} cells tall"
    )
    print("  (spec §5: a band under ~3 cells wide is mesh-limited, not physics)")

    print(
        f"\nT median {np.median(temperature):.0f}K  p95 {np.percentile(temperature, 95):.0f}K"
        f"  max {temperature.max():.1f}K  at melt: {(temperature > 1350).sum()}"
    )

    steady = force[len(force) // 2 :]
    cutting, thrust = steady[:, 0].mean(), steady[:, 1].mean()
    print(
        f"\nFc {cutting / 1e3:.1f} kN/m   Ft {thrust / 1e3:.1f} kN/m   ratio {cutting / thrust:.2f}"
    )
    print(f"specific cutting energy {cutting / DEPTH_OF_CUT / 1e9:.2f} GPa")

    signal = steady[:, 0] - steady[:, 0].mean()
    frequencies = np.fft.rfftfreq(len(signal), time_step_s)
    amplitude = np.abs(np.fft.rfft(signal))
    print(
        f"force oscillation {steady[:, 0].std() / abs(cutting) * 100:.1f}%"
        f"   dominant {frequencies[1:][np.argmax(amplitude[1:])] / 1e3:.0f} kHz"
    )


if __name__ == "__main__":
    main()
