"""Small contained orthogonal cutting case — OFHC copper, MPM, Johnson-Cook.

Deliberately small: a 1.0 x 0.3 mm workpiece at 50 um resolution, cut by a
zero-rake tool at 20 m/s. That is coarse — only two elements through the uncut
chip thickness — so it can show *whether* a chip and a shear zone form, but it
cannot resolve a shear band well enough to measure one (spec §5 calls a band
under ~3 elements wide mesh-limited). Resolution comes after the mechanism is
shown to work.

MPM needs no contact algorithm: the tool and workpiece interact through the
shared background grid. That is the property that made MPM attractive in
spec §3, and here it is what makes a cutting case tractable at all.

    uv run python bench/orthogonal_cutting/build_case.py
"""

import json
import pathlib

from quinney.materials.ofhc_copper import kratos_material_variables

HERE = pathlib.Path(__file__).parent

# --- geometry (metres) ---
CELL = 50e-6

WORKPIECE_X0, WORKPIECE_X1 = 0.0, 1.0e-3
WORKPIECE_Y0, WORKPIECE_Y1 = 0.0, 0.30e-3

DEPTH_OF_CUT = 100e-6  # uncut chip thickness, t0 — two cells
TOOL_GAP = 50e-6  # standoff so the tool starts clear of the workpiece
TOOL_WIDTH, TOOL_HEIGHT = 0.20e-3, 0.20e-3

TOOL_X0 = WORKPIECE_X1 + TOOL_GAP
TOOL_X1 = TOOL_X0 + TOOL_WIDTH
TOOL_Y0 = WORKPIECE_Y1 - DEPTH_OF_CUT  # rake face is vertical: zero rake angle
TOOL_Y1 = TOOL_Y0 + TOOL_HEIGHT

GRID_X0, GRID_X1 = -0.10e-3, TOOL_X1 + 0.10e-3
GRID_Y0, GRID_Y1 = -0.10e-3, TOOL_Y1 + 0.15e-3

# --- cutting conditions ---
CUTTING_SPEED_M_S = 20.0  # 1200 m/min; high, but it keeps the run to minutes
TOOL_DENSITY_KG_M3 = 50000.0  # heavy: the tool is driven, not free to recoil
TOOL_YIELD_PA = 1.0e12  # never reached, so the tool stays elastic

# Elastic wave speed in copper is ~3614 m/s, so a 50 um cell allows about
# 1.4e-8 s. Run at 60% of that.
TIME_STEP_S = 8.0e-9
TRAVEL_M = TOOL_GAP + 0.45e-3  # approach, then 0.45 mm of cut
END_TIME_S = TRAVEL_M / CUTTING_SPEED_M_S


def structured_triangles(x0, x1, y0, y1, cell, first_node_id=1, first_element_id=1):
    """Nodes and triangles for a structured rectangle, with explicit id offsets."""
    nx = round((x1 - x0) / cell)
    ny = round((y1 - y0) / cell)
    grid = {}
    node_id = first_node_id - 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            node_id += 1
            grid[(i, j)] = (node_id, x0 + i * cell, y0 + j * cell)
    triangles = []
    element_id = first_element_id - 1
    for j in range(ny):
        for i in range(nx):
            a, b = grid[(i, j)][0], grid[(i + 1, j)][0]
            c, d = grid[(i + 1, j + 1)][0], grid[(i, j + 1)][0]
            element_id += 1
            triangles.append((element_id, a, b, c))
            element_id += 1
            triangles.append((element_id, a, c, d))
    return list(grid.values()), triangles


def write_mdpa(path, properties, nodes, element_name, element_blocks, sub_model_parts):
    lines = ["Begin ModelPartData", "End ModelPartData", ""]
    for properties_id in properties:
        lines += [f"Begin Properties {properties_id}", "End Properties", ""]

    lines.append("Begin Nodes")
    lines += [f"{node_id} {x:.9f} {y:.9f} 0.0" for node_id, x, y in nodes]
    lines += ["End Nodes", ""]

    for properties_id, triangles in element_blocks:
        lines.append(f"Begin Elements {element_name}")
        lines += [f"{eid} {properties_id} {a} {b} {c}" for eid, a, b, c in triangles]
        lines += ["End Elements", ""]

    for name, (node_ids, element_ids) in sub_model_parts.items():
        lines.append(f"Begin SubModelPart {name}")
        lines.append("  Begin SubModelPartNodes")
        lines += [f"  {n}" for n in node_ids]
        lines.append("  End SubModelPartNodes")
        lines.append("  Begin SubModelPartElements")
        lines += [f"  {e}" for e in element_ids]
        lines.append("  End SubModelPartElements")
        lines.append("End SubModelPart")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    # --- background grid ---
    grid_nodes, grid_triangles = structured_triangles(GRID_X0, GRID_X1, GRID_Y0, GRID_Y1, CELL)
    # The workpiece is held from below and from its left face; without the left
    # face it would simply be pushed along ahead of the tool instead of cut.
    clamped = [
        n
        for n, x, y in grid_nodes
        if y <= WORKPIECE_Y0 + 0.5 * CELL or x <= WORKPIECE_X0 - 0.5 * CELL
    ]
    write_mdpa(
        HERE / "grid.mdpa",
        [0],
        grid_nodes,
        "Element2D3N",
        [(0, grid_triangles)],
        {"fixed_support": (clamped, [])},
    )

    # --- bodies: workpiece (properties 1) and tool (properties 2) ---
    work_nodes, work_triangles = structured_triangles(
        WORKPIECE_X0, WORKPIECE_X1, WORKPIECE_Y0, WORKPIECE_Y1, CELL
    )
    tool_nodes, tool_triangles = structured_triangles(
        TOOL_X0,
        TOOL_X1,
        TOOL_Y0,
        TOOL_Y1,
        CELL,
        first_node_id=len(work_nodes) + 1,
        first_element_id=len(work_triangles) + 1,
    )
    write_mdpa(
        HERE / "body.mdpa",
        [1, 2],
        work_nodes + tool_nodes,
        "MPMUpdatedLagrangian2D3N",
        [(1, work_triangles), (2, tool_triangles)],
        {
            "Parts_Workpiece": (
                [n for n, _x, _y in work_nodes],
                [e for e, *_ in work_triangles],
            ),
            "Parts_Tool": (
                [n for n, _x, _y in tool_nodes],
                [e for e, *_ in tool_triangles],
            ),
        },
    )

    # --- materials: one law, two properties blocks ---
    # The tool reuses the Johnson-Cook law with a yield stress it will never
    # reach, which avoids introducing a second constitutive law.
    law = {"name": "JohnsonCookThermalPlastic2DPlaneStrainLaw"}
    materials = {
        "properties": [
            {
                "model_part_name": "Initial_MPM_Material.Parts_Workpiece",
                "properties_id": 1,
                "Material": {
                    "constitutive_law": law,
                    "Variables": kratos_material_variables(),
                    "Tables": {},
                },
            },
            {
                "model_part_name": "Initial_MPM_Material.Parts_Tool",
                "properties_id": 2,
                "Material": {
                    "constitutive_law": law,
                    "Variables": kratos_material_variables(
                        yield_stress_pa=TOOL_YIELD_PA,
                        density_kg_m3=TOOL_DENSITY_KG_M3,
                    ),
                    "Tables": {},
                },
            },
        ]
    }
    (HERE / "materials.json").write_text(json.dumps(materials, indent=2), encoding="utf-8")

    parameters = {
        "problem_data": {
            "problem_name": "orthogonal_cutting",
            "parallel_type": "OpenMP",
            "echo_level": 0,
            "start_time": 0.0,
            "end_time": END_TIME_S,
        },
        "solver_settings": {
            "solver_type": "Dynamic",
            "model_part_name": "MPM_Material",
            "domain_size": 2,
            "echo_level": 0,
            "time_integration_method": "explicit",
            "analysis_type": "linear",
            "scheme_type": "central_difference",
            "stress_update": "usl",
            "model_import_settings": {"input_type": "mdpa", "input_filename": str(HERE / "body")},
            "grid_model_import_settings": {
                "input_type": "mdpa",
                "input_filename": str(HERE / "grid"),
            },
            "material_import_settings": {"materials_filename": str(HERE / "materials.json")},
            "time_stepping": {"time_step": TIME_STEP_S},
            "problem_domain_sub_model_part_list": ["Parts_Workpiece", "Parts_Tool"],
            "processes_sub_model_part_list": ["fixed_support"],
        },
        "processes": {
            "constraints_process_list": [
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "Parameters": {
                        "model_part_name": "Background_Grid.fixed_support",
                        "variable_name": "DISPLACEMENT",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0],
                    },
                }
            ],
            "loads_process_list": [],
            "list_other_processes": [
                {
                    "python_module": "assign_initial_velocity_to_material_point_process",
                    "kratos_module": "KratosMultiphysics.MPMApplication",
                    "Parameters": {
                        "model_part_name": "Initial_MPM_Material.Parts_Tool",
                        "modulus": CUTTING_SPEED_M_S,
                        "direction": [-1.0, 0.0, 0.0],
                    },
                }
            ],
        },
        "output_processes": {},
    }
    (HERE / "ProjectParameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")

    print(f"grid       {len(grid_nodes):5d} nodes  {len(grid_triangles):5d} elements")
    print(f"workpiece  {len(work_nodes):5d} nodes  {len(work_triangles):5d} elements")
    print(f"tool       {len(tool_nodes):5d} nodes  {len(tool_triangles):5d} elements")
    print(f"clamped    {len(clamped):5d} grid nodes")
    print(f"steps      {round(END_TIME_S / TIME_STEP_S)}  over {END_TIME_S * 1e6:.1f} us")
    print(f"depth of cut {DEPTH_OF_CUT * 1e6:.0f} um = {DEPTH_OF_CUT / CELL:.0f} cells")


if __name__ == "__main__":
    main()
