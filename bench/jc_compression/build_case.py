"""Minimal MPM + Johnson-Cook case: a copper block driven into a fixed grid boundary.

Purpose is not physics validation. It is to answer the three open questions in
docs/kratos_api_notes.md with a case small enough to run in seconds:
  1. Is MP_DELTA_PLASTIC_STRAIN per-step or cumulative?
  2. Is MP_PRESSURE positive in compression?
  3. Does gather_material_point_state() work on real material points?
"""

import json
import pathlib

from quinney.materials.ofhc_copper import kratos_material_variables

HERE = pathlib.Path(__file__).parent

# --- geometry (metres) ---
GRID_X, GRID_Y = 0.006, 0.016
CELL = 0.0005
BODY_X0, BODY_X1 = 0.001, 0.005
BODY_Y0, BODY_Y1 = 0.006, 0.014
FIXED_BELOW_Y = BODY_Y0 + 1e-9  # grid nodes at/below the block's base are clamped

IMPACT_SPEED_M_S = 150.0
TIME_STEP_S = 2.0e-8
END_TIME_S = 4.0e-6


def structured_triangles(x0, x1, y0, y1, cell):
    """Nodes (1-based) and triangle connectivity for a structured rectangle."""
    nx = round((x1 - x0) / cell)
    ny = round((y1 - y0) / cell)
    nodes = {}
    node_id = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            node_id += 1
            nodes[(i, j)] = (node_id, x0 + i * cell, y0 + j * cell)
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a = nodes[(i, j)][0]
            b = nodes[(i + 1, j)][0]
            c = nodes[(i + 1, j + 1)][0]
            d = nodes[(i, j + 1)][0]
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return list(nodes.values()), triangles


def write_mdpa(path, nodes, triangles, element_name, properties_id, sub_model_parts):
    lines = ["Begin ModelPartData", "End ModelPartData", ""]
    lines += [f"Begin Properties {properties_id}", "End Properties", ""]
    lines.append("Begin Nodes")
    for node_id, x, y in nodes:
        lines.append(f"{node_id} {x:.9f} {y:.9f} 0.0")
    lines += ["End Nodes", ""]
    lines.append(f"Begin Elements {element_name}")
    for element_id, (a, b, c) in enumerate(triangles, start=1):
        lines.append(f"{element_id} {properties_id} {a} {b} {c}")
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
    # Background grid: plain elements, one sub model part holding the clamped nodes.
    grid_nodes, grid_triangles = structured_triangles(0.0, GRID_X, 0.0, GRID_Y, CELL)
    clamped = [n for n, _x, y in grid_nodes if y <= FIXED_BELOW_Y]
    write_mdpa(
        HERE / "grid.mdpa",
        grid_nodes,
        grid_triangles,
        "Element2D3N",
        0,
        {"fixed_base": (clamped, [])},
    )

    # Body: becomes material points.
    body_nodes, body_triangles = structured_triangles(BODY_X0, BODY_X1, BODY_Y0, BODY_Y1, CELL)
    write_mdpa(
        HERE / "body.mdpa",
        body_nodes,
        body_triangles,
        "MPMUpdatedLagrangian2D3N",
        1,
        {
            "Parts_Body": (
                [n for n, _x, _y in body_nodes],
                list(range(1, len(body_triangles) + 1)),
            )
        },
    )

    materials = {
        "properties": [
            {
                "model_part_name": "Initial_MPM_Material.Parts_Body",
                "properties_id": 1,
                "Material": {
                    "constitutive_law": {"name": "JohnsonCookThermalPlastic2DPlaneStrainLaw"},
                    "Variables": kratos_material_variables(),
                    "Tables": {},
                },
            }
        ]
    }
    (HERE / "materials.json").write_text(json.dumps(materials, indent=2), encoding="utf-8")

    parameters = {
        "problem_data": {
            "problem_name": "jc_smoke",
            "parallel_type": "OpenMP",
            "echo_level": 1,
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
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": str(HERE / "body"),
            },
            "grid_model_import_settings": {
                "input_type": "mdpa",
                "input_filename": str(HERE / "grid"),
            },
            "material_import_settings": {
                "materials_filename": str(HERE / "materials.json"),
            },
            "time_stepping": {"time_step": TIME_STEP_S},
            "problem_domain_sub_model_part_list": ["Parts_Body"],
            "processes_sub_model_part_list": ["fixed_base"],
        },
        "processes": {
            "constraints_process_list": [
                {
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "Parameters": {
                        "model_part_name": "Background_Grid.fixed_base",
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
                        "model_part_name": "Initial_MPM_Material.Parts_Body",
                        "modulus": IMPACT_SPEED_M_S,
                        "direction": [0.0, -1.0, 0.0],
                    },
                }
            ],
        },
        "output_processes": {},
    }
    (HERE / "ProjectParameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")

    print(f"grid: {len(grid_nodes)} nodes, {len(grid_triangles)} elements, {len(clamped)} clamped")
    print(f"body: {len(body_nodes)} nodes, {len(body_triangles)} elements")
    print(f"material points: ~{len(body_triangles) * 3}")


if __name__ == "__main__":
    main()
