"""Copper block impacting a frictional floor — verifies Zorev friction acts.

Cutting geometry is not needed to test a friction law, and a cutting case is a
slow way to find out that the plumbing is wrong. This case presses copper into
a slip boundary hard enough that contact pressure reaches the GPa range, so the
Zorev shear cap genuinely engages rather than sitting dormant in the sliding
regime.

The observable is lateral spread: a block landing on a frictionless floor
spreads more than one landing on a frictional floor. Run at two friction
coefficients and the difference is the friction doing work.

    uv run python bench/zorev_friction/build_case.py
"""

import json
import pathlib

from quinney.materials.ofhc_copper import kratos_material_variables

HERE = pathlib.Path(__file__).parent

CELL = 500e-6
GRID_X, GRID_Y = 6.0e-3, 16.0e-3
FLOOR_Y = 6.0e-3  # the slip boundary sits here

BODY_X0, BODY_X1 = 1.0e-3, 5.0e-3
BODY_Y0, BODY_Y1 = FLOOR_Y, 14.0e-3

IMPACT_SPEED_M_S = 150.0
TIME_STEP_S = 2.0e-8
END_TIME_S = 4.0e-6

# Penalty stiffness for the slip boundary. Both must be positive or Kratos
# leaves friction switched off entirely.
PENALTY_COEFFICIENT = 1.0e13
TANGENTIAL_PENALTY_COEFFICIENT = 1.0e13


def structured_triangles(x0, x1, y0, y1, cell):
    nx, ny = round((x1 - x0) / cell), round((y1 - y0) / cell)
    grid, node_id = {}, 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            node_id += 1
            grid[(i, j)] = (node_id, x0 + i * cell, y0 + j * cell)
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a, b = grid[(i, j)][0], grid[(i + 1, j)][0]
            c, d = grid[(i + 1, j + 1)][0], grid[(i, j + 1)][0]
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return grid, list(grid.values()), triangles


def main():
    grid_map, grid_nodes, grid_triangles = structured_triangles(0.0, GRID_X, 0.0, GRID_Y, CELL)

    # Floor nodes, left to right, become the slip boundary's line conditions.
    row = round(FLOOR_Y / CELL)
    nx = round(GRID_X / CELL)
    floor_ids = [grid_map[(i, row)][0] for i in range(nx + 1)]
    conditions = [
        (index + 1, floor_ids[index], floor_ids[index + 1]) for index in range(len(floor_ids) - 1)
    ]
    # Everything strictly below the floor is clamped, so the floor itself is
    # free to carry contact rather than being held rigid by its neighbours.
    clamped = [n for n, _x, y in grid_nodes if y < FLOOR_Y - 0.5 * CELL]

    lines = [
        "Begin ModelPartData",
        "End ModelPartData",
        "",
        "Begin Properties 0",
        "End Properties",
        "",
        "Begin Nodes",
    ]
    lines += [f"{n} {x:.9f} {y:.9f} 0.0" for n, x, y in grid_nodes]
    lines += ["End Nodes", "", "Begin Elements Element2D3N"]
    lines += [f"{i} 0 {a} {b} {c}" for i, (a, b, c) in enumerate(grid_triangles, 1)]
    lines += ["End Elements", "", "Begin Conditions LineCondition2D2N"]
    lines += [f"{cid} 0 {a} {b}" for cid, a, b in conditions]
    lines += ["End Conditions", ""]
    lines += ["Begin SubModelPart slip_floor", "  Begin SubModelPartNodes"]
    lines += [f"  {n}" for n in floor_ids]
    lines += ["  End SubModelPartNodes", "  Begin SubModelPartConditions"]
    lines += [f"  {cid}" for cid, *_ in conditions]
    lines += ["  End SubModelPartConditions", "End SubModelPart", ""]
    lines += ["Begin SubModelPart fixed_base", "  Begin SubModelPartNodes"]
    lines += [f"  {n}" for n in clamped]
    lines += ["  End SubModelPartNodes", "End SubModelPart", ""]
    (HERE / "grid.mdpa").write_text("\n".join(lines), encoding="utf-8")

    _, body_nodes, body_triangles = structured_triangles(BODY_X0, BODY_X1, BODY_Y0, BODY_Y1, CELL)
    body = [
        "Begin ModelPartData",
        "End ModelPartData",
        "",
        "Begin Properties 1",
        "End Properties",
        "",
        "Begin Nodes",
    ]
    body += [f"{n} {x:.9f} {y:.9f} 0.0" for n, x, y in body_nodes]
    body += ["End Nodes", "", "Begin Elements MPMUpdatedLagrangian2D3N"]
    body += [f"{i} 1 {a} {b} {c}" for i, (a, b, c) in enumerate(body_triangles, 1)]
    body += ["End Elements", "", "Begin SubModelPart Parts_Block", "  Begin SubModelPartNodes"]
    body += [f"  {n}" for n, _x, _y in body_nodes]
    body += ["  End SubModelPartNodes", "  Begin SubModelPartElements"]
    body += [f"  {i}" for i in range(1, len(body_triangles) + 1)]
    body += ["  End SubModelPartElements", "End SubModelPart", ""]
    (HERE / "body.mdpa").write_text("\n".join(body), encoding="utf-8")

    materials = {
        "properties": [
            {
                "model_part_name": "Initial_MPM_Material.Parts_Block",
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

    def parameters(friction_coefficient, name):
        return {
            "problem_data": {
                "problem_name": name,
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
                "model_import_settings": {
                    "input_type": "mdpa",
                    "input_filename": str(HERE / "body"),
                },
                "grid_model_import_settings": {
                    "input_type": "mdpa",
                    "input_filename": str(HERE / "grid"),
                },
                "material_import_settings": {"materials_filename": str(HERE / "materials.json")},
                "time_stepping": {"time_step": TIME_STEP_S},
                # mpm_analysis reads this before the solver applies its own
                # defaults, so it must be present even when empty.
                # NORMAL is not among the solver's own nodal variables, and the slip
                # boundary writes its computed normals there.
                "auxiliary_variables_list": ["NORMAL"],
                "problem_domain_sub_model_part_list": ["Parts_Block"],
                "processes_sub_model_part_list": ["slip_floor", "fixed_base"],
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
                    },
                    {
                        "python_module": "apply_mpm_slip_boundary_process",
                        "kratos_module": "KratosMultiphysics.MPMApplication",
                        # mpm_analysis only adds the FRICTION_STATE / STICK_FORCE
                        # nodal variables when it sees this exact process_name.
                        "process_name": "ApplyMPMSlipBoundaryProcess",
                        "Parameters": {
                            "model_part_name": "Background_Grid.slip_floor",
                            "friction_coefficient": friction_coefficient,
                            "tangential_penalty_coefficient": (
                                TANGENTIAL_PENALTY_COEFFICIENT if friction_coefficient > 0 else 0.0
                            ),
                        },
                    },
                ],
                "loads_process_list": [],
                "list_other_processes": [
                    {
                        "python_module": "assign_initial_velocity_to_material_point_process",
                        "kratos_module": "KratosMultiphysics.MPMApplication",
                        "Parameters": {
                            "model_part_name": "Initial_MPM_Material.Parts_Block",
                            "modulus": IMPACT_SPEED_M_S,
                            "direction": [0.0, -1.0, 0.0],
                        },
                    }
                ],
            },
            "output_processes": {},
        }

    for coefficient, name in ((0.0, "frictionless"), (0.5, "frictional")):
        (HERE / f"ProjectParameters_{name}.json").write_text(
            json.dumps(parameters(coefficient, name), indent=2), encoding="utf-8"
        )

    print(f"grid {len(grid_nodes)} nodes, {len(grid_triangles)} elements")
    print(f"slip floor: {len(conditions)} line conditions on {len(floor_ids)} nodes")
    print(f"clamped below floor: {len(clamped)} nodes")
    print(f"block {len(body_nodes)} nodes, {len(body_triangles)} elements")
    print(f"steps {round(END_TIME_S / TIME_STEP_S)}")


if __name__ == "__main__":
    main()
