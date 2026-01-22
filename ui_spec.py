"""UI Specification - JSON schemas and constants."""

INPUT_SNAPSHOT_SCHEMA = {
    "timestamp": "ISO datetime",
    "name": "run name",
    "raw_positions": "original text",
    "position_unit": "arcsec|mas|uas|rad",
    "distance_mode": "Normalized|Direct|Redshifts",
    "n_points": "int",
    "mode": "QUAD|RING|DOUBLE"
}

SCENE3D_SCHEMA = {
    "input_mode": "str",
    "inputs": {"D_L_val": "float", "D_L_unit": "str", "z_L": "float"},
    "internal_units": {"angle": "rad", "distance": "m", "time": "s", "mass": "kg"},
    "computed": {"D_L_m": "float", "theta_E_rad": "float", "R_E_m": "float"},
    "display": {"D_L": "1.3 Gpc", "theta_E": "1.2 arcsec"}
}

UNIT_RULES = """
ANGLES: <1e-9 rad → nrad, <1e-6 → µas, <1e-3 → mas, else arcsec
DISTANCES: <0.01 AU → km, <1000 AU → AU, <1000 ly → ly, <1000 pc → pc, 
           <1000 kpc → kpc, <1000 Mpc → Mpc, else Gpc
"""

print("UI Spec loaded. Schemas: INPUT_SNAPSHOT_SCHEMA, SCENE3D_SCHEMA")
