"""DataHub validation: enforce no-NaN, no-null, required fields only."""
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class DatasetValidationError(Exception):
    """Raised when a dataset fails validation."""
    pass


DATAHUB_ROOT = Path(__file__).parent


def load_manifest() -> Dict:
    """Load datahub manifest."""
    manifest_path = DATAHUB_ROOT / "manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dataset_config(dataset_id: str) -> Dict:
    """Get config for a specific dataset."""
    manifest = load_manifest()
    for ds in manifest["datasets"]:
        if ds["id"] == dataset_id:
            return ds
    raise ValueError(f"Dataset {dataset_id} not found in manifest")


def validate_csv_no_nan_no_null(filepath: Path, required_fields: List[str]) -> List[str]:
    """
    Validate CSV has no NaN/null and all required fields.
    
    Returns list of issues (empty = valid).
    """
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        
        # Check required fields present
        for field in required_fields:
            if field not in headers and field not in ['z_lens', 'z_source']:
                issues.append(f"Missing required field: {field}")
        
        # Check all values
        for row_num, row in enumerate(reader, start=2):
            for col, val in row.items():
                if val is None or val.strip() == '':
                    issues.append(f"Row {row_num}, {col}: empty value")
                elif val.lower() in ('nan', 'null', 'none', 'inf', '-inf', 'na'):
                    issues.append(f"Row {row_num}, {col}: invalid value '{val}'")
                elif col in ('x', 'y', 'sx', 'sy'):
                    try:
                        v = float(val)
                        if not np.isfinite(v):
                            issues.append(f"Row {row_num}, {col}: non-finite '{val}'")
                    except ValueError:
                        issues.append(f"Row {row_num}, {col}: not numeric '{val}'")
    
    return issues


def validate_meta_json(filepath: Path, required_fields: List[str]) -> List[str]:
    """Validate meta.json has required fields and no null/NaN."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # Check z_lens and z_source if required
    if 'z_lens' in required_fields:
        if meta.get('z_lens') is None:
            issues.append("Missing z_lens in meta")
        elif not isinstance(meta['z_lens'], (int, float)):
            issues.append(f"z_lens not numeric: {meta['z_lens']}")
    
    if 'z_source' in required_fields:
        if meta.get('z_source') is None:
            issues.append("Missing z_source in meta")
        elif not isinstance(meta['z_source'], (int, float)):
            issues.append(f"z_source not numeric: {meta['z_source']}")
    
    # Check data_quality flags
    dq = meta.get('data_quality', {})
    if not dq.get('all_values_finite', False):
        issues.append("data_quality.all_values_finite not True")
    if not dq.get('no_null', False):
        issues.append("data_quality.no_null not True")
    if not dq.get('no_nan', False):
        issues.append("data_quality.no_nan not True")
    
    return issues


def validate_snapshot(dataset_id: str) -> Tuple[bool, List[str]]:
    """
    Validate a complete snapshot.
    
    Returns (valid, issues).
    """
    config = get_dataset_config(dataset_id)
    snapshot_dir = DATAHUB_ROOT / "snapshots" / dataset_id
    required = config.get('required_fields', [])
    issues = []
    
    # Validate CSV
    if config['type'] == 'quad_points':
        csv_path = snapshot_dir / "images.csv"
    else:
        csv_path = snapshot_dir / "arc_points.csv"
    
    if not csv_path.exists():
        issues.append(f"CSV not found: {csv_path}")
    else:
        csv_issues = validate_csv_no_nan_no_null(csv_path, required)
        issues.extend(csv_issues)
    
    # Validate meta.json
    meta_path = snapshot_dir / "meta.json"
    if not meta_path.exists():
        issues.append(f"meta.json not found: {meta_path}")
    else:
        meta_issues = validate_meta_json(meta_path, required)
        issues.extend(meta_issues)
    
    # Validate provenance.md exists
    prov_path = snapshot_dir / "provenance.md"
    if not prov_path.exists():
        issues.append(f"provenance.md not found: {prov_path}")
    
    return len(issues) == 0, issues


def validate_all_snapshots() -> Dict[str, List[str]]:
    """Validate all snapshots in manifest."""
    manifest = load_manifest()
    results = {}
    
    for ds in manifest["datasets"]:
        valid, issues = validate_snapshot(ds["id"])
        results[ds["id"]] = {
            "valid": valid,
            "issues": issues
        }
    
    return results


def assert_snapshot_valid(dataset_id: str) -> None:
    """Assert snapshot is valid, raise if not."""
    valid, issues = validate_snapshot(dataset_id)
    if not valid:
        raise DatasetValidationError(
            f"Dataset {dataset_id} failed validation:\n" +
            "\n".join(f"  - {i}" for i in issues)
        )


if __name__ == "__main__":
    print("Validating all datahub snapshots...")
    results = validate_all_snapshots()
    
    all_valid = True
    for ds_id, result in results.items():
        status = "VALID" if result["valid"] else "INVALID"
        print(f"  {ds_id}: {status}")
        if not result["valid"]:
            all_valid = False
            for issue in result["issues"]:
                print(f"    - {issue}")
    
    if all_valid:
        print("\nAll snapshots valid!")
    else:
        print("\nSome snapshots have issues!")
        exit(1)
