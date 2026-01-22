"""No-Null/No-NaN validation for complete numeric outputs."""
import json
import csv
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


class NullOrNaNFoundError(Exception):
    """Raised when null or NaN found in data that should be complete."""
    pass


def is_null_or_nan(val: Any) -> bool:
    """Check if value is null, NaN, or Inf."""
    if val is None:
        return True
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return True
    if isinstance(val, str):
        lower = val.lower().strip()
        if lower in ('', 'nan', 'null', 'none', 'inf', '-inf', 'na', 'n/a'):
            return True
    return False


def validate_dict_no_null(
    d: Dict,
    path: str = "",
    skip_keys: List[str] = None
) -> List[str]:
    """
    Recursively validate dict has no null/NaN values.
    
    Returns list of paths where issues found (empty = valid).
    """
    issues = []
    skip_keys = skip_keys or []
    
    for key, val in d.items():
        current_path = f"{path}.{key}" if path else key
        
        if key in skip_keys:
            continue
        
        if is_null_or_nan(val):
            issues.append(f"{current_path}: {repr(val)}")
        elif isinstance(val, dict):
            issues.extend(validate_dict_no_null(val, current_path, skip_keys))
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                item_path = f"{current_path}[{i}]"
                if is_null_or_nan(item):
                    issues.append(f"{item_path}: {repr(item)}")
                elif isinstance(item, dict):
                    issues.extend(validate_dict_no_null(item, item_path, skip_keys))
    
    return issues


def validate_json_file_no_null(filepath: Union[str, Path]) -> List[str]:
    """Validate JSON file has no null/NaN values."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return validate_dict_no_null(data, str(filepath))


def validate_csv_file_no_null(
    filepath: Union[str, Path],
    numeric_columns: List[str] = None
) -> List[str]:
    """
    Validate CSV file has no null/NaN in numeric columns.
    
    If numeric_columns not specified, checks all columns.
    """
    issues = []
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # +2 for header + 1-index
            cols_to_check = numeric_columns or row.keys()
            for col in cols_to_check:
                if col in row:
                    val = row[col]
                    if is_null_or_nan(val):
                        issues.append(f"{filepath}:{row_num}:{col} = {repr(val)}")
    
    return issues


def validate_run_bundle_no_null(bundle_dir: Union[str, Path]) -> List[str]:
    """
    Validate entire run bundle has no null/NaN.
    
    Checks all JSON and CSV files recursively.
    """
    issues = []
    bundle_dir = Path(bundle_dir)
    
    # Check all JSON files
    for json_file in bundle_dir.rglob("*.json"):
        try:
            file_issues = validate_json_file_no_null(json_file)
            issues.extend(file_issues)
        except Exception as e:
            issues.append(f"{json_file}: Error reading - {e}")
    
    # Check all CSV files
    for csv_file in bundle_dir.rglob("*.csv"):
        try:
            file_issues = validate_csv_file_no_null(csv_file)
            issues.extend(file_issues)
        except Exception as e:
            issues.append(f"{csv_file}: Error reading - {e}")
    
    return issues


def assert_no_null_no_nan(data: Any, name: str = "data") -> None:
    """
    Assert data contains no null/NaN values.
    
    Raises NullOrNaNFoundError if any found.
    """
    if isinstance(data, dict):
        issues = validate_dict_no_null(data)
    elif isinstance(data, (str, Path)):
        path = Path(data)
        if path.is_dir():
            issues = validate_run_bundle_no_null(path)
        elif path.suffix == '.json':
            issues = validate_json_file_no_null(path)
        elif path.suffix == '.csv':
            issues = validate_csv_file_no_null(path)
        else:
            issues = [f"Unknown file type: {path}"]
    else:
        issues = []
        if is_null_or_nan(data):
            issues.append(f"{name}: {repr(data)}")
    
    if issues:
        raise NullOrNaNFoundError(
            f"No-Null/No-NaN contract violated in '{name}':\n" +
            "\n".join(f"  - {issue}" for issue in issues[:20]) +
            (f"\n  ... and {len(issues)-20} more" if len(issues) > 20 else "")
        )


def summarize_provenance(data: Dict) -> Dict[str, int]:
    """
    Count measured vs assumed values in data with provenance flags.
    
    Looks for *_is_measured fields and counts them.
    """
    measured = 0
    assumed = 0
    
    def count_recursive(d):
        nonlocal measured, assumed
        for key, val in d.items():
            if key.endswith('_is_measured'):
                if val is True:
                    measured += 1
                else:
                    assumed += 1
            elif isinstance(val, dict):
                count_recursive(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        count_recursive(item)
    
    count_recursive(data)
    return {
        'measured': measured,
        'assumed': assumed,
        'total': measured + assumed,
        'fraction_measured': measured / (measured + assumed) if (measured + assumed) > 0 else 1.0
    }
