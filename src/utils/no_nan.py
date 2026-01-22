"""No-NaN Contract utilities: strict enforcement against NaN/Inf values."""
import numpy as np
from typing import Any, Dict, List, Optional, Union
import json


class NaNDetectedError(Exception):
    """Raised when NaN or Inf values are detected."""
    pass


def assert_finite(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Assert all values in array are finite (no NaN, no Inf).
    
    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str
        Name for error message
        
    Returns
    -------
    np.ndarray
        Same array if valid
        
    Raises
    ------
    NaNDetectedError
        If any NaN or Inf values found
    """
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        raise NaNDetectedError(
            f"No-NaN Contract violated in '{name}': "
            f"found {nan_count} NaN, {inf_count} Inf values"
        )
    return arr


def sanitize_value(val: Any) -> Any:
    """
    Sanitize a single value: NaN/Inf -> None.
    
    For use when building JSON outputs where NaN would be invalid.
    """
    if val is None:
        return None
    if isinstance(val, (int, str, bool)):
        return val
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(val, np.floating):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.ndarray):
        return sanitize_array(val)
    if isinstance(val, (list, tuple)):
        return [sanitize_value(v) for v in val]
    if isinstance(val, dict):
        return sanitize_dict(val)
    return val


def sanitize_array(arr: np.ndarray) -> List:
    """Convert array to list, replacing NaN/Inf with None."""
    arr = np.asarray(arr)
    result = []
    for val in arr.flat:
        if np.isnan(val) or np.isinf(val):
            result.append(None)
        else:
            result.append(float(val))
    return result


def sanitize_dict(d: Dict) -> Dict:
    """Recursively sanitize dict, replacing NaN/Inf with None."""
    return {k: sanitize_value(v) for k, v in d.items()}


def sanitize_no_nan(obj: Any) -> Any:
    """
    Sanitize any object for JSON export: NaN/Inf -> None.
    
    Use this before json.dump() to ensure valid JSON output.
    """
    return sanitize_value(obj)


def validate_no_nan_in_dict(d: Dict, path: str = "") -> List[str]:
    """
    Recursively check dict for NaN/Inf values.
    
    Returns list of paths where NaN/Inf found.
    """
    issues = []
    for key, val in d.items():
        current_path = f"{path}.{key}" if path else key
        if val is None:
            continue
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                issues.append(f"{current_path}: {val}")
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    issues.append(f"{current_path}[{i}]: {v}")
                elif isinstance(v, dict):
                    issues.extend(validate_no_nan_in_dict(v, f"{current_path}[{i}]"))
        elif isinstance(val, dict):
            issues.extend(validate_no_nan_in_dict(val, current_path))
    return issues


def safe_divide(a: float, b: float, default: Optional[float] = None) -> Optional[float]:
    """
    Safe division: returns None (not NaN) if b is zero or result is non-finite.
    """
    if b == 0:
        return default
    result = a / b
    if not np.isfinite(result):
        return default
    return result


def safe_sqrt(x: float, default: Optional[float] = None) -> Optional[float]:
    """
    Safe square root: returns None (not NaN) if x is negative.
    """
    if x < 0:
        return default
    result = np.sqrt(x)
    if not np.isfinite(result):
        return default
    return float(result)


def parse_float_safe(s: str, default: Optional[float] = None) -> Optional[float]:
    """
    Parse string to float safely: returns default (not NaN) on failure.
    """
    try:
        val = float(s.strip())
        if not np.isfinite(val):
            return default
        return val
    except (ValueError, AttributeError):
        return default


class NoNaNJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to null."""
    
    def default(self, obj):
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return sanitize_array(obj)
        return super().default(obj)
    
    def encode(self, obj):
        return super().encode(sanitize_no_nan(obj))


def dump_json_no_nan(obj: Any, filepath: str, indent: int = 2) -> None:
    """
    Dump object to JSON file with No-NaN guarantee.
    
    Validates before writing and raises if NaN/Inf found.
    """
    sanitized = sanitize_no_nan(obj)
    issues = validate_no_nan_in_dict(sanitized) if isinstance(sanitized, dict) else []
    if issues:
        raise NaNDetectedError(f"No-NaN Contract violated: {issues}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sanitized, f, indent=indent, ensure_ascii=False)
