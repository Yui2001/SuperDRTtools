# -*- coding: utf-8 -*-

"""Safe project-file I/O for SuperDRTtools.

The project format is a ZIP container with a JSON manifest and NumPy arrays.
It deliberately avoids pickling live GUI, solver, CVXOPT, or Matplotlib objects,
which can make repeated save/load operations hang or terminate the process.
"""

from __future__ import annotations

import io
import json
import math
import os
import zipfile
from datetime import datetime
from typing import Any, Dict

import numpy as np


PROJECT_EXTENSION = ".sdrtp"
PROJECT_FORMAT = "SuperDRTtools Project"
PROJECT_VERSION = 3
_MANIFEST_NAME = "manifest.json"


def ensure_project_extension(path: str) -> str:
    path = str(path or "").strip()
    if not path:
        raise ValueError("Empty project path.")
    if not path.lower().endswith(PROJECT_EXTENSION):
        path += PROJECT_EXTENSION
    return path


def _encode_value(value: Any, arrays: Dict[str, np.ndarray], counter: list[int]) -> Any:
    """Convert supported Python/NumPy values into JSON plus external .npy arrays."""
    if value is None or isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return {"__type__": "float", "value": "nan"}
        if math.isinf(value):
            return {"__type__": "float", "value": "inf" if value > 0 else "-inf"}
        return value

    if isinstance(value, complex):
        return {"__type__": "complex", "real": float(value.real), "imag": float(value.imag)}

    if isinstance(value, np.generic):
        return _encode_value(value.item(), arrays, counter)

    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            return {
                "__type__": "object_array",
                "shape": list(arr.shape),
                "items": _encode_value(arr.tolist(), arrays, counter),
            }
        key = f"arrays/{counter[0]:08d}.npy"
        counter[0] += 1
        arrays[key] = np.ascontiguousarray(arr)
        return {"__type__": "ndarray", "path": key}

    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_value(v, arrays, counter) for v in value]}

    if isinstance(value, list):
        return [_encode_value(v, arrays, counter) for v in value]

    if isinstance(value, set):
        return {"__type__": "set", "items": [_encode_value(v, arrays, counter) for v in value]}

    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "items": [
                [_encode_value(k, arrays, counter), _encode_value(v, arrays, counter)]
                for k, v in value.items()
            ],
        }

    # Preserve pandas objects without pickling them.
    module_name = type(value).__module__
    class_name = type(value).__name__
    if module_name.startswith("pandas") and class_name == "DataFrame":
        split = value.to_dict(orient="split")
        return {"__type__": "dataframe", "data": _encode_value(split, arrays, counter)}
    if module_name.startswith("pandas") and class_name == "Series":
        payload = {"name": value.name, "index": value.index.tolist(), "data": value.tolist()}
        return {"__type__": "series", "data": _encode_value(payload, arrays, counter)}

    raise TypeError(f"Unsupported project value type: {type(value).__module__}.{type(value).__name__}")


def _decode_value(value: Any, zf: zipfile.ZipFile) -> Any:
    if isinstance(value, list):
        return [_decode_value(v, zf) for v in value]

    if not isinstance(value, dict) or "__type__" not in value:
        return value

    kind = value["__type__"]

    if kind == "float":
        token = value.get("value")
        return float(token)

    if kind == "complex":
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))

    if kind == "ndarray":
        raw = zf.read(value["path"])
        return np.load(io.BytesIO(raw), allow_pickle=False)

    if kind == "object_array":
        items = _decode_value(value.get("items", []), zf)
        arr = np.asarray(items, dtype=object)
        shape = tuple(int(x) for x in value.get("shape", arr.shape))
        return arr.reshape(shape)

    if kind == "tuple":
        return tuple(_decode_value(v, zf) for v in value.get("items", []))

    if kind == "set":
        return set(_decode_value(v, zf) for v in value.get("items", []))

    if kind == "dict":
        out = {}
        for key, item in value.get("items", []):
            out[_decode_value(key, zf)] = _decode_value(item, zf)
        return out

    if kind == "dataframe":
        split = _decode_value(value.get("data", {}), zf)
        try:
            import pandas as pd
            return pd.DataFrame(data=split.get("data", []), columns=split.get("columns", []), index=split.get("index", []))
        except Exception:
            return split

    if kind == "series":
        payload = _decode_value(value.get("data", {}), zf)
        try:
            import pandas as pd
            return pd.Series(payload.get("data", []), index=payload.get("index", []), name=payload.get("name"))
        except Exception:
            return payload

    raise ValueError(f"Unknown project value type: {kind}")


def save_project_state(path: str, state: dict) -> str:
    """Atomically save a project without pickling live runtime objects."""
    if not isinstance(state, dict):
        raise ValueError("Project state must be a dictionary.")

    path = ensure_project_extension(path)
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    manifest_state = _encode_value(state, arrays, [0])
    manifest = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "state": manifest_state,
    }

    temp_path = path + ".tmp"
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            zf.writestr(_MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
            for name, arr in arrays.items():
                with zf.open(name, "w", force_zip64=True) as fh:
                    np.save(fh, arr, allow_pickle=False)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return path


def load_project_state(path: str) -> dict:
    """Load and validate the safe ZIP-based project format."""
    if not zipfile.is_zipfile(path):
        raise ValueError(
            "This is not a valid current SuperDRTtools project file. "
            "Older pickle-based project files are not opened because they may crash the process."
        )

    with zipfile.ZipFile(path, "r") as zf:
        try:
            manifest = json.loads(zf.read(_MANIFEST_NAME).decode("utf-8"))
        except KeyError as exc:
            raise ValueError("Invalid SuperDRTtools project: manifest.json is missing.") from exc

        if manifest.get("format") != PROJECT_FORMAT:
            raise ValueError("Invalid SuperDRTtools project format.")
        if int(manifest.get("version", 0)) != PROJECT_VERSION:
            raise ValueError(f"Unsupported project version: {manifest.get('version')}")

        state = _decode_value(manifest.get("state"), zf)

    if not isinstance(state, dict):
        raise ValueError("Invalid project state.")
    return state


def save_project(path: str, gui) -> str:
    """Compatibility wrapper used by older GUI code."""
    if not hasattr(gui, "_build_project_state"):
        raise AttributeError("GUI does not provide _build_project_state().")
    return save_project_state(path, gui._build_project_state())


def load_project(path: str, gui) -> dict:
    """Compatibility wrapper used by older GUI code."""
    state = load_project_state(path)
    if not hasattr(gui, "_restore_project_state"):
        raise AttributeError("GUI does not provide _restore_project_state().")
    gui._restore_project_state(state)
    return state
