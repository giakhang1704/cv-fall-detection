"""Shared helpers: project root, hashing, JSON output and configuration loading."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Keys that store SHA-256 hashes of source files. They are recorded for provenance only and are
# ignored when deciding whether an existing cache can be reused (formatting must not invalidate it).
CODE_HASH_KEYS = {
    "selector_sha256",
    "extractor_sha256",
    "preprocessing_sha256",
    "feature_code",
    "prepare_code",
    "models_code",
}


def digest(path):
    """SHA-256 of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    """Path as shown to the user: relative to the project root when possible (never an absolute path)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def project_path(path):
    """Resolve a stored data path on any machine.

    Checkpoints written before this fix hold an absolute path of the author's computer; fall back
    to the same location below this project's `outputs/` folder.
    """
    path = Path(path)
    if not path.is_absolute():
        return ROOT / path
    if path.exists():
        return path
    parts = path.parts
    if "outputs" in parts:
        return ROOT.joinpath(*parts[parts.index("outputs"):])
    return path


def fingerprint(value):
    """SHA-256 of a JSON-serialisable value (key order independent)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def without_code_hashes(settings):
    """Copy of a settings dict without the source-code hash entries."""
    return {key: value for key, value in settings.items() if key not in CODE_HASH_KEYS}


def write_json(path, value):
    """Write ``value`` as indented UTF-8 JSON, creating parent directories; NaN is rejected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def config(path=None):
    """Load configs/default.json, optionally overridden by another JSON file, and validate it.

    `source_project` is the data root that contains `data/manifests` and `data/raw` (default: this project).
    """
    cfg = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
    if path:
        changes = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(changes) - set(cfg):
            raise ValueError("Unknown configuration keys")
        cfg.update(changes)
    cfg["source_project"] = str((ROOT / Path(cfg["source_project"])).resolve())
    from .preprocessing import VARIANTS

    if cfg["preprocessing"] not in VARIANTS or cfg["view"] not in ("person", "full"):
        raise ValueError("Invalid image preprocessing/view")
    if min(cfg["window"], cfg["stride"], cfg["sample_hz"], cfg["epochs"], cfg["threads"]) <= 0:
        raise ValueError("Invalid temporal/training settings")
    return cfg
