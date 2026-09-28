from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
_PKG_DIR = Path(__file__).resolve().parent

def repo_root() -> Path:
    if _PKG_DIR.parent.name == 'src':
        return _PKG_DIR.parent.parent
    return Path.cwd()

def default_config_path() -> Path:
    return repo_root() / 'config' / 'default.yaml'

def load_config(path: Path | None=None) -> dict[str, Any]:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(Path.cwd() / 'config' / 'default.yaml')
    candidates.append(default_config_path())
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(encoding='utf-8') as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f'Invalid configuration in {candidate}')
            return data
    raise FileNotFoundError('No config/default.yaml file found.')