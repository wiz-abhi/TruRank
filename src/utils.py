"""
Utility module — logging setup, config loading, and shared helpers.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOG_PATH = PROJECT_ROOT / "pipeline.log"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLE = PROJECT_ROOT / "data" / "sample"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EMBEDDINGS_CACHE = PROJECT_ROOT / "embeddings_cache"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_config_cache: Dict[str, Any] | None = None


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load the YAML configuration, caching after first read.

    Args:
        path: Optional override for config file location.

    Returns:
        Parsed config dictionary.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        _config_cache = yaml.safe_load(fh)
    return _config_cache


def reset_config_cache() -> None:
    """Reset the cached config (useful for testing)."""
    global _config_cache
    _config_cache = None


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Create a logger that writes to both console and ``pipeline.log``.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Configured :class:`logging.Logger`.
    """
    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_cfg.get("console", True):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler
    log_file = log_cfg.get("log_file", "pipeline.log")
    log_path = PROJECT_ROOT / log_file
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist.

    Args:
        path: Directory path to ensure.

    Returns:
        The same path, for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value: Any, default: float = 0.0) -> float:
    """Attempt to parse *value* as a float, returning *default* on failure.

    Args:
        value: Input value.
        default: Fallback.

    Returns:
        Parsed float or default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi].

    Args:
        value: Input.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped float.
    """
    return max(lo, min(hi, value))
