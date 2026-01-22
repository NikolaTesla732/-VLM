# defaults.py
from typing import Any, Optional

def pick(value: Any, cfg: Optional[object], attr_path: str, default: Any) -> Any:
    """
    value: значение аргумента (может быть None)
    cfg: конфиг (может быть None)
    attr_path: путь вида "clip_len_frames" или "paths.index_dir"
    default: fallback, если и value=None, и cfg=None/нет атрибута
    """
    if value is not None:
        return value
    if cfg is None:
        return default

    cur = cfg
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur
