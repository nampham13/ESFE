"""Register custom backbone modules for Ultralytics model parsing."""

from __future__ import annotations

import ultralytics.nn.modules as ult_modules
import ultralytics.nn.tasks as ult_tasks

from my_backbone import ESFENet

_CUSTOM_MODULES = [ESFENet]


def register() -> None:
    """Inject custom modules into Ultralytics namespaces used by parse_model."""
    for cls in _CUSTOM_MODULES:
        setattr(ult_modules, cls.__name__, cls)
        setattr(ult_tasks, cls.__name__, cls)
