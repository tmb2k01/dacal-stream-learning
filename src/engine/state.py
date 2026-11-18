from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineState:
    step: int = 0
    drift_mode: bool = False
    drift_steps_left: int = 0
    queried_count: int = 0
    labeled_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
