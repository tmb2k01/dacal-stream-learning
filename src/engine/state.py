from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class EngineState:
    step: int = 0
    drift_mode: bool = False
    queried_count: int = 0
    labeled_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)