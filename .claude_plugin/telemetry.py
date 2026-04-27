"""Simulated NPU telemetry data model."""

import random
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class NPUStatus:
    online: bool
    temperature_c: float
    utilization_pct: float
    target: str = "SAKURA_II"
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TelemetrySnapshot:
    inference_latency_ms: float
    throughput_inferences_per_sec: float
    power_draw_watts: float
    window_ms: int
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


def sample_npu_status() -> NPUStatus:
    return NPUStatus(
        online=True,
        temperature_c=round(random.uniform(42.0, 68.0), 1),
        utilization_pct=round(random.uniform(0.0, 95.0), 1),
    )


def sample_telemetry(window_ms: int = 1000) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        inference_latency_ms=round(random.uniform(0.8, 4.5), 2),
        throughput_inferences_per_sec=round(random.uniform(200, 1200), 1),
        power_draw_watts=round(random.uniform(5.0, 15.0), 2),
        window_ms=window_ms,
    )
