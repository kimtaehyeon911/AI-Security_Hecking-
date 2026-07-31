"""LLM knowledge-cutoff gate (anti-contamination).

Step 2 mandate: *"백테스트 구간은 사용 모델의 knowledge cutoff 이후로만 잡아라. cutoff
이전 구간 결과는 참고용(오염 가능)으로 별도 표기."* A model that memorized what a
stock did in the test window is not forecasting — it is recalling. So a decision
dated on/before the model's cutoff is flagged **contaminated** and its segment is
labelled reference-only, never counted as a clean result.

We refuse to invent cutoff dates: the registry ships with ``null`` cutoffs that
the user fills and marks ``verified``. An unknown/unverified/null cutoff yields
status ``UNKNOWN`` — the gate will not certify a segment as clean on a guess.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from enum import Enum
from importlib import resources
from pathlib import Path


class Contamination(str, Enum):
    CLEAN = "clean"              # decision date strictly after the model cutoff
    CONTAMINATED = "contaminated"  # on/before cutoff — 참고용(오염 가능)
    UNKNOWN = "unknown"          # cutoff not known/verified — cannot certify


class CutoffRegistry:
    """Maps model id → verified knowledge-cutoff date and classifies decision dates."""

    def __init__(self, models: dict[str, dict]) -> None:
        self._models = models

    @classmethod
    def load(cls, path: str | Path | None = None) -> CutoffRegistry:
        if path is None:
            text = resources.files("vts.backtest").joinpath("model_cutoffs.json").read_text()
        else:
            text = Path(path).read_text()
        data = json.loads(text)
        return cls(data.get("models", {}))

    def cutoff_for(self, model_id: str) -> date | None:
        entry = self._models.get(model_id)
        if not entry or not entry.get("verified") or not entry.get("cutoff"):
            return None
        return datetime.strptime(entry["cutoff"], "%Y-%m-%d").date()

    def classify(self, model_id: str, decision_date: date | datetime) -> Contamination:
        """Classify one decision date for one model."""
        d = decision_date.date() if isinstance(decision_date, datetime) else decision_date
        cutoff = self.cutoff_for(model_id)
        if cutoff is None:
            return Contamination.UNKNOWN
        return Contamination.CLEAN if d > cutoff else Contamination.CONTAMINATED

    def segment_status(
        self, model_id: str, dates: list[date | datetime]
    ) -> Contamination:
        """A segment is CLEAN only if every date is clean; UNKNOWN if cutoff is unset;
        otherwise CONTAMINATED if any date is on/before the cutoff."""
        if self.cutoff_for(model_id) is None:
            return Contamination.UNKNOWN
        statuses = {self.classify(model_id, d) for d in dates}
        return Contamination.CONTAMINATED if Contamination.CONTAMINATED in statuses else Contamination.CLEAN


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()
