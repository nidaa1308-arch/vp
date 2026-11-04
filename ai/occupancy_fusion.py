"""Prototype visibility-aware fusion of multiple camera observations.

This is a research baseline, not proven novel and not deployed to the backend.
Only real model-derived occupancy probabilities should be provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, isfinite


@dataclass(frozen=True)
class Observation:
    camera_id: str
    slot_code: str
    occupied_probability: float
    visibility: float
    image_quality: float
    timestamp: datetime

    def __post_init__(self):
        if not self.camera_id or not self.slot_code:
            raise ValueError("Camera and parking-space IDs are required")
        if self.timestamp.tzinfo is None:
            raise ValueError("Observation timestamps must be timezone-aware")
        for name in ("occupied_probability", "visibility", "image_quality"):
            value = getattr(self, name)
            if not isfinite(value) or not (0 <= value <= 1):
                raise ValueError(f"{name} must be a finite value in [0,1]")


class FusionEngine:
    """Per-slot latest-observation fusion with stale filtering and temporal gating."""

    def __init__(self, *, max_age_seconds=15, min_confidence=0.80,
                 stable_updates=3, min_visibility=0.15,
                 min_image_quality=0.15):
        if max_age_seconds <= 0 or stable_updates < 1:
            raise ValueError("Age and stable-updates parameters must be positive")
        if not (0.5 < min_confidence <= 1):
            raise ValueError("min_confidence must be in (0.5,1]")
        self.max_age = max_age_seconds
        self.min_confidence = min_confidence
        self.stable_updates = stable_updates
        self.min_visibility = min_visibility
        self.min_image_quality = min_image_quality
        self.latest = {}   # slot -> camera -> latest Observation
        self.streak = {}   # slot -> (candidate, count)

    def observe(self, observation: Observation, now=None):
        now = now or datetime.now(timezone.utc)
        cameras = self.latest.setdefault(observation.slot_code, {})
        previous = cameras.get(observation.camera_id)
        if previous is None or observation.timestamp > previous.timestamp:
            cameras[observation.camera_id] = observation
        result = self.fuse(observation.slot_code, now=now)
        candidate = result["candidate"]
        prior, count = self.streak.get(observation.slot_code, (None, 0))
        if candidate == "unknown":
            self.streak[observation.slot_code] = (None, 0)
        else:
            count = count + 1 if candidate == prior else 1
            self.streak[observation.slot_code] = (candidate, count)
        result["status"] = candidate if candidate != "unknown" and count >= self.stable_updates else "unknown"
        result["stable_updates"] = count if candidate != "unknown" else 0
        return result

    def fuse(self, slot_code, now=None):
        now = now or datetime.now(timezone.utc)
        active = []
        for camera_id, obs in self.latest.get(slot_code, {}).items():
            age = (now - obs.timestamp).total_seconds()
            if not 0 <= age <= self.max_age:
                continue
            if obs.visibility < self.min_visibility or obs.image_quality < self.min_image_quality:
                continue
            weight = obs.visibility * obs.image_quality * exp(-age / self.max_age)
            active.append((camera_id, obs.occupied_probability, weight))
        if not active:
            return {"candidate": "unknown", "status": "unknown",
                    "confidence": 0.0, "occupied_probability": None, "cameras_used": []}
        weighted = sum(prob * weight for _, prob, weight in active)
        total = sum(weight for _, _, weight in active)
        probability = weighted / total
        confidence = max(probability, 1 - probability)
        candidate = ("occupied" if probability >= 0.5 else "free") \
            if confidence >= self.min_confidence else "unknown"
        # Strong, comparably reliable disagreeing cameras need human review.
        reliable = [(prob, weight) for _, prob, weight in active if weight >= 0.25 * max(
            x[2] for x in active)]
        if any(prob >= 0.8 for prob, _ in reliable) and any(prob <= 0.2 for prob, _ in reliable):
            candidate = "unknown"
        return {"candidate": candidate, "status": candidate,
                "confidence": round(confidence, 6),
                "occupied_probability": round(probability, 6),
                "cameras_used": [camera for camera, _, _ in active]}
