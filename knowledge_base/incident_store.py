"""Simple in-memory incident store for past incident knowledge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Incident:
    """Record of a past incident for pattern matching."""

    fault_type: str
    service: str
    action_taken: str
    resolved: bool
    details: dict[str, Any] = field(default_factory=dict)


class IncidentStore:
    """In-memory incident store (no database, per spec)."""

    def __init__(self):
        self._incidents: list[Incident] = []

    def record(self, incident: Incident) -> None:
        self._incidents.append(incident)

    def find_similar(self, fault_type: str, service: str) -> list[Incident]:
        """Find past incidents with same fault type or service."""
        return [
            i for i in self._incidents
            if i.fault_type == fault_type or i.service == service
        ]

    @property
    def count(self) -> int:
        return len(self._incidents)
