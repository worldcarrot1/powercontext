# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi v3 session capture: select-branch parsing, durable per-event Sources, incremental Connector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_NAME = "pi-event"
PI_EVENT_VERSION = "1"
PI_CONNECTOR_NAME = "pi-session-history"
PI_CONNECTOR_VERSION = "1"


class PiSessionInput(BaseModel):
    """One complete offline Pi v3 session archive plus an optional selected leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi event on the selected branch with its full original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """Resolved per-event Source that retains the complete input record for re-reads."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the delimiter-safe identity of one Pi event, decided only by session and event ids."""

    if not session_id or session_id != session_id.strip():
        raise ValueError("session id must be a non-empty string without leading or trailing whitespace")
    if not event_id or event_id != event_id.strip():
        raise ValueError("event id must be a non-empty string without leading or trailing whitespace")
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def _require_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without leading or trailing whitespace")
    return value


def _compact_json(arguments: object) -> str:
    """Render tool-call arguments with sorted keys, no spaces, and Unicode preserved."""

    return json.dumps(
        {} if arguments is None else arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _visible_body(content: object) -> str:
    """Join visible text and tool-call fragments; thinking, image, and unknown blocks are dropped."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif block_type == "toolCall":
                name = block.get("name") if isinstance(block.get("name"), str) else "unknown"
                call_id = block.get("id") if isinstance(block.get("id"), str) else ""
                parts.append(f"[tool_call {name} {call_id}] {_compact_json(block.get('arguments'))}")
        return "\n".join(parts)
    return ""


def _extract_event(record: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return (event_type, role, content) for a visible event, or None for control/no-body nodes."""

    event_type = record.get("type")
    if not isinstance(event_type, str):
        return None
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        if role == "toolResult":
            tool_name = message.get("toolName") if isinstance(message.get("toolName"), str) else "unknown"
            call_id = message.get("toolCallId") if isinstance(message.get("toolCallId"), str) else ""
            text = _visible_body(message.get("content"))
            if not text:
                return None
            body = f"[tool_result {tool_name} {call_id}] {text}"
        elif role == "bashExecution":
            command = message.get("command") if isinstance(message.get("command"), str) else ""
            output = message.get("output") if isinstance(message.get("output"), str) else ""
            body = f"$ {command}\n{output}"
        else:
            body = _visible_body(message.get("content"))
            if not body:
                return None
        return event_type, role, f"{role}: {body}"
    if event_type in ("compaction", "branch_summary"):
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return event_type, event_type, f"{event_type}: {summary}"
    if event_type == "custom_message":
        body = _visible_body(record.get("content"))
        if not body:
            return None
        return event_type, "custom", f"custom: {body}"
    return None


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse a Pi v3 archive and return the root-to-leaf ancestor chain of visible events atomically."""

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ValueError("each Pi session line must be a single JSON object") from error
        if not isinstance(record, dict):
            raise ValueError("each Pi session line must be a JSON object")
        records.append(record)

    if not records:
        raise ValueError("Pi session must begin with a session header record")
    header, *node_records = records
    if header.get("type") != "session":
        raise ValueError("the first Pi session record must have type 'session'")
    if header.get("version") != 3:
        raise ValueError("the Pi session header must declare version 3")
    session_id = _require_id(header.get("id"), "session id")

    nodes: dict[str, dict[str, Any]] = {}
    for record in node_records:
        event_id = _require_id(record.get("id"), "event id")
        if event_id in nodes:
            raise ValueError(f"duplicate Pi event id {event_id!r}")
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or parent_id not in nodes:
                raise ValueError(f"Pi event {event_id!r} references a parent that was not seen earlier")
        nodes[event_id] = record

    if not nodes:
        raise ValueError("Pi session must contain at least one event")

    if value.leaf_id is None:
        leaf_id = next(reversed(nodes))
    else:
        leaf_id = value.leaf_id
        if leaf_id not in nodes:
            raise ValueError(f"selected leaf {leaf_id!r} is not part of the Pi session graph")

    chain: list[str] = []
    visited: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in visited:
            raise ValueError("Pi session parent links contain a cycle")
        visited.add(current)
        chain.append(current)
        parent = nodes[current].get("parentId")
        current = parent if isinstance(parent, str) else None
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        record = nodes[event_id]
        extracted = _extract_event(record)
        if extracted is None:
            continue
        event_type, role, content = extracted
        parent_id = record.get("parentId")
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parent_id if isinstance(parent_id, str) else None,
                event_type=event_type,
                role=role,
                content=content,
                record=record,
            )
        )
    return tuple(events)


class PiEventTextEvidenceProjection:
    """Project one captured Pi event into the standard ``powercontext.text-evidence`` shape."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiEventDefinition:
    """Bind the Pi event adapter contract to version ``1`` and its text evidence projection."""

    input_class = PiEventInput
    name = PI_EVENT_NAME
    source_class = PiEventSource
    version = PI_EVENT_VERSION
    projections = (PiEventTextEvidenceProjection(),)

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        _require_id(value.session_id, "session id")
        _require_id(value.event_id, "event id")
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            definition_version=PI_EVENT_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = PiEventDefinition()


def _record_fingerprint(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _imported_from(checkpoint: object) -> dict[str, str]:
    if isinstance(checkpoint, dict):
        imported = checkpoint.get("imported")
        if isinstance(imported, dict):
            return {str(key): str(value) for key, value in imported.items()}
    return {}


class PiSessionConnector:
    """Re-parse a Pi file each run and import only the not-yet-captured events on the selected branch."""

    name = PI_CONNECTOR_NAME
    version = PI_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self._leaf_id))

        imported = _imported_from(session.checkpoint)
        new_imported = dict(imported)
        for event in events:
            identity = pi_event_source_id(event.session_id, event.event_id)
            fingerprint = _record_fingerprint(event.record)
            if identity in imported:
                if imported[identity] != fingerprint:
                    raise ValueError(f"persisted Pi event {identity!r} was rewritten")
                continue
            outcome = await session.submit(identity, PI_EVENT_NAME, event)
            if outcome.status is ConnectorSubmissionStatus.ACCEPTED:
                new_imported[identity] = fingerprint

        checkpoint: dict[str, Any] = {"version": 1, "imported": new_imported}
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventDefinition",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
