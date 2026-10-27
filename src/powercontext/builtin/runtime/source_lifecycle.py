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

"""Source-conditioned withdrawal and restore propagation over one scope's evidence graph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator
from sqlalchemy import select

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation
from powercontext.builtin.artifacts.topic_memory import TopicMemory
from powercontext.builtin.persistence.artifacts import ArtifactRepository
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import LifecycleNodeRow, SourceLifecycleRepository
from powercontext.builtin.persistence.sources import SourceRepository
from powercontext.builtin.persistence.sqlite.topic_memory_index import (
    _DELETE_CHUNK_FTS_SQL,
    _DELETE_TOPIC_FTS_SQL,
    _INSERT_CHUNK_FTS_SQL,
    _INSERT_TOPIC_FTS_SQL,
)
from powercontext.builtin.persistence.tables import (
    TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE,
    TOPIC_MEMORY_ACTIVE_TOPICS_TABLE,
)
from powercontext.builtin.persistence.topic_memory import TopicMemoryRepository
from powercontext.builtin.sources import validate_scope_id
from powercontext.builtin.statistics import ModelUsagePurpose
from powercontext.errors import ArtifactNotFoundError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

__all__ = [
    "LifecycleApplication",
    "ScopedSourceLifecycleService",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleWiring",
]

_TOPIC_REFERENCE_ADAPTER = TypeAdapter(ArtifactRef)
_SOURCE_REFERENCE_ADAPTER = TypeAdapter(SourceRef)
_MEMORY_REFERENCE_ADAPTER = TypeAdapter(MemoryCitation)
_HELD_MARKER = "held"


class SourceLifecycleError(ValueError):
    """Raised when one lifecycle command violates the scope's evidence graph."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SourceLifecycleEvent(BaseModel):
    """One availability observation for a registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: Annotated[str, Field(min_length=1, max_length=MAX_SOURCE_ID_LENGTH)]
    revision: Annotated[int, Field(ge=1)]
    sequence: Annotated[int, Field(ge=1)]
    available: bool
    reason: Annotated[str, Field(min_length=1, max_length=2_000)]

    @field_validator("node_id", "reason")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle text must contain non-whitespace content")  # noqa: TRY003
        return value


@dataclass(frozen=True, slots=True)
class SourceLifecycleWiring:
    """Persistence seams assembled once for every scoped lifecycle service."""

    database: AsyncDatabase
    nodes: SourceLifecycleRepository
    topic_memories: TopicMemoryRepository
    sources: SourceRepository
    artifacts: ArtifactRepository


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _event_identity(event: SourceLifecycleEvent) -> str:
    return _canonical_json(
        {"available": event.available, "revision": event.revision, "sequence": event.sequence},
    )


def _reference_of(row: LifecycleNodeRow) -> SourceRef | MemoryCitation | ArtifactRef:
    if row.kind == "source":
        return _SOURCE_REFERENCE_ADAPTER.validate_json(row.reference)
    if row.kind == "memory":
        return _MEMORY_REFERENCE_ADAPTER.validate_json(row.reference)
    return _TOPIC_REFERENCE_ADAPTER.validate_json(row.reference)


def _identity_of(row: LifecycleNodeRow) -> str:
    return _canonical_json(
        {
            "kind": row.kind,
            "reference": json.loads(row.reference),
            "parents": list(row.parents),
            "policy": row.policy,
            "revision": row.revision,
        },
    )


def _active_map(rows: Mapping[str, LifecycleNodeRow]) -> dict[str, bool]:
    """Evaluate source availability through the all/any edges to a fixed point."""

    state: dict[str, bool] = {}
    for node_id, row in rows.items():
        if row.kind == "source":
            state[node_id] = True if row.available is None else row.available
        else:
            state[node_id] = False
    for _round in range(len(rows) + 1):
        changed = False
        for node_id, row in rows.items():
            if row.kind == "source":
                continue
            support = [state.get(parent, False) for parent in row.parents]
            value = all(support) if row.policy == "all" else any(support)
            if value != state[node_id]:
                state[node_id] = value
                changed = True
        if not changed:
            break
    return state


def _validate_node_id(node_id: object) -> str:
    if not isinstance(node_id, str) or not node_id.strip() or len(node_id) > MAX_SOURCE_ID_LENGTH:
        raise SourceLifecycleError("node-id-invalid")
    return node_id


def _validate_binding(parents: tuple[str, ...], policy: object) -> tuple[str, ...]:
    if not isinstance(parents, tuple) or not parents:
        raise SourceLifecycleError("parents-required")
    if not isinstance(policy, str) or policy not in {"all", "any"}:
        raise SourceLifecycleError("policy-invalid")
    normalized: list[str] = []
    for parent in parents:
        normalized.append(_validate_node_id(parent))
    if len(set(normalized)) != len(normalized):
        raise SourceLifecycleError("parents-duplicate")
    return tuple(normalized)


class ScopedSourceLifecycleService:
    """Register, propagate, and resolve one scope's evidence DAG."""

    def __init__(self, runtime: BuiltinRuntime, wiring: SourceLifecycleWiring, scope_id: str) -> None:
        self._runtime = runtime
        self._wiring = wiring
        self._scope = validate_scope_id(scope_id)

    async def register_source(
        self,
        node_id: str,
        source_ref: SourceRef,
        *,
        revision: int = 1,
    ) -> None:
        node_id = _validate_node_id(node_id)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SourceLifecycleError("revision-invalid")
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("reference-type")
        row = LifecycleNodeRow(
            node_id=node_id,
            kind="source",
            reference=_canonical_json(source_ref.model_dump(mode="json")),
            parents=(),
            policy=None,
            revision=revision,
            sequence=None,
            available=None,
            event_state=None,
            topic_hold=None,
        )
        await self._register(row)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        node_id = _validate_node_id(node_id)
        parents = _validate_binding(parents, policy)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("reference-type")
        row = LifecycleNodeRow(
            node_id=node_id,
            kind="memory",
            reference=_canonical_json(citation.model_dump(mode="json")),
            parents=parents,
            policy=str(policy),
            revision=None,
            sequence=None,
            available=None,
            event_state=None,
            topic_hold=None,
        )
        await self._require_memory_readable(row)
        await self._register(row)

    async def register_topic(
        self,
        node_id: str,
        artifact: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        node_id = _validate_node_id(node_id)
        parents = _validate_binding(parents, policy)
        if not isinstance(artifact, ArtifactRef) or artifact.family != TopicMemory.family:
            raise SourceLifecycleError("reference-type")
        row = LifecycleNodeRow(
            node_id=node_id,
            kind="topic",
            reference=_canonical_json(artifact.model_dump(mode="json")),
            parents=parents,
            policy=str(policy),
            revision=None,
            sequence=None,
            available=None,
            event_state=None,
            topic_hold=None,
        )
        await self._register(row)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event-type")
        async with self._runtime._scope_operation(self._scope):
            async with self._wiring.database.transaction() as connection:
                rows = await self._wiring.nodes.load(connection, self._scope)
                row = rows.get(event.node_id)
                if row is None or row.kind != "source":
                    raise SourceLifecycleError("unknown-source")
                if row.revision != event.revision:
                    raise SourceLifecycleError("revision-mismatch")
                identity = _event_identity(event)
                if row.sequence is not None:
                    if event.sequence < row.sequence:
                        raise SourceLifecycleError("sequence-stale")
                    if event.sequence == row.sequence and row.event_state != identity:
                        raise SourceLifecycleError("sequence-conflict")
                await self._wiring.nodes.set_event(
                    connection,
                    self._scope,
                    event.node_id,
                    sequence=event.sequence,
                    available=event.available,
                    event_state=identity,
                )
            return await self._propagate(event.reason)

    async def resolve(
        self,
        node_id: str,
        *,
        purpose: Literal["generation", "audit"],
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose-invalid")
        async with self._runtime._scope_operation(self._scope):
            async with self._wiring.database.transaction() as connection:
                rows = await self._wiring.nodes.load(connection, self._scope)
        row = rows.get(_validate_node_id(node_id))
        if row is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "generation" and not _active_map(rows).get(row.node_id, False):
            raise SourceLifecycleError("not-available")
        return _reference_of(row)

    async def _register(self, row: LifecycleNodeRow) -> None:
        async with self._runtime._scope_operation(self._scope):
            async with self._wiring.database.transaction() as connection:
                rows = await self._wiring.nodes.load(connection, self._scope)
                existing = rows.get(row.node_id)
                if existing is not None:
                    if _identity_of(existing) == _identity_of(row):
                        return
                    raise SourceLifecycleError("registration-conflict")
                if row.kind != "source":
                    for parent in row.parents:
                        parent_row = rows.get(parent)
                        if parent_row is None:
                            raise SourceLifecycleError("parent-unknown")
                    if self._creates_cycle(rows, row):
                        raise SourceLifecycleError("cycle-detected")
                await self._require_registered_reference(connection, row)
                await self._wiring.nodes.add(connection, self._scope, row)
                rows[row.node_id] = row
            if row.kind == "source":
                return
            active = _active_map(rows)
            if not active.get(row.node_id, False):
                await self._propagate_to(row, reason="source-lifecycle")

    async def _require_memory_readable(self, row: LifecycleNodeRow) -> None:
        citation = _MEMORY_REFERENCE_ADAPTER.validate_json(row.reference)
        async with self._runtime._context(self._scope) as context:
            try:
                await context.artifacts.memory.validate_citation(citation)
            except Exception as error:
                raise SourceLifecycleError("reference-unreadable") from error

    async def _require_registered_reference(self, connection: Any, row: LifecycleNodeRow) -> None:
        try:
            if row.kind == "topic":
                await self._wiring.artifacts.get(connection, self._scope, _reference_of(row))
            elif row.kind == "memory":
                citation = _MEMORY_REFERENCE_ADAPTER.validate_json(row.reference)
                if citation.memory_ref.family != "memory":
                    raise SourceLifecycleError("reference-family")
            elif row.kind == "source":
                await self._wiring.sources.get(connection, self._scope, _reference_of(row))
        except SourceLifecycleError:
            raise
        except Exception as error:
            raise SourceLifecycleError("reference-unreadable") from error

    @staticmethod
    def _creates_cycle(rows: Mapping[str, LifecycleNodeRow], row: LifecycleNodeRow) -> bool:
        if row.node_id in row.parents:
            return True
        pending = list(row.parents)
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == row.node_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            parent_row = rows.get(current)
            if parent_row is not None:
                pending.extend(parent_row.parents)
        return False

    async def _propagate(self, reason: str) -> list[str]:
        async with self._wiring.database.transaction() as connection:
            rows = await self._wiring.nodes.load(connection, self._scope)
        active = _active_map(rows)
        memory_rows = [row for row in rows.values() if row.kind == "memory"]
        topic_rows = [row for row in rows.values() if row.kind == "topic"]
        if memory_rows:
            async with self._runtime._context(
                self._scope,
                embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING,
            ) as context:
                await self._sync_memories(context, memory_rows, active, reason)
        if topic_rows:
            async with self._wiring.database.transaction() as connection:
                await self._sync_topics(connection, topic_rows, active)
        return sorted(node_id for node_id, is_active in active.items() if is_active)

    async def _propagate_to(self, row: LifecycleNodeRow, *, reason: str) -> None:
        if row.kind == "memory":
            async with self._runtime._context(
                self._scope,
                embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING,
            ) as context:
                await self._sync_memories(context, [row], {row.node_id: False}, reason)
        elif row.kind == "topic":
            async with self._wiring.database.transaction() as connection:
                await self._sync_topics(connection, [row], {row.node_id: False})

    async def _sync_memories(
        self,
        context: Any,
        memory_rows: list[LifecycleNodeRow],
        active: Mapping[str, bool],
        reason: str,
    ) -> None:
        service = context.artifacts.memory
        try:
            current = await service.head(context.artifacts.memory_artifact_id)
        except ArtifactNotFoundError:
            return
        versions = {entry.entry_id: entry for entry in await service.entries(current)}
        states = {item.entry_id: item.state for item in current.content.manifest.entries}
        retires: list[Any] = []
        activates: list[Any] = []
        for row in memory_rows:
            citation = _MEMORY_REFERENCE_ADAPTER.validate_json(row.reference)
            version = versions.get(citation.entry_id)
            if version is None or version.entry_version_id != citation.entry_version_id:
                continue
            state = states.get(citation.entry_id)
            if state is None:
                continue
            if active.get(row.node_id, False):
                if state == "inactive":
                    activates.append(version)
            elif state == "active":
                retires.append(version)
        if not retires and not activates:
            return
        async with self._runtime._locked(self._scope):
            updated = current
            if retires:
                updated = await service.forget(updated, entries=tuple(retires), reason=reason)
            if activates:
                updated = await service.reactivate(updated, entries=tuple(activates), reason=reason)

    async def _sync_topics(
        self,
        connection: Any,
        topic_rows: list[LifecycleNodeRow],
        active: Mapping[str, bool],
    ) -> None:
        """Exclude withdrawn topics from the retrieval index without touching published rows.

        The Topic Memory store keeps one active head per artifact and rebuilds FTS from
        those rows only when its marker is absent, so removal from the FTS channels is
        the durable exclusion: search and Context stop seeing the topic while exact
        reads, browses, and the restart invariants keep working. A held Topic keeps the
        active rows untouched and carries a hold marker for the restore pass.
        """

        for row in topic_rows:
            reference = _TOPIC_REFERENCE_ADAPTER.validate_json(row.reference)
            artifact_id = reference.artifact_id
            if active.get(row.node_id, False):
                if row.topic_hold is None:
                    continue
                if connection.dialect.name == "sqlite":
                    topic_row = (
                        (
                            await connection.execute(
                                select(
                                    TOPIC_MEMORY_ACTIVE_TOPICS_TABLE.c.revision,
                                    TOPIC_MEMORY_ACTIVE_TOPICS_TABLE.c.title,
                                    TOPIC_MEMORY_ACTIVE_TOPICS_TABLE.c.summary,
                                    TOPIC_MEMORY_ACTIVE_TOPICS_TABLE.c.searchable_text,
                                ).where(
                                    TOPIC_MEMORY_ACTIVE_TOPICS_TABLE.c.scope_id == self._scope,
                                    TOPIC_MEMORY_ACTIVE_TOPICS_TABLE.c.artifact_id == artifact_id,
                                )
                            )
                        )
                        .mappings()
                        .one_or_none()
                    )
                    if topic_row is not None:
                        identity = {"scope_id": self._scope, "artifact_id": artifact_id}
                        topic_values = {**identity, **dict(topic_row)}
                        await connection.execute(_INSERT_TOPIC_FTS_SQL, topic_values)
                        chunk_rows = (
                            (
                                await connection.execute(
                                    select(
                                        TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE.c.revision,
                                        TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE.c.chunk_ordinal,
                                        TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE.c.start_offset,
                                        TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE.c.chunk_text,
                                        TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE.c.searchable_text,
                                    ).where(
                                        TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE.c.scope_id == self._scope,
                                        TOPIC_MEMORY_ACTIVE_CHUNKS_TABLE.c.artifact_id == artifact_id,
                                    )
                                )
                            )
                            .mappings()
                            .all()
                        )
                        for chunk in chunk_rows:
                            await connection.execute(
                                _INSERT_CHUNK_FTS_SQL,
                                {**topic_values, **dict(chunk)},
                            )
                await self._wiring.nodes.set_hold(connection, self._scope, row.node_id, topic_hold=None)
                continue
            if row.topic_hold is not None:
                continue
            if connection.dialect.name == "sqlite":
                identity = {"scope_id": self._scope, "artifact_id": artifact_id}
                await connection.execute(_DELETE_TOPIC_FTS_SQL, identity)
                await connection.execute(_DELETE_CHUNK_FTS_SQL, identity)
            await self._wiring.nodes.set_hold(connection, self._scope, row.node_id, topic_hold=_HELD_MARKER)


class LifecycleApplication:
    """Select the lifecycle service for one scope, mirroring the other Runtime families."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleService:
        wiring = self._runtime._lifecycle_wiring
        if wiring is None:
            raise SourceLifecycleError("lifecycle-unavailable")
        return ScopedSourceLifecycleService(self._runtime, wiring, scope_id)
