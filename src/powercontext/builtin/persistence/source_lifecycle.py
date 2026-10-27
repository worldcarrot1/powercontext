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

"""Durable Source-lifecycle registrations, event watermarks, and Topic holds."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE


@dataclass(frozen=True, slots=True)
class LifecycleNodeRow:
    """One stored lifecycle node: identity, dependency edge, and watermark state."""

    node_id: str
    kind: str
    reference: str
    parents: tuple[str, ...]
    policy: str | None
    revision: int | None
    sequence: int | None
    available: bool | None
    event_state: str | None
    topic_hold: str | None


class SourceLifecycleRepository:
    """Persist one node table: registration identity plus per-source event watermark."""

    async def load(self, connection: AsyncConnection, scope_id: str, /) -> dict[str, LifecycleNodeRow]:
        rows = (
            (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_NODES_TABLE).where(
                        SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id
                    ).order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.node_id)
                )
            )
            .mappings()
            .all()
        )
        return {str(row["node_id"]): _decode_row(row) for row in rows}

    async def add(self, connection: AsyncConnection, scope_id: str, row: LifecycleNodeRow, /) -> None:
        await connection.execute(
            insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                scope_id=scope_id,
                node_id=row.node_id,
                node_kind=row.kind,
                reference=row.reference,
                parents=_dumps(list(row.parents)),
                policy=row.policy,
                lifecycle_revision=row.revision,
                sequence=None,
                available=None,
                event_state=None,
                topic_hold=None,
            )
        )

    async def set_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
        *,
        sequence: int,
        available: bool,
        event_state: str,
    ) -> None:
        await connection.execute(
            update(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
            )
            .values(sequence=sequence, available=available, event_state=event_state)
        )

    async def set_hold(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
        *,
        topic_hold: str | None,
    ) -> None:
        await connection.execute(
            update(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
            )
            .values(topic_hold=topic_hold)
        )


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(value: str) -> Any:
    return json.loads(value)


def _decode_row(row: Mapping[Any, Any]) -> LifecycleNodeRow:
    return LifecycleNodeRow(
        node_id=str(row["node_id"]),
        kind=str(row["node_kind"]),
        reference=str(row["reference"]),
        parents=tuple(str(item) for item in _loads(str(row["parents"]))),
        policy=None if row["policy"] is None else str(row["policy"]),
        revision=None if row["lifecycle_revision"] is None else int(row["lifecycle_revision"]),
        sequence=None if row["sequence"] is None else int(row["sequence"]),
        available=None if row["available"] is None else bool(row["available"]),
        event_state=None if row["event_state"] is None else str(row["event_state"]),
        topic_hold=None if row["topic_hold"] is None else str(row["topic_hold"]),
    )


__all__ = ["LifecycleNodeRow", "SourceLifecycleRepository"]
