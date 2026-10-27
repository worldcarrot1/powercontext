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

"""Count-gated acceptance of Source-driven processing (challenge processing-01)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext._logging import log_safely
from powercontext.builtin.artifacts.topic_memory import TOPIC_MEMORY_SOURCE_WINDOW_BINDING
from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.processing import ArtifactProcessingPendingRepository
from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.tables import SOURCE_JOURNAL_HEADS_TABLE, SOURCES_TABLE
from powercontext.builtin.runtime.config import BuiltinConfig
from powercontext.builtin.sources import validate_scope_id
from powercontext.builtin.triggers import SOURCE_WINDOW_TRIGGER_NAME

logger = logging.getLogger(__name__)

__all__ = ["SourceCountTrigger", "maybe_note_source_count"]

_INTENTS = ArtifactProcessingIntentRepository()
_CURSORS = SourceCursorRepository()
_PENDING = ArtifactProcessingPendingRepository()


@dataclass(slots=True)
class SourceCountTrigger:
    """Accept one existing processing call per scope and Family once its backlog qualifies."""

    database: AsyncDatabase
    memory_threshold: int | None = None
    topic_threshold: int | None = None
    notifier: Callable[[str], None] | None = None
    _family_bindings: dict[str, str] = field(
        default_factory=lambda: {
            "memory": SOURCE_WINDOW_TRIGGER_NAME,
            "topic-memory": TOPIC_MEMORY_SOURCE_WINDOW_BINDING,
        },
        repr=False,
    )

    @property
    def enabled(self) -> bool:
        return self.memory_threshold is not None or self.topic_threshold is not None

    def threshold_for(self, family: str) -> int | None:
        if family == "memory":
            return self.memory_threshold
        if family == "topic-memory":
            return self.topic_threshold
        return None

    async def note(self, scope_id: str) -> None:
        """Open one accepted request per Family whose post-cursor journal count reached its threshold.

        The count is the number of persisted journal identities ahead of that
        processor's committed cursor, so repeated captures of one identity and
        receipt trust upgrades (which never advance the head) do not add to it.
        A lineage-only server Source occupies a journal position and starts
        processing while remaining excluded from generation evidence by the
        existing eligibility rules. Acceptance is the same durable request the
        explicit flush API raises, so the Supervisor admits one existing
        invocation; an accepted-but-unhandled request is never stacked.
        """

        if not self.enabled:
            return
        scope = validate_scope_id(scope_id)
        fired: list[str] = []
        async with self.database.transaction() as connection:
            # Lock the Source head before Pending or Intent, matching capture order.
            await connection.execute(
                update(SOURCE_JOURNAL_HEADS_TABLE)
                .where(SOURCE_JOURNAL_HEADS_TABLE.c.scope_id == scope)
                .values(position=SOURCE_JOURNAL_HEADS_TABLE.c.position)
            )
            high = int(
                await connection.scalar(
                    select(func.coalesce(func.max(SOURCES_TABLE.c.journal_position), 0)).where(
                        SOURCES_TABLE.c.scope_id == scope
                    )
                )
                or 0
            )
            for family, threshold in (
                ("memory", self.memory_threshold),
                ("topic-memory", self.topic_threshold),
            ):
                if threshold is None:
                    continue
                binding = self._family_bindings[family]
                if not await self._qualify(connection, scope, binding, high, threshold):
                    continue
                if family == "topic-memory":
                    # The wave target freezes at the accepted head, exactly as
                    # an explicit flush does; the threshold never re-gates later
                    # windows of an already accepted wave.
                    if await _PENDING.request_flush(connection, scope, binding) is None:
                        continue
                await _INTENTS.request(connection, scope, binding)
                fired.append(binding)
        if self.notifier is not None:
            for binding in fired:
                self.notifier(binding)

    @staticmethod
    async def _qualify(
        connection: AsyncConnection,
        scope: str,
        binding: str,
        high: int,
        threshold: int,
    ) -> bool:
        if high < threshold:
            return False
        cursor = await _CURSORS.load(connection, scope, binding)
        if high - (0 if cursor is None else cursor.cursor.sequence) < threshold:
            return False
        intent = await _INTENTS.load(connection, scope, binding, for_update=True)
        return intent is None or intent.requested_generation <= intent.handled_generation


async def maybe_note_source_count(
    database: AsyncDatabase,
    config: BuiltinConfig,
    family: str,
    scope_id: str,
) -> None:
    """Re-check the Family's threshold after one successful Worker window.

    Memory re-arms only while the remaining count still reaches its threshold,
    so one acceptance never drains a backlog the count would not have started.
    A re-check failure can never fail the completed window it follows.
    """

    threshold = (
        config.runtime.memory_source_count_threshold
        if family == "memory"
        else config.runtime.topic_memory_source_count_threshold
        if family == "topic-memory"
        else None
    )
    if threshold is None:
        return
    trigger = SourceCountTrigger(
        database=database,
        memory_threshold=threshold if family == "memory" else None,
        topic_threshold=threshold if family == "topic-memory" else None,
        notifier=None,
    )
    try:
        await trigger.note(scope_id)
    except Exception as error:  # noqa: BLE001 - a completed window must survive the re-check
        _log_recheck_failure(family, scope_id, error)


def _log_recheck_failure(family: str, scope_id: str, error: BaseException) -> None:
    log_safely(
        logger,
        logging.WARNING,
        "Source count re-check failed after a completed processing window",
        exc_info=error,
        extra={
            "event": "source_count_trigger.recheck_failed",
            "family": family,
            "scope": scope_id,
            "outcome": "retry_on_next_event",
        },
    )
