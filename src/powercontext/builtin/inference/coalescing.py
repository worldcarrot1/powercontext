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

"""Cross-call embedding coalescing with a bounded LRU vector cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True)
class _Batch:
    texts: tuple[str, ...]
    task: asyncio.Task[None] | None = None
    vectors: dict[str, tuple[float, ...]] = field(default_factory=dict)
    usage: InferenceUsage | None = None
    error: BaseException | None = None


class CoalescingEmbeddingModel:
    """Share in-flight embedding work across calls and cache successful vectors."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._base = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._claimed: dict[str, _Batch] = {}
        self.profile = model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return ordered vectors, joining shared batches and cached vectors first."""

        unique: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique.append(text)
        if not unique:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        hits: dict[str, tuple[float, ...]] = {}
        missing: list[str] = []
        for text in unique:
            if self._cache_size and text in self._cache:
                self._cache.move_to_end(text)
                hits[text] = self._cache[text]
            else:
                missing.append(text)

        needed: list[_Batch] = []
        needed_seen: set[int] = set()
        unclaimed: list[str] = []

        def _register(batch: _Batch) -> None:
            if id(batch) not in needed_seen:
                needed_seen.add(id(batch))
                needed.append(batch)

        for text in missing:
            batch = self._claimed.get(text)
            if batch is None:
                unclaimed.append(text)
                continue
            if batch.error is not None or batch.task is None:
                continue
            _register(batch)

        if unclaimed:
            for start in range(0, len(unclaimed), self._max_batch_size):
                chunk = tuple(unclaimed[start : start + self._max_batch_size])
                batch = _Batch(texts=chunk)
                for text in chunk:
                    self._claimed[text] = batch
                batch.task = asyncio.create_task(self._run(batch))
                _register(batch)

        by_text: dict[str, _Batch] = {}
        for batch in needed:
            for text in batch.texts:
                by_text[text] = batch

        for batch in needed:
            if batch.task is not None:
                await asyncio.shield(batch.task)
        for batch in needed:
            if batch.error is not None:
                raise batch.error

        vectors: list[tuple[float, ...]] = []
        for text in texts:
            if text in hits:
                vectors.append(hits[text])
                continue
            batch = by_text.get(text)
            if batch is not None and text in batch.vectors:
                vectors.append(batch.vectors[text])
                continue
            if self._cache_size and text in self._cache:
                vectors.append(self._cache[text])
                continue
            raise ValueError("embedding batch lost its result before delivery")

        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        for batch in needed:
            usage = batch.usage
            if usage is None:
                continue
            requests += usage.requests
            if usage.input_tokens is not None:
                input_tokens = (input_tokens or 0) + usage.input_tokens
            if usage.output_tokens is not None:
                output_tokens = (output_tokens or 0) + usage.output_tokens
        return EmbeddingResult(
            vectors=tuple(vectors),
            usage=InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens),
        )

    async def _run(self, batch: _Batch) -> None:
        try:
            if self._coalesce_seconds > 0.0:
                await asyncio.sleep(self._coalesce_seconds)
            result = await self._base.embed(batch.texts)
            if len(result.vectors) != len(batch.texts):
                raise ValueError("embedding model returned a vector count that does not match the batch size")
        except asyncio.CancelledError:
            self._release(batch)
            batch.error = RuntimeError("embedding batch was cancelled")
            raise
        except Exception as error:
            self._release(batch)
            batch.error = error
            return
        batch.vectors = dict(zip(batch.texts, result.vectors, strict=True))
        batch.usage = result.usage
        if self._cache_size:
            for text, vector in batch.vectors.items():
                self._cache[text] = vector
                self._cache.move_to_end(text)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        self._release(batch)

    def _release(self, batch: _Batch) -> None:
        for text in batch.texts:
            if self._claimed.get(text) is batch:
                del self._claimed[text]


__all__ = ["CoalescingEmbeddingModel"]
