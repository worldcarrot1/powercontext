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

"""HTTP cross-encoder scoring service adapter for Memory reranking."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score the coarse candidate pool at an HTTP endpoint in bounded batches."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Send one request per bounded batch and select the global score order."""

        candidate_count = len(candidates)
        if candidate_count < 1:
            raise ValueError("rerank requires at least one candidate")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > candidate_count:
            raise ValueError("rerank limit must be between one and the candidate count")

        scored: list[tuple[float, int]] = []
        requests = 0
        for start in range(0, candidate_count, self._batch_size):
            batch = candidates[start : start + self._batch_size]
            batch_ranks = list(range(start + 1, start + 1 + len(batch)))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in zip(batch_ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            response.raise_for_status()
            scored.extend(self._parse_scores(response, batch_ranks))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected_ranks = tuple(rank for _, rank in scored[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected_ranks,
            usage=InferenceUsage(requests=requests),
            discarded_rank_count=candidate_count - limit,
        )

    @staticmethod
    def _parse_scores(response: httpx.Response, batch_ranks: list[int]) -> list[tuple[float, int]]:
        try:
            body = response.json()
        except ValueError as error:
            raise ValueError("rerank response body must be valid JSON") from error
        if not isinstance(body, dict) or set(body) != {"scores"}:
            raise ValueError("rerank response must contain exactly one 'scores' field")
        scores = body["scores"]
        if not isinstance(scores, list):
            raise ValueError("rerank 'scores' must be a list")

        expected = set(batch_ranks)
        observed: dict[int, float] = {}
        for record in scores:
            if not isinstance(record, dict) or set(record) != {"rank", "score"}:
                raise ValueError("each rerank score must contain exactly 'rank' and 'score'")
            rank = record["rank"]
            score = record["score"]
            if isinstance(rank, bool) or not isinstance(rank, int) or rank not in expected:
                raise ValueError("rerank score rank is outside this batch")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("rerank score must be a finite number")
            if rank in observed:
                raise ValueError("rerank score ranks must be unique within a batch")
            observed[rank] = float(score)
        if set(observed) != expected:
            raise ValueError("rerank scores must cover every batch rank exactly once")
        return [(value, rank) for rank, value in observed.items()]


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
