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

"""Deterministic context packs and prefix deltas that rebuild prepared content exactly."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass

PACK_SCHEMA = "powercontext.context-pack.v1"
PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContextPackError(ValueError):
    """Protocol, chunking, or digest validation failure for context packs and deltas."""


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": [_b64(item) for item in items]}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_b64(value: object) -> bytes:
    if not isinstance(value, str):
        raise ContextPackError("chunk entries must be Base64 strings")  # noqa: TRY003
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ContextPackError("chunk entries must be valid Base64") from error
    if _b64(decoded) != value:
        raise ContextPackError("chunk entries must use canonical Base64")
    return decoded


def _require_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.match(value):
        raise ContextPackError(f"{field} must be a lowercase sha256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete prepared context expressed as deterministic, digest-bound chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> "ContextPack":
        """Construct a pack over already chunked bytes."""

        if not isinstance(items, tuple) or not all(isinstance(item, bytes) for item in items):
            raise ContextPackError("pack items must be a tuple of bytes")
        digest = _pack_digest(items)
        payload_base = {"schema": PACK_SCHEMA, "items": [_b64(item) for item in items], "digest": digest}
        byte_count = 0
        for _ in range(16):
            candidate = len(_canonical({**payload_base, "byte_count": byte_count}))
            if candidate == byte_count:
                break
            byte_count = candidate
        else:
            raise ContextPackError("pack byte count did not converge")
        return cls(items=items, digest=digest, byte_count=byte_count)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> "ContextPack":
        """Split prepared text into bounded, valid-UTF-8 chunks deterministically."""

        if not isinstance(content, str):
            raise ContextPackError("content must be a string")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if not content:
            return cls.build(())
        items: list[bytes] = []
        current = bytearray()
        current_size = 0
        for character in content:
            encoded = character.encode("utf-8")
            if current_size + len(encoded) > chunk_bytes:
                if current_size == 0:
                    raise ContextPackError("a single character exceeds the chunk byte limit")
                items.append(bytes(current))
                current = bytearray()
                current_size = 0
            current += encoded
            current_size += len(encoded)
        if current_size:
            items.append(bytes(current))
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        """Serialize exactly as the byte_count committed to."""

        payload = {
            "schema": PACK_SCHEMA,
            "items": [_b64(item) for item in self.items],
            "digest": self.digest,
            "byte_count": self.byte_count,
        }
        return _canonical(payload)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ContextPack":
        """Validate and decode a complete pack, including its digest and byte count."""

        if not isinstance(raw, (bytes, bytearray)):
            raise ContextPackError("pack wire input must be bytes")
        try:
            payload = json.loads(bytes(raw))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContextPackError("pack wire input must be canonical JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"schema", "items", "digest", "byte_count"}:
            raise ContextPackError("pack wire payload fields are invalid")
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("pack schema is unknown")
        if not isinstance(payload["items"], list):
            raise ContextPackError("pack items must be a list")
        items = tuple(_decode_b64(value) for value in payload["items"])
        digest = _require_digest(payload["digest"], "digest")
        byte_count = payload["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count != len(bytes(raw)):
            raise ContextPackError("pack byte_count does not match the encoded pack")
        if digest != _pack_digest(items):
            raise ContextPackError("pack digest does not match the pack items")
        return cls(items=items, digest=digest, byte_count=byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A prefix-reuse instruction from one identified base pack to one identified target."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> "ContextPackDelta":
        """Reuse the longest common item prefix and carry the remaining target items."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be context packs")
        prefix = 0
        while (
            prefix < len(base.items)
            and prefix < len(target.items)
            and base.items[prefix] == target.items[prefix]
        ):
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        """Wire size of the serialized delta, never a JSON field itself."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        payload = {
            "schema": PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_b64(item) for item in self.appended],
        }
        return _canonical(payload)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ContextPackDelta":
        """Decode and structurally validate a delta wire payload."""

        if not isinstance(raw, (bytes, bytearray)):
            raise ContextPackError("delta wire input must be bytes")
        try:
            payload = json.loads(bytes(raw))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContextPackError("delta wire input must be canonical JSON") from error
        fields = {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ContextPackError("delta wire payload fields are invalid")
        if payload["schema"] != PACK_DELTA_SCHEMA:
            raise ContextPackError("delta schema is unknown")
        base_digest = _require_digest(payload["base_digest"], "base_digest")
        target_digest = _require_digest(payload["target_digest"], "target_digest")
        prefix_items = payload["prefix_items"]
        if isinstance(prefix_items, bool) or not isinstance(prefix_items, int) or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        if not isinstance(payload["appended"], list):
            raise ContextPackError("delta appended must be a list")
        appended = tuple(_decode_b64(value) for value in payload["appended"])
        return cls(base_digest, target_digest, prefix_items, appended)


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from the base and one validated delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("apply requires a pack and a delta")
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the supplied pack")
    if (
        isinstance(delta.prefix_items, bool)
        or not isinstance(delta.prefix_items, int)
        or delta.prefix_items < 0
        or delta.prefix_items > len(base.items)
    ):
        raise ContextPackError("delta prefix_items is out of bounds")
    rebuilt = base.items[: delta.prefix_items] + tuple(delta.appended)
    pack = ContextPack.build(rebuilt)
    if pack.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the reconstructed pack")
    return pack


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
