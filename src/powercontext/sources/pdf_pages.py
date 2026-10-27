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

"""Offline PDF capture: every physical page becomes one durable Source with exact page citations."""

from __future__ import annotations

import hashlib
import io
import math
import re
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_NAME = "pdf-page"
PDF_PAGE_VERSION = "1"

_TOKEN_RE = re.compile(r"[a-z]+|\d+")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

_STOPWORDS = frozenset(
    """
    a an about above after again against all also am and any are as at be because been before being
    below between both but by can cannot could did do does doing don down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it its
    itself just me more most my myself no nor not now of off on once only or other our ours ourselves
    out over own same she should so some such than that the their theirs them themselves then thence
    there these they this those through to too toward under until unto up upon us use used using very
    was we well were what when where whether which while who whom why will with within without would
    yet you your yours yourself yourselves define answer answers assuming assume respond response
    perspective analyst investment details shown within statement statements give calculated calculate
    calculation give based millions usd year fiscal quarter report reported reports
    """.split()
)


class PDFDocumentInput(BaseModel):
    """One complete offline PDF addressed by a stable document id."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page with its document identity, ordinal, and extractable text."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Resolved page Source retaining the complete extraction for re-reads."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the delimiter-safe identity decided by (document id, sha256, physical page)."""

    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty without leading or trailing whitespace")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive integer counting from one")
    source_id = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of the complete PDF, or fail atomically with ``ValueError``."""

    document_id = value.document_id
    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty or only whitespace")
    content = value.content
    if not content:
        raise ValueError("PDF content must not be empty")
    document_sha256 = hashlib.sha256(content).hexdigest()

    pages: list[PDFPageInput] = []
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise ValueError("PDF requires a non-empty user password")
        total = len(reader.pages)
        if total == 0:
            raise ValueError("PDF contains no pages")
        for index in range(total):
            extracted = reader.pages[index].extract_text()
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=index + 1,
                    text=extracted if extracted else "",
                )
            )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"PDF could not be parsed completely: {error}") from error
    return tuple(pages)


class PDFPageTextEvidenceProjection:
    """Project one captured page into the standard ``powercontext.text-evidence`` shape."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class PdfPageDefinition:
    """Bind the PDF page adapter contract to version ``1`` and its text evidence projection."""

    input_class = PDFPageInput
    name = PDF_PAGE_NAME
    source_class = PDFPageSource
    version = PDF_PAGE_VERSION
    projections = (PDFPageTextEvidenceProjection(),)

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        if not _SHA_RE.match(value.document_sha256):
            raise ValueError("document_sha256 must be a lowercase 64-digit hex digest")
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            definition_version=PDF_PAGE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = PdfPageDefinition()


def _query_tokens(question: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(question.casefold()):
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _citation_key(hit) -> tuple[str, int]:
    return (hit.entry_id, hit.entry_version_id)


async def _collect(memory, query: str, limit: int, fused: dict, refs: dict, weight: float) -> None:
    try:
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=limit))
    except Exception:
        return
    for rank, hit in enumerate(page.hits):
        if hit.memory_ref is None:
            continue
        key = _citation_key(hit)
        refs.setdefault(key, hit.memory_ref)
        fused[key] = fused.get(key, 0.0) + weight / (60.0 + rank + 1.0)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Retrieve up to ``limit`` distinct page citations, fused and reranked fully offline."""

    tokens = _query_tokens(question)
    if not tokens or limit < 1:
        return ()

    fused: dict[tuple, float] = {}
    refs: dict[tuple, object] = {}
    await _collect(memory, question, 60, fused, refs, 2.0)

    single_terms = tokens[:16]
    pairs = [f"{tokens[i]} {tokens[i + 1]}" for i in range(min(len(tokens) - 1, 7))]

    for token in single_terms:
        await _collect(memory, token, 24, fused, refs, 1.0)
    for pair in pairs:
        await _collect(memory, pair, 30, fused, refs, 1.2)

    if not fused:
        return ()

    # Rerank the fused pool with an offline BM25 over the retrieved entry texts.
    ranked_keys = sorted(fused, key=fused.get, reverse=True)[:44]
    documents: list[tuple[tuple, MemoryCitation, dict[str, int], int]] = []
    entry_sources: dict[tuple, tuple] = {}
    for key in ranked_keys:
        citation = MemoryCitation(
            memory_ref=refs[key], entry_id=key[0], entry_version_id=key[1]
        )
        try:
            entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        except Exception:
            continue
        if not entry.sources:
            continue
        entry_sources[key] = entry.sources
        counts: dict[str, int] = {}
        for token in _TOKEN_RE.findall(entry.text.casefold()):
            counts[token] = counts.get(token, 0) + 1
        documents.append((key, citation, counts, sum(counts.values()) or 1))

    if not documents:
        return ()

    k1, b = 1.4, 0.75
    total_docs = len(documents)
    avg_length = sum(doc[3] for doc in documents) / total_docs
    idf = {
        token: math.log(1.0 + (total_docs - sum(1 for doc in documents if token in doc[2])) + 0.5)
        / (sum(1 for doc in documents if token in doc[2]) + 0.5)
        for token in tokens
    }
    scored: list[tuple[float, MemoryCitation, tuple[str, ...]]] = []
    for key, citation, counts, length in documents:
        lexical = 0.0
        for token in tokens:
            tf = counts.get(token, 0)
            if tf:
                lexical += idf[token] * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * length / avg_length))
        coverage = sum(1 for token in tokens if counts.get(token)) / len(tokens)
        page_key = tuple(sorted(reference.source_id for reference in entry_sources[key]))
        scored.append((0.72 * lexical + 2.0 * coverage + fused[key], citation, page_key))

    scored.sort(key=lambda item: item[0], reverse=True)
    seen_pages: set = set()
    citations: list[MemoryCitation] = []
    for _, citation, page_key in scored:
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        citations.append(citation)
        if len(citations) == limit:
            break
    return tuple(citations)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
