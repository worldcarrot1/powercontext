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

"""Offline CSV capture: one durable, per-row Source with exact evidence projection."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_NAME = "csv-row"
CSV_ROW_VERSION = "1"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document captured under a stable identity."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One data row of an offline CSV document, addressed by its primary key value."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Resolved single-row Source carrying the full original record for re-reads."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate a whole CSV document, or fail atomically with ``ValueError``."""

    document_id = value.document_id
    primary_key = value.primary_key
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty without leading or trailing whitespace")
    if not primary_key:
        raise ValueError("primary_key must not be empty")
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content is not valid UTF-8") from error
    if not text:
        raise ValueError("CSV document must not be empty")

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"CSV document is malformed: {error}") from error

    header, *data_records = records
    if not header:
        raise ValueError("CSV header must not be empty")
    for field in header:
        if not field or not field.strip():
            raise ValueError("header fields must be non-empty and not only whitespace")
    if len(set(header)) != len(header):
        raise ValueError("header fields must be unique")
    if primary_key not in header:
        raise ValueError(f"primary_key {primary_key!r} does not match any header field exactly")
    key_index = header.index(primary_key)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in data_records:
        if not record:
            continue
        if len(record) != len(header):
            raise ValueError("data records must contain exactly one value per header column")
        primary_key_value = record[key_index]
        if not primary_key_value or not primary_key_value.strip():
            raise ValueError("primary key values must not be empty or only whitespace")
        if primary_key_value in seen_keys:
            raise ValueError(f"duplicate primary key value {primary_key_value!r}")
        seen_keys.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=tuple(header),
                values=tuple(record),
            )
        )
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the delimiter-safe identity of one row across its document."""

    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty without leading or trailing whitespace")
    if not primary_key_value or not primary_key_value.strip():
        raise ValueError("primary key values must not be empty or only whitespace")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


class CSVRowTextEvidenceProjection:
    """Project one captured row into the standard ``powercontext.text-evidence`` shape."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values))
        return TextEvidence(
            source_type=CSV_ROW_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


class CsvRowDefinition:
    """Bind the CSV row adapter contract to version ``1`` and its text evidence projection."""

    input_class = CSVRowInput
    name = CSV_ROW_NAME
    source_class = CSVRowSource
    version = CSV_ROW_VERSION
    projections = (CSVRowTextEvidenceProjection(),)

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        if not value.document_id or value.document_id != value.document_id.strip():
            raise ValueError("document_id must be non-empty without leading or trailing whitespace")
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        if value.columns and len(value.values) != len(value.columns):
            raise ValueError("row values must align with the header columns")
        return CSVRowSource(
            name=source_id,
            definition_version=CSV_ROW_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = CsvRowDefinition()

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
