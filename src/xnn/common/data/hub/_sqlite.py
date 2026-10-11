"""A read-only reader of SQLite table files in plain Python, so no compiled ``sqlite3`` is needed.

Covers what reading a table takes (table b-trees, overflow pages, records, UTF-8 text).
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

_MAGIC = b"SQLite format 3\x00"
_LEAF_TABLE, _INTERIOR_TABLE = 13, 5


def _varint(buf: bytes, i: int) -> Tuple[int, int]:
    """The variable-length integer at ``buf[i]`` and the offset after it."""
    value = 0
    for k in range(8):
        byte = buf[i + k]
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, i + k + 1
    return (value << 8) | buf[i + 8], i + 9


def _value(serial: int, body: bytes, i: int):
    """Decode one record value of serial type ``serial`` at ``body[i]``."""
    if serial == 0:
        return None, i
    if 1 <= serial <= 6:
        size = (1, 2, 3, 4, 6, 8)[serial - 1]
        return int.from_bytes(body[i:i + size], "big", signed=True), i + size
    if serial == 7:
        return struct.unpack(">d", body[i:i + 8])[0], i + 8
    if serial in (8, 9):
        return serial - 8, i
    if serial >= 12:
        size = (serial - 12) // 2
        raw = bytes(body[i:i + size])
        return (raw if serial % 2 == 0 else raw.decode("utf-8")), i + size
    raise ValueError(f"reserved SQLite serial type {serial}")


def _record(payload: bytes) -> list:
    """The column values of one record."""
    header_size, i = _varint(payload, 0)
    serials = []
    while i < header_size:
        serial, i = _varint(payload, i)
        serials.append(serial)
    values, j = [], header_size
    for serial in serials:
        value, j = _value(serial, payload, j)
        values.append(value)
    return values


class _Database:
    """The pages of one database file."""

    def __init__(self, data: bytes):
        if data[:16] != _MAGIC:
            raise ValueError("not an SQLite 3 database file")
        page_size = int.from_bytes(data[16:18], "big")
        self.page_size = 65536 if page_size == 1 else page_size
        self.usable = self.page_size - data[20]
        if data[18] == 2 or data[19] == 2:
            raise ValueError("the database is in write-ahead-log mode; checkpoint it first")
        encoding = int.from_bytes(data[56:60], "big")
        if encoding not in (0, 1):
            raise ValueError(f"text encoding {encoding} is not supported (only UTF-8)")
        self.data = data

    def page(self, number: int) -> Tuple[bytes, int]:
        """Page ``number`` (1-based) and the offset of its b-tree header."""
        start = (number - 1) * self.page_size
        page = self.data[start:start + self.page_size]
        if len(page) != self.page_size:
            raise ValueError(f"page {number} lies beyond the end of the file")
        return page, (100 if number == 1 else 0)

    def payload(self, page: bytes, i: int, size: int) -> bytes:
        """The full payload of a table-leaf cell whose local part starts at ``page[i]``."""
        u = self.usable
        max_local = u - 35
        if size <= max_local:
            return page[i:i + size]
        min_local = (u - 12) * 32 // 255 - 23
        local = min_local + (size - min_local) % (u - 4)
        if local > max_local:
            local = min_local
        parts = [page[i:i + local]]
        remaining = size - local
        nxt = int.from_bytes(page[i + local:i + local + 4], "big")
        while remaining > 0:
            if nxt == 0:
                raise ValueError("truncated overflow chain")
            overflow, _ = self.page(nxt)
            chunk = overflow[4:4 + min(remaining, u - 4)]
            parts.append(chunk)
            remaining -= len(chunk)
            nxt = int.from_bytes(overflow[:4], "big")
        return b"".join(parts)

    def rows(self, root: int) -> Iterator[Tuple[int, list]]:
        """``(rowid, values)`` of the table b-tree rooted at page ``root``, in rowid order."""
        stack: List[int] = [root]
        visited = set()
        while stack:
            number = stack.pop()
            if number in visited:
                raise ValueError(f"page {number} is referenced twice (corrupt b-tree)")
            visited.add(number)
            page, h = self.page(number)
            kind = page[h]
            n_cells = int.from_bytes(page[h + 3:h + 5], "big")
            header = 12 if kind == _INTERIOR_TABLE else 8
            pointers = [int.from_bytes(page[h + header + 2 * k:h + header + 2 * k + 2], "big")
                        for k in range(n_cells)]
            if kind == _INTERIOR_TABLE:
                children = [int.from_bytes(page[p:p + 4], "big") for p in pointers]
                children.append(int.from_bytes(page[h + 8:h + 12], "big"))
                stack.extend(reversed(children))       # leftmost child is visited first
            elif kind == _LEAF_TABLE:
                for p in pointers:
                    size, i = _varint(page, p)
                    rowid, i = _varint(page, i)
                    if rowid >= 1 << 63:
                        rowid -= 1 << 64
                    yield rowid, _record(self.payload(page, i, size))
            else:
                raise ValueError(f"page {number} is not a table b-tree page (type {kind})")


def _columns(sql: str) -> Tuple[List[str], Optional[int]]:
    """Column names of a ``CREATE TABLE`` statement and the index of its rowid alias."""
    body = sql[sql.index("(") + 1:sql.rindex(")")]
    parts, depth, start = [], 0, 0
    for k, ch in enumerate(body):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(body[start:k])
            start = k + 1
    parts.append(body[start:])
    names, alias = [], None
    constraint = re.compile(r"^(constraint|primary|unique|check|foreign)\b", re.IGNORECASE)
    for part in (p.strip() for p in parts):
        if not part or constraint.match(part):
            continue
        tokens = part.split()
        name = tokens[0].strip('"`[]')
        decl = " ".join(tokens[1:]).upper()
        if decl.startswith("INTEGER") and "PRIMARY KEY" in decl and "DESC" not in decl:
            alias = len(names)
        names.append(name)
    return names, alias


def read_table(path: Union[str, Path], table: str) -> Tuple[List[str], List[tuple]]:
    """All rows of one table of an SQLite database file, in rowid order.

    Parameters
    ----------
    path : str or Path
        The database file.
    table : str
        The table name.

    Returns
    -------
    tuple of (list of str, list of tuple)
        The column names and one tuple per row. Integers are ``int``, reals
        ``float``, text ``str``, blobs ``bytes`` and NULL ``None``; an
        ``INTEGER PRIMARY KEY`` column holds the rowid.

    Raises
    ------
    ValueError
        If the file is not an SQLite database, uses a feature this reader
        does not cover, or has no table ``table``.
    """
    db = _Database(Path(path).read_bytes())
    schema = {}
    for _, (kind, name, _tbl, root, sql) in db.rows(1):
        if kind == "table":
            schema[name] = (root, sql)
    if table not in schema:
        raise ValueError(f"no table {table!r} in {path} (tables: {sorted(schema)})")
    root, sql = schema[table]
    names, alias = _columns(sql)
    rows = []
    for rowid, values in db.rows(root):
        values = values + [None] * (len(names) - len(values))   # columns added later
        if alias is not None:
            values[alias] = rowid
        rows.append(tuple(values[:len(names)]))
    return names, rows
