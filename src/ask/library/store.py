"""sqlite-vec store: files, chunks, and their vectors in one sqlite file.

A single sqlite file holds files + chunks + vectors together: brute-force exact
KNN over a vec0 virtual table, incremental upsert, metadata filtering, and orphan
sweep. Exact KNN is fast at library scale (hundreds of thousands of passages) and
recalls better than an ANN index. Atomicity and self-healing come natively from
sqlite transactions and CREATE ... IF NOT EXISTS, so a crash mid-write leaves a
consistent store.

Schema:
  files(path PRIMARY KEY, hash, mtime, pages, status, indexed_at)
  chunks(id INTEGER PRIMARY KEY, path, page, bbox_json, text)
  vec_chunks USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[384])
"""

import json
import sqlite3
from pathlib import Path

import sqlite_vec

from .embedder import EMBED_DIM

STATUSES = ("ok", "ocr", "no_text", "error")


def _connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class Store:
    """A handle on one library.sqlite3 file. Create the schema on construction."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = _connect(self.db_path)
        self._init_schema()

    def close(self):
        # Fold the WAL back into the main file so data/<corpus>.sqlite3 is a single
        # self-contained artifact - safe to copy between machines (e.g. build the
        # index on a high-RAM box, copy it to a small one to query).
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _init_schema(self):
        c = self.conn
        c.execute(
            "CREATE TABLE IF NOT EXISTS files("
            "path TEXT PRIMARY KEY, hash TEXT, mtime REAL, pages INTEGER, "
            "status TEXT, indexed_at TEXT)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS chunks("
            "id INTEGER PRIMARY KEY, path TEXT, page INTEGER, bbox_json TEXT, text TEXT)"
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
        # distance_metric=cosine: rank by cosine distance regardless of whether the
        # embedder returns normalized vectors, giving brute-force exact cosine KNN.
        c.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{EMBED_DIM}] distance_metric=cosine)"
        )
        c.commit()

    # -- diff inputs -----------------------------------------------------------

    def file_index(self):
        """{path: {"hash", "status"}} for every known file (drives diff + sweep)."""
        rows = self.conn.execute("SELECT path, hash, status FROM files").fetchall()
        return {r["path"]: {"hash": r["hash"], "status": r["status"]} for r in rows}

    def get_file(self, path):
        return self.conn.execute(
            "SELECT * FROM files WHERE path=?", (str(path),)
        ).fetchone()

    # -- mutations -------------------------------------------------------------

    def _delete_chunks(self, path):
        p = str(path)
        self.conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE path=?)",
            (p,),
        )
        self.conn.execute("DELETE FROM chunks WHERE path=?", (p,))

    def delete_file(self, path):
        """Orphan sweep: drop a file's chunks, vectors, and its files row."""
        p = str(path)
        self._delete_chunks(p)
        self.conn.execute("DELETE FROM files WHERE path=?", (p,))
        self.conn.commit()

    def upsert_file(self, path, file_hash, mtime, pages, status, indexed_at):
        """Record a files row with no chunks (no_text / error cases)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO files(path,hash,mtime,pages,status,indexed_at) "
            "VALUES (?,?,?,?,?,?)",
            (str(path), file_hash, mtime, pages, status, indexed_at),
        )
        self.conn.commit()

    def replace_file(self, path, file_hash, mtime, pages, status, indexed_at, chunks, vectors):
        """Atomically swap a file's chunks + vectors and update its files row.

        chunks is [{page, bbox, text}]; vectors is a parallel list of float lists.
        Old chunks/vectors for this path are removed first, so a re-index of a
        changed file leaves no orphans. All-or-nothing: a failure rolls back.
        """
        p = str(path)
        try:
            self._delete_chunks(p)
            self.conn.execute(
                "INSERT OR REPLACE INTO files(path,hash,mtime,pages,status,indexed_at) "
                "VALUES (?,?,?,?,?,?)",
                (p, file_hash, mtime, pages, status, indexed_at),
            )
            for chunk, vector in zip(chunks, vectors):
                cur = self.conn.execute(
                    "INSERT INTO chunks(path,page,bbox_json,text) VALUES (?,?,?,?)",
                    (p, chunk.get("page"), json.dumps(chunk.get("bbox")), chunk.get("text", "")),
                )
                self.conn.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(vector)),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- reads -----------------------------------------------------------------

    def counts(self):
        c = self.conn
        files = c.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vectors = c.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        by_status = {
            r["status"]: r["n"]
            for r in c.execute(
                "SELECT status, COUNT(*) AS n FROM files GROUP BY status"
            ).fetchall()
        }
        last = c.execute("SELECT MAX(indexed_at) FROM files").fetchone()[0]
        return {
            "files": files,
            "chunks": chunks,
            "vectors": vectors,
            "by_status": by_status,
            "last_indexed": last,
        }

    def search(self, query_vector, k=8):
        """Brute-force exact KNN -> ranked chunk rows joined to their file.

        Returns [{chunk_id, path, page, bbox_json, text, distance}] ordered by
        ascending cosine distance (closest first).
        """
        rows = self.conn.execute(
            "SELECT c.id AS chunk_id, c.path, c.page, c.bbox_json, c.text, v.distance "
            "FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (sqlite_vec.serialize_float32(query_vector), k),
        ).fetchall()
        return [dict(r) for r in rows]
