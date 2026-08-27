from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .domain import Asset, ObjectDetection, OcrLine, RegionEmbedding, SearchHit
from .interfaces import AssetRepository
from .text import (
    compact_identifier,
    fts_query,
    identifier_tokens,
    identifier_trigrams,
    metadata_filters,
    normalize_text,
    object_query_labels,
    search_document,
)


SCHEMA_VERSION = 5


class SQLiteAssetRepository(AssetRepository):
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        self._migrate()
        self.recover_interrupted_stages()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def _migrate(self) -> None:
        db = self._connection()
        db.executescript("""
        CREATE TABLE IF NOT EXISTS app_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS roots(path TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS assets(
          id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, size INTEGER NOT NULL,
          modified_ns INTEGER NOT NULL, sha256 TEXT NOT NULL, width INTEGER NOT NULL,
          height INTEGER NOT NULL, thumbnail TEXT NOT NULL, ocr_text TEXT NOT NULL DEFAULT '',
          ocr_json TEXT NOT NULL DEFAULT '[]', ocr_raw_text TEXT NOT NULL DEFAULT '',
          ocr_diagnostics TEXT NOT NULL DEFAULT '{}',
          search_text TEXT NOT NULL DEFAULT '', model_version TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'ready', error TEXT NOT NULL DEFAULT '',
          indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS asset_fts USING fts5(asset_id UNINDEXED, search_text);
        """)
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(assets)")}
        if "ocr_raw_text" not in columns:
            db.execute("ALTER TABLE assets ADD COLUMN ocr_raw_text TEXT NOT NULL DEFAULT ''")
        if "ocr_diagnostics" not in columns:
            db.execute("ALTER TABLE assets ADD COLUMN ocr_diagnostics TEXT NOT NULL DEFAULT '{}'")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS ocr_lines(
          id INTEGER PRIMARY KEY,
          asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          line_index INTEGER NOT NULL,
          text TEXT NOT NULL,
          normalized_text TEXT NOT NULL,
          confidence REAL NOT NULL,
          detection_confidence REAL NOT NULL DEFAULT 0,
          source_pass TEXT NOT NULL DEFAULT 'full',
          search_terms TEXT NOT NULL DEFAULT '',
          box_json TEXT NOT NULL DEFAULT '[]',
          searchable INTEGER NOT NULL DEFAULT 0,
          UNIQUE(asset_id, line_index)
        );
        CREATE INDEX IF NOT EXISTS idx_ocr_lines_asset ON ocr_lines(asset_id);
        CREATE TABLE IF NOT EXISTS pipeline_stages(
          asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          stage TEXT NOT NULL,
          status TEXT NOT NULL,
          model_version TEXT NOT NULL,
          input_sha256 TEXT NOT NULL,
          error TEXT NOT NULL DEFAULT '',
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY(asset_id, stage)
        );
        CREATE TABLE IF NOT EXISTS asset_embeddings(
          asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
          model_version TEXT NOT NULL,
          dimensions INTEGER NOT NULL,
          vector BLOB NOT NULL,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS region_embeddings(
          id INTEGER PRIMARY KEY,
          asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          region_index INTEGER NOT NULL,
          box_json TEXT NOT NULL,
          model_version TEXT NOT NULL,
          dimensions INTEGER NOT NULL,
          vector BLOB NOT NULL,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(asset_id, region_index)
        );
        CREATE INDEX IF NOT EXISTS idx_region_asset ON region_embeddings(asset_id);
        CREATE INDEX IF NOT EXISTS idx_region_version ON region_embeddings(model_version, dimensions);
        CREATE TABLE IF NOT EXISTS object_detections(
          id INTEGER PRIMARY KEY,
          asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          label TEXT NOT NULL,
          label_zh TEXT NOT NULL DEFAULT '',
          confidence REAL NOT NULL,
          box_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_object_asset ON object_detections(asset_id);
        CREATE INDEX IF NOT EXISTS idx_object_label ON object_detections(label, confidence DESC);
        """)
        line_columns = {str(row[1]) for row in db.execute("PRAGMA table_info(ocr_lines)")}
        if "detection_confidence" not in line_columns:
            db.execute("ALTER TABLE ocr_lines ADD COLUMN detection_confidence REAL NOT NULL DEFAULT 0")
        if "source_pass" not in line_columns:
            db.execute("ALTER TABLE ocr_lines ADD COLUMN source_pass TEXT NOT NULL DEFAULT 'full'")
        if "search_terms" not in line_columns:
            db.execute("ALTER TABLE ocr_lines ADD COLUMN search_terms TEXT NOT NULL DEFAULT ''")
        previous = db.execute("SELECT value FROM app_meta WHERE key='schema_version'").fetchone()
        previous_version = int(previous[0]) if previous else 0
        if previous_version < 3:
            self._backfill_search_and_ocr(db)
            # OCR v3 changes detection resolution and searchable line data.
            # Preserve metadata and SigLIP embeddings; only OCR is queued again.
            db.execute(
                "UPDATE pipeline_stages SET status='pending',error='',updated_at=CURRENT_TIMESTAMP "
                "WHERE stage='ocr'"
            )
        db.execute("INSERT OR REPLACE INTO app_meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        db.commit()

    @staticmethod
    def _backfill_search_and_ocr(db: sqlite3.Connection) -> None:
        db.execute("DELETE FROM asset_fts")
        for row in db.execute("SELECT id,path,ocr_text,ocr_json,ocr_raw_text FROM assets").fetchall():
            raw_text = str(row["ocr_raw_text"] or "")
            try:
                payload = json.loads(row["ocr_json"] or "[]")
            except (TypeError, ValueError):
                payload = []
            if not raw_text:
                raw_text = " ".join(str(line.get("text", "")) for line in payload if line.get("text"))
                db.execute("UPDATE assets SET ocr_raw_text=? WHERE id=?", (raw_text, row["id"]))
            db.execute("DELETE FROM ocr_lines WHERE asset_id=?", (row["id"],))
            for index, line in enumerate(payload):
                text = str(line.get("text", "")).strip()
                if not text:
                    continue
                confidence = float(line.get("confidence", 0.0))
                searchable = bool(line.get("searchable", confidence >= 0.55))
                db.execute(
                    "INSERT INTO ocr_lines(asset_id,line_index,text,normalized_text,confidence,"
                    "detection_confidence,source_pass,search_terms,box_json,searchable) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (row["id"], index, text, normalize_text(text), confidence,
                     float(line.get("detection_confidence", 0.0)), str(line.get("source_pass", "full")),
                     " ".join(line.get("search_terms", [])),
                     json.dumps(line.get("box", []), ensure_ascii=False), int(searchable)),
                )
            document = search_document(f"{Path(row['path']).name} {row['ocr_text']}")
            db.execute("UPDATE assets SET search_text=? WHERE id=?", (document, row["id"]))
            db.execute("INSERT INTO asset_fts(asset_id,search_text) VALUES(?,?)", (row["id"], document))

    @staticmethod
    def _asset(row: sqlite3.Row) -> Asset:
        fields = {key: row[key] for key in Asset.__dataclass_fields__ if key in row.keys()}
        return Asset(**fields)

    def upsert(self, asset: Asset, ocr_lines: list[OcrLine] | None = None) -> int:
        db = self._connection()
        search_text = search_document(f"{Path(asset.path).name} {asset.ocr_text}")
        values = asdict(asset)
        values.pop("id")
        values["search_text"] = search_text
        columns = list(values)
        update = ",".join(f"{column}=excluded.{column}" for column in columns if column != "path")
        db.execute(
            f"INSERT INTO assets({','.join(columns)}) VALUES({','.join(':'+c for c in columns)}) "
            f"ON CONFLICT(path) DO UPDATE SET {update}, indexed_at=CURRENT_TIMESTAMP",
            values,
        )
        row = db.execute("SELECT id FROM assets WHERE path=?", (asset.path,)).fetchone()
        asset_id = int(row[0])
        db.execute("DELETE FROM asset_fts WHERE asset_id=?", (asset_id,))
        db.execute("INSERT INTO asset_fts(asset_id,search_text) VALUES(?,?)", (asset_id, search_text))
        if ocr_lines is not None:
            db.execute("DELETE FROM ocr_lines WHERE asset_id=?", (asset_id,))
            for index, line in enumerate(ocr_lines):
                db.execute(
                    "INSERT INTO ocr_lines(asset_id,line_index,text,normalized_text,confidence,"
                    "detection_confidence,source_pass,search_terms,box_json,searchable) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (asset_id, index, line.text, normalize_text(line.text), float(line.confidence),
                     float(line.detection_confidence), line.source_pass, " ".join(line.search_terms),
                     json.dumps(line.box, ensure_ascii=False), int(line.searchable)),
                )
        db.commit()
        return asset_id

    def get(self, asset_id: int) -> Asset | None:
        row = self._connection().execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        return self._asset(row) if row else None

    def get_by_path(self, path: str) -> Asset | None:
        row = self._connection().execute("SELECT * FROM assets WHERE path=?", (path,)).fetchone()
        return self._asset(row) if row else None

    def get_missing_by_hash(self, sha256: str) -> Asset | None:
        rows = self._connection().execute("SELECT * FROM assets WHERE sha256=? ORDER BY indexed_at DESC", (sha256,)).fetchall()
        return next((self._asset(row) for row in rows if not Path(row["path"]).exists()), None)

    def relocate(self, asset_id: int, path: str, size: int, modified_ns: int) -> Asset:
        db = self._connection()
        db.execute("UPDATE assets SET path=?,size=?,modified_ns=?,indexed_at=CURRENT_TIMESTAMP WHERE id=?",
                   (path, size, modified_ns, asset_id))
        row = db.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        asset = self._asset(row)
        self.upsert(asset)
        return asset

    def text_search(self, query: str, limit: int = 100) -> list[SearchHit]:
        expression = fts_query(query)
        if not expression:
            return []
        rows = self._connection().execute(
            "SELECT a.*, bm25(asset_fts) rank FROM asset_fts JOIN assets a ON a.id=asset_fts.asset_id "
            "WHERE asset_fts MATCH ? AND a.status IN ('ready','partial') ORDER BY rank LIMIT ?",
            (expression, max(limit, 1) * 4),
        ).fetchall()
        candidates: list[tuple[float, float, SearchHit]] = []
        query_normalized = normalize_text(query)
        query_compact = compact_identifier(query)
        query_tokens = identifier_tokens(query)
        query_trigrams = set(identifier_trigrams(query))
        for fts_rank, row in enumerate(rows, 1):
            asset = self._asset(row)
            reasons: list[str] = []
            bonus = 0.0
            filename = normalize_text(Path(asset.path).name)
            if query_normalized and query_normalized in filename:
                bonus += 0.20
                reasons.append("檔名完整命中")
            coverage = 0.0
            for line in self.ocr_lines(int(asset.id)):
                normalized_line = normalize_text(line.text)
                compact_line = compact_identifier(line.text)
                if query_normalized and query_normalized in normalized_line:
                    bonus = max(bonus, 0.18 + (0.20 if query_normalized in filename else 0.0))
                    reasons.append(f"OCR 完整命中：{line.text[:80]}")
                if query_compact and len(query_compact) >= 4 and query_compact in compact_line:
                    bonus += 0.15 if not any(reason.startswith("程式碼識別字匹配") for reason in reasons) else 0.0
                    reasons.append(f"程式碼識別字匹配：{line.text[:80]}")
                if query_tokens and all(token in normalized_line or token in compact_line for token in query_tokens):
                    bonus += 0.08 if "查詢詞位於同一 OCR 行" not in reasons else 0.0
                    reasons.append("查詢詞位於同一 OCR 行")
                if query_trigrams:
                    line_trigrams = set(identifier_trigrams(line.text))
                    coverage = max(coverage, len(query_trigrams & line_trigrams) / len(query_trigrams))
            bonus = min(0.61, bonus)
            text_score = max(coverage, 1.0 / (1.0 + max(0.0, float(row["rank"]))))
            hit = SearchHit(asset, 0.0, text_score=text_score, text_rank=fts_rank,
                            bonus_score=bonus, reasons=list(dict.fromkeys(reasons)))
            candidates.append((bonus, coverage, hit))
        candidates.sort(key=lambda item: (item[0], item[1], -int(item[2].text_rank or 0)), reverse=True)
        result: list[SearchHit] = []
        for rank, (_bonus, _coverage, hit) in enumerate(candidates[:limit], 1):
            hit.text_rank = rank
            result.append(hit)
        return result

    def metadata_search(self, query: str, limit: int = 200) -> list[SearchHit]:
        filters = metadata_filters(query)
        if not filters:
            return []
        assets = [asset for asset in self.all_assets() if asset.status in ("ready", "partial")]
        assets = [asset for asset in assets if asset_matches_metadata(asset, filters)]
        assets.sort(key=lambda asset: asset.modified_ns, reverse=True)
        return [
            SearchHit(asset, 0.0, metadata_rank=rank, reasons=[metadata_reason(filters)])
            for rank, asset in enumerate(assets[:limit], 1)
        ]

    def object_search(self, query: str, limit: int = 200) -> list[SearchHit]:
        labels = object_query_labels(query)
        normalized_query = normalize_text(query)
        compact_query = normalized_query.replace(" ", "")
        # Prompt-Free YOLOE may contain labels that are not in our curated
        # bilingual aliases. Exact label phrases remain searchable without a
        # code release whenever a new fixed vocabulary is published.
        available = self._connection().execute(
            "SELECT DISTINCT label,label_zh FROM object_detections"
        ).fetchall()
        for row in available:
            label = str(row["label"])
            translated = str(row["label_zh"])
            candidates = [normalize_text(label), normalize_text(translated)]
            if any(
                candidate and (candidate in normalized_query or candidate.replace(" ", "") in compact_query)
                for candidate in candidates
            ):
                labels.append(label)
        labels = list(dict.fromkeys(labels))
        if not labels:
            return []
        placeholders = ",".join("?" for _ in labels)
        rows = self._connection().execute(
            f"SELECT a.*,COUNT(DISTINCT o.label) matched,MAX(o.confidence) confidence "
            f"FROM object_detections o JOIN assets a ON a.id=o.asset_id "
            f"WHERE o.label IN ({placeholders}) AND a.status IN ('ready','partial') "
            "GROUP BY a.id ORDER BY matched DESC,confidence DESC LIMIT ?",
            (*labels, max(1, int(limit))),
        ).fetchall()
        return [
            SearchHit(
                self._asset(row), 0.0, object_rank=rank,
                reasons=["物件命中：" + "、".join(labels)],
                metadata={"object_labels": labels, "object_confidence": float(row["confidence"])},
            )
            for rank, row in enumerate(rows, 1)
        ]

    def replace_objects(self, asset_id: int, detections: list[ObjectDetection]) -> None:
        db = self._connection()
        db.execute("DELETE FROM object_detections WHERE asset_id=?", (asset_id,))
        db.executemany(
            "INSERT INTO object_detections(asset_id,label,label_zh,confidence,box_json) VALUES(?,?,?,?,?)",
            [
                (asset_id, item.label, item.label_zh, float(item.confidence),
                 json.dumps([float(value) for value in item.box]))
                for item in detections
            ],
        )
        db.commit()

    def object_detections(self, asset_id: int) -> list[ObjectDetection]:
        rows = self._connection().execute(
            "SELECT label,label_zh,confidence,box_json FROM object_detections "
            "WHERE asset_id=? ORDER BY confidence DESC", (asset_id,),
        ).fetchall()
        return [
            ObjectDetection(str(row["label"]), float(row["confidence"]), json.loads(row["box_json"]), str(row["label_zh"]))
            for row in rows
        ]

    def replace_ocr(self, asset_id: int, lines: list[OcrLine], diagnostics: dict[str, object] | None = None) -> None:
        asset = self.get(asset_id)
        if asset is None:
            return
        asset.ocr_raw_text = " ".join(line.text for line in lines if line.text.strip())
        asset.ocr_text = " ".join(line.text for line in lines if line.text.strip() and line.searchable)
        asset.ocr_json = json.dumps([
            {"text": line.text, "confidence": line.confidence, "box": line.box, "searchable": line.searchable,
             "detection_confidence": line.detection_confidence, "source_pass": line.source_pass,
             "search_terms": line.search_terms}
            for line in lines
        ], ensure_ascii=False)
        asset.ocr_diagnostics = json.dumps(diagnostics or {}, ensure_ascii=False)
        self.upsert(asset, lines)

    def ocr_lines(self, asset_id: int) -> list[OcrLine]:
        rows = self._connection().execute(
            "SELECT text,confidence,detection_confidence,source_pass,search_terms,box_json,searchable "
            "FROM ocr_lines WHERE asset_id=? ORDER BY line_index",
            (asset_id,),
        ).fetchall()
        return [OcrLine(str(row["text"]), float(row["confidence"]), json.loads(row["box_json"]),
                        bool(row["searchable"]), float(row["detection_confidence"]),
                        str(row["source_pass"]), str(row["search_terms"]).split()) for row in rows]

    def set_stage(self, asset_id: int, stage: str, status: str, model_version: str, input_sha256: str, error: str = "") -> None:
        db = self._connection()
        db.execute(
            "INSERT INTO pipeline_stages(asset_id,stage,status,model_version,input_sha256,error) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(asset_id,stage) DO UPDATE SET status=excluded.status,model_version=excluded.model_version,"
            "input_sha256=excluded.input_sha256,error=excluded.error,updated_at=CURRENT_TIMESTAMP",
            (asset_id, stage, status, model_version, input_sha256, error),
        )
        db.commit()

    def stage_ready(self, asset_id: int, stage: str, model_version: str, input_sha256: str) -> bool:
        row = self._connection().execute(
            "SELECT 1 FROM pipeline_stages WHERE asset_id=? AND stage=? AND status='ready' "
            "AND model_version=? AND input_sha256=?",
            (asset_id, stage, model_version, input_sha256),
        ).fetchone()
        return row is not None

    def stage_complete(self, asset_id: int, stage: str, model_version: str, input_sha256: str) -> bool:
        row = self._connection().execute(
            "SELECT 1 FROM pipeline_stages WHERE asset_id=? AND stage=? AND status IN ('ready','skipped') "
            "AND model_version=? AND input_sha256=?",
            (asset_id, stage, model_version, input_sha256),
        ).fetchone()
        return row is not None

    def stage_states(self, asset_id: int) -> dict[str, dict[str, str]]:
        rows = self._connection().execute(
            "SELECT stage,status,model_version,input_sha256,error,updated_at FROM pipeline_stages WHERE asset_id=?",
            (asset_id,),
        ).fetchall()
        return {str(row["stage"]): {key: str(row[key]) for key in row.keys() if key != "stage"} for row in rows}

    def invalidate_stages(self, asset_id: int) -> None:
        db = self._connection()
        db.execute("UPDATE pipeline_stages SET status='pending',input_sha256='',error='',updated_at=CURRENT_TIMESTAMP WHERE asset_id=?", (asset_id,))
        db.execute("DELETE FROM asset_embeddings WHERE asset_id=?", (asset_id,))
        db.execute("DELETE FROM region_embeddings WHERE asset_id=?", (asset_id,))
        db.commit()

    def recover_interrupted_stages(self) -> None:
        db = self._connection()
        db.execute("UPDATE pipeline_stages SET status='pending',error='Interrupted; queued to resume',updated_at=CURRENT_TIMESTAMP WHERE status='running'")
        db.commit()

    def queue_stale_stage(self, stage: str, model_version: str) -> int:
        """Mark old stage versions pending so diagnostics and resume state stay truthful."""
        db = self._connection()
        rows = db.execute(
            "SELECT asset_id FROM pipeline_stages WHERE stage=? AND model_version<>?",
            (stage, model_version),
        ).fetchall()
        asset_ids = [int(row[0]) for row in rows]
        if not asset_ids:
            return 0
        db.execute(
            "UPDATE pipeline_stages SET status='pending',error='Stage version changed; queued to resume',"
            "updated_at=CURRENT_TIMESTAMP WHERE stage=? AND model_version<>?",
            (stage, model_version),
        )
        marks = ",".join("?" for _ in asset_ids)
        db.execute(
            f"UPDATE assets SET status='partial' WHERE id IN ({marks}) AND status='ready'",
            asset_ids,
        )
        db.commit()
        return len(asset_ids)

    def queue_resolved_failures(self) -> int:
        """Retry failures addressed by the runtime memory/serialization fixes."""
        db = self._connection()
        markers = (
            "%CL_OUT_OF_RESOURCES%",
            "%CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST%",
            "%could not create a memory%",
            "%not JSON serializable%",
        )
        where = " OR ".join("error LIKE ?" for _marker in markers)
        rows = db.execute(
            f"SELECT asset_id FROM pipeline_stages WHERE status='error' AND ({where})",
            markers,
        ).fetchall()
        asset_ids = list(dict.fromkeys(int(row[0]) for row in rows))
        if not rows:
            return 0
        db.execute(
            f"UPDATE pipeline_stages SET status='pending',error='Resolved runtime failure; queued to resume',"
            f"updated_at=CURRENT_TIMESTAMP WHERE status='error' AND ({where})",
            markers,
        )
        if asset_ids:
            marks = ",".join("?" for _ in asset_ids)
            db.execute(
                f"UPDATE assets SET status='partial' WHERE id IN ({marks})",
                asset_ids,
            )
        db.commit()
        return len(rows)

    def save_embedding(self, asset_id: int, vector: np.ndarray, model_version: str) -> None:
        value = np.asarray(vector, dtype=np.float16).reshape(-1)
        db = self._connection()
        db.execute(
            "INSERT INTO asset_embeddings(asset_id,model_version,dimensions,vector) VALUES(?,?,?,?) "
            "ON CONFLICT(asset_id) DO UPDATE SET model_version=excluded.model_version,dimensions=excluded.dimensions,"
            "vector=excluded.vector,updated_at=CURRENT_TIMESTAMP",
            (asset_id, model_version, value.size, value.tobytes()),
        )
        db.commit()

    def embeddings(self, model_version: str, dimensions: int) -> list[tuple[int, np.ndarray]]:
        rows = self._connection().execute(
            "SELECT asset_id,vector FROM asset_embeddings WHERE model_version=? AND dimensions=?",
            (model_version, dimensions),
        ).fetchall()
        return [(int(row["asset_id"]), np.frombuffer(row["vector"], dtype=np.float16).astype(np.float32)) for row in rows]

    def region_ids(self, asset_id: int) -> list[int]:
        return [
            int(row[0]) for row in self._connection().execute(
                "SELECT id FROM region_embeddings WHERE asset_id=? ORDER BY region_index", (asset_id,)
            )
        ]

    def replace_region_embeddings(
        self, asset_id: int, boxes: list[list[int]], vectors: list[np.ndarray], model_version: str
    ) -> list[RegionEmbedding]:
        if len(boxes) != len(vectors):
            raise ValueError("Region boxes and vectors must have the same length")
        db = self._connection()
        db.execute("DELETE FROM region_embeddings WHERE asset_id=?", (asset_id,))
        result: list[RegionEmbedding] = []
        for index, (box, vector) in enumerate(zip(boxes, vectors)):
            value = np.asarray(vector, dtype=np.float16).reshape(-1)
            cursor = db.execute(
                "INSERT INTO region_embeddings(asset_id,region_index,box_json,model_version,dimensions,vector) "
                "VALUES(?,?,?,?,?,?)",
                (asset_id, index, json.dumps([int(item) for item in box]), model_version,
                 value.size, value.tobytes()),
            )
            result.append(RegionEmbedding(
                int(cursor.lastrowid), asset_id, index, [int(item) for item in box],
                value.astype(np.float32), model_version,
            ))
        db.commit()
        return result

    def region_embeddings(self, model_version: str, dimensions: int) -> list[RegionEmbedding]:
        rows = self._connection().execute(
            "SELECT id,asset_id,region_index,box_json,model_version,vector FROM region_embeddings "
            "WHERE model_version=? AND dimensions=? ORDER BY id",
            (model_version, dimensions),
        ).fetchall()
        return [RegionEmbedding(
            int(row["id"]), int(row["asset_id"]), int(row["region_index"]),
            [int(item) for item in json.loads(row["box_json"])],
            np.frombuffer(row["vector"], dtype=np.float16).astype(np.float32),
            str(row["model_version"]),
        ) for row in rows]

    def region_lookup(self, region_id: int) -> tuple[Asset, list[int]] | None:
        row = self._connection().execute(
            "SELECT r.box_json,a.* FROM region_embeddings r JOIN assets a ON a.id=r.asset_id WHERE r.id=?",
            (region_id,),
        ).fetchone()
        if row is None:
            return None
        return self._asset(row), [int(item) for item in json.loads(row["box_json"])]

    def region_lookup_many(self, region_ids: list[int]) -> dict[int, tuple[Asset, list[int]]]:
        """Resolve an ANN result batch with one SQLite query, not 500 round trips."""
        keys = list(dict.fromkeys(int(item) for item in region_ids))
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        rows = self._connection().execute(
            f"SELECT a.*,r.id region_id,r.box_json region_box FROM region_embeddings r "
            f"JOIN assets a ON a.id=r.asset_id WHERE r.id IN ({placeholders})",
            keys,
        ).fetchall()
        return {
            int(row["region_id"]): (
                self._asset(row),
                [int(item) for item in json.loads(row["region_box"] or "[]")],
            )
            for row in rows
        }

    def region_count(self) -> int:
        return int(self._connection().execute("SELECT count(*) FROM region_embeddings").fetchone()[0])

    def set_asset_status(self, asset_id: int, status: str, error: str = "") -> None:
        db = self._connection()
        db.execute("UPDATE assets SET status=?,error=?,indexed_at=CURRENT_TIMESTAMP WHERE id=?", (status, error, asset_id))
        db.commit()

    def list_roots(self) -> list[str]:
        return [str(row[0]) for row in self._connection().execute("SELECT path FROM roots ORDER BY path")]

    def add_root(self, path: str) -> None:
        db = self._connection()
        db.execute("INSERT OR IGNORE INTO roots(path) VALUES(?)", (path,))
        db.commit()

    def all_assets(self) -> list[Asset]:
        return [self._asset(row) for row in self._connection().execute("SELECT * FROM assets ORDER BY path")]

    def delete_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        db = self._connection()
        db.execute(f"DELETE FROM asset_fts WHERE asset_id IN ({marks})", ids)
        db.execute(f"DELETE FROM assets WHERE id IN ({marks})", ids)
        db.commit()

    def count(self) -> int:
        return int(self._connection().execute("SELECT count(*) FROM assets WHERE status='ready'").fetchone()[0])

    def counts_by_status(self) -> dict[str, int]:
        rows = self._connection().execute("SELECT status,count(*) count FROM assets GROUP BY status").fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def index_coverage(self) -> dict[str, object]:
        """Compact index coverage snapshot used by the developer search trace."""
        db = self._connection()
        stages: dict[str, dict[str, int]] = {}
        for row in db.execute(
            "SELECT stage,status,count(*) count FROM pipeline_stages GROUP BY stage,status"
        ).fetchall():
            stages.setdefault(str(row["stage"]), {})[str(row["status"])] = int(row["count"])
        stage_errors = [
            {"stage": str(row["stage"]), "error": str(row["error"]), "count": int(row["count"])}
            for row in db.execute(
                "SELECT stage,error,count(*) count FROM pipeline_stages WHERE status='error' "
                "GROUP BY stage,error ORDER BY count DESC LIMIT 8"
            ).fetchall()
        ]
        return {
            "asset_status": self.counts_by_status(),
            "stages": stages,
            "stage_errors": stage_errors,
            "full_embeddings": int(db.execute("SELECT count(*) FROM asset_embeddings").fetchone()[0]),
            "region_vectors": int(db.execute("SELECT count(*) FROM region_embeddings").fetchone()[0]),
            "region_assets": int(db.execute("SELECT count(DISTINCT asset_id) FROM region_embeddings").fetchone()[0]),
            "ocr_assets": int(db.execute("SELECT count(DISTINCT asset_id) FROM ocr_lines").fetchone()[0]),
            "ocr_lines": int(db.execute("SELECT count(*) FROM ocr_lines").fetchone()[0]),
            "object_assets": int(db.execute("SELECT count(DISTINCT asset_id) FROM object_detections").fetchone()[0]),
            "objects": int(db.execute("SELECT count(*) FROM object_detections").fetchone()[0]),
        }


def asset_matches_metadata(asset: Asset, filters: dict[str, str]) -> bool:
    extension = filters.get("extension")
    if extension and Path(asset.path).suffix.casefold() != extension:
        return False
    folder = filters.get("folder")
    if folder and folder not in normalize_text(str(Path(asset.path).parent)):
        return False
    date_prefix = filters.get("date_prefix")
    if date_prefix:
        date = datetime.fromtimestamp(asset.modified_ns / 1_000_000_000).strftime("%Y-%m-%d")
        if not date.startswith(date_prefix):
            return False
    return True


def metadata_reason(filters: dict[str, str]) -> str:
    values = []
    if "extension" in filters:
        values.append(f"格式：{filters['extension']}")
    if "date_prefix" in filters:
        values.append(f"日期：{filters['date_prefix']}")
    if "folder" in filters:
        values.append(f"資料夾：{filters['folder']}")
    return "Metadata 條件：" + "、".join(values)
