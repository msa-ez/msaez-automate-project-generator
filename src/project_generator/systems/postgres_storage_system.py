"""PostgreSQL Storage 시스템 구현 (v1.0.30)

project-generator 의 StorageSystem 인터페이스를 PostgreSQL 로 구현한다.
설계 근거: DB-migration-plan.md §4·§5. eventstorming-generator 의
postgres_system.py 와 코어(경로 라우팅·JSONB read/write)는 동일하며,
StorageSystem 이 추가로 요구하는 async / fire-and-forget / transaction /
watch 메서드를 더한다.

핵심 모델 — "행 식별 + value 내부 JSON 경로":
    AceBase 경로  jobs/{ns}/{jobId}/state/outputs/logs
      -> 테이블 jobs, 행 (job_id={jobId}, namespace={ns})
      -> value 내부 JSON 경로 {state, outputs, logs}
"""
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Any, Optional, Callable, List

try:
    import psycopg
    from psycopg.types.json import Jsonb
    from psycopg_pool import ConnectionPool
except ImportError:  # 의존성 미설치 시 import 단계에서 죽지 않도록
    psycopg = None
    Jsonb = None
    ConnectionPool = None

from ..utils.logging_util import LoggingUtil
from .storage_system import StorageSystem


@dataclass
class _Route:
    """경로 라우팅 결과."""
    kind: str                       # 'row' | 'collection' | 'kv'
    table: str = ""
    pk: Dict[str, str] = field(default_factory=dict)
    filters: Dict[str, str] = field(default_factory=dict)
    insert_cols: Dict[str, str] = field(default_factory=dict)
    subpath: List[str] = field(default_factory=list)
    key_col: str = ""
    path: str = ""


_TABLE_PK = {
    "jobs": ["job_id"],
    "requested_jobs": ["job_id"],
    "definitions": ["project_id"],
    "user_lists": ["uid", "list_type", "project_id"],
    "definition_queue": ["project_id", "seq_key"],
    "definition_snapshots": ["project_id", "snapshot_key"],
    "users": ["uid"],
}


class PostgresStorageSystem(StorageSystem):
    """PostgreSQL Storage 시스템 구현 (싱글톤)."""

    _instance: Optional["PostgresStorageSystem"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, host: str = None, port: int = None, dbname: str = None,
                 user: str = None, password: str = None):
        if self._initialized:
            return
        if psycopg is None:
            raise RuntimeError(
                "psycopg 가 설치되어 있지 않습니다. pyproject.toml 의 의존성을 확인하세요 "
                "(psycopg[binary], psycopg-pool)."
            )
        if host is None or port is None or dbname is None:
            raise ValueError("host, port, dbname 은 필수 매개변수입니다.")

        conninfo = (
            f"host={host} port={port} dbname={dbname} "
            f"user={user or ''} password={password or ''}"
        )
        self._pool = ConnectionPool(conninfo, min_size=1, max_size=10, open=False)
        self._pool.open(wait=True, timeout=10)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self._initialized = True
        LoggingUtil.info("postgres_storage_system",
                         f"PostgreSQL 연결 초기화: {host}:{port}/{dbname}")

    @classmethod
    def initialize(cls, host: str = None, port: int = None, dbname: str = None,
                   user: str = None, password: str = None, **kwargs) -> "PostgresStorageSystem":
        """싱글톤 인스턴스 초기화."""
        if cls._instance is None or not cls._instance._initialized:
            cls._instance = cls(host, port, dbname, user, password)
        return cls._instance

    @classmethod
    def instance(cls) -> "PostgresStorageSystem":
        """초기화된 싱글톤 인스턴스 반환."""
        if cls._instance is None or not cls._instance._initialized:
            raise RuntimeError(
                "PostgresStorageSystem 이 초기화되지 않았습니다. "
                "먼저 PostgresStorageSystem.initialize() 를 호출하세요."
            )
        return cls._instance

    # =========================================================================
    # 경로 라우팅
    # =========================================================================
    def _route(self, path: str) -> _Route:
        """AceBase 경로를 (테이블, 행 키, JSON 서브경로) 로 변환한다."""
        segs = [s for s in (path or "").strip("/").split("/") if s != ""]
        if not segs:
            return _Route(kind="kv", table="kv_store", path="")

        head = segs[0]

        if head in ("jobs", "requestedJobs"):
            table = "jobs" if head == "jobs" else "requested_jobs"
            if len(segs) == 1:
                return _Route(kind="collection", table=table, key_col="job_id")
            namespace = segs[1]
            if len(segs) == 2:
                return _Route(kind="collection", table=table,
                              filters={"namespace": namespace}, key_col="job_id")
            return _Route(kind="row", table=table,
                          pk={"job_id": segs[2]},
                          insert_cols={"namespace": namespace},
                          subpath=segs[3:])

        if head == "definitions":
            if len(segs) == 1:
                return _Route(kind="collection", table="definitions", key_col="project_id")
            pid = segs[1]
            if len(segs) >= 3 and segs[2] == "queue":
                if len(segs) == 3:
                    return _Route(kind="collection", table="definition_queue",
                                  filters={"project_id": pid}, key_col="seq_key")
                return _Route(kind="row", table="definition_queue",
                              pk={"project_id": pid, "seq_key": segs[3]},
                              subpath=segs[4:])
            if len(segs) >= 3 and segs[2] == "snapshotLists":
                if len(segs) == 3:
                    return _Route(kind="collection", table="definition_snapshots",
                                  filters={"project_id": pid}, key_col="snapshot_key")
                return _Route(kind="row", table="definition_snapshots",
                              pk={"project_id": pid, "snapshot_key": segs[3]},
                              subpath=segs[4:])
            return _Route(kind="row", table="definitions",
                          pk={"project_id": pid}, subpath=segs[2:])

        if head == "userLists":
            if len(segs) == 3:
                return _Route(kind="collection", table="user_lists",
                              filters={"uid": segs[1], "list_type": segs[2]},
                              key_col="project_id")
            if len(segs) >= 4:
                return _Route(kind="row", table="user_lists",
                              pk={"uid": segs[1], "list_type": segs[2], "project_id": segs[3]},
                              subpath=segs[4:])
            return _Route(kind="kv", table="kv_store", path=path.strip("/"))

        if head == "users" and len(segs) >= 2:
            return _Route(kind="row", table="users",
                          pk={"uid": segs[1]}, subpath=segs[2:])

        return _Route(kind="kv", table="kv_store", path=path.strip("/"))

    # =========================================================================
    # JSON 서브경로 헬퍼
    # =========================================================================
    @staticmethod
    def _nested_get(root: Any, subpath: List[str]) -> Any:
        cur = root
        for k in subpath:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur

    @staticmethod
    def _nested_set(root: Dict[str, Any], subpath: List[str], value: Any) -> None:
        cur = root
        for k in subpath[:-1]:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        cur[subpath[-1]] = value

    @staticmethod
    def _nested_merge(root: Dict[str, Any], subpath: List[str], value: Dict[str, Any]) -> None:
        cur = root
        for k in subpath:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        if isinstance(value, dict):
            cur.update(value)

    def _find_data_differences(self, new_data: Dict[str, Any], old_data: Dict[str, Any],
                               path_prefix: str = "") -> Dict[str, Any]:
        """두 딕셔너리를 재귀 비교하여 변경 경로-값 딕셔너리를 반환한다."""
        updates: Dict[str, Any] = {}
        for key, new_value in new_data.items():
            current_path = f"{path_prefix}/{key}" if path_prefix else key
            old_value = old_data.get(key) if old_data else None
            if isinstance(new_value, dict) and isinstance(old_value, dict):
                updates.update(self._find_data_differences(new_value, old_value, current_path))
            elif new_value != old_value:
                updates[current_path] = new_value
        if old_data:
            for key in old_data:
                if key not in new_data:
                    current_path = f"{path_prefix}/{key}" if path_prefix else key
                    updates[current_path] = None
        return updates

    # =========================================================================
    # 공통 실행 래퍼
    # =========================================================================
    def _safe(self, op_name: str, fn: Callable, default: Any) -> Any:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            LoggingUtil.exception("postgres_storage_system", f"{op_name} 실패", e)
            return default

    @staticmethod
    def _pk_where(r: _Route):
        where = " AND ".join(f"{c} = %s" for c in r.pk)
        return where, list(r.pk.values())

    # =========================================================================
    # set / update / get / delete
    # =========================================================================
    def set_data(self, path: str, data: Any) -> bool:
        def _op():
            r = self._route(path)
            if r.kind == "kv":
                return self._kv_write(r.path, data, merge=False)
            if r.kind == "collection":
                LoggingUtil.warning("postgres_storage_system", f"set_data 컬렉션 경로 미지원: {path}")
                return False
            if r.subpath:
                return self._row_subpath_write(r, data, mode="set")
            return self._row_write(r, data, merge=False)
        return self._safe("데이터 업로드", _op, False)

    def update_data(self, path: str, data: Any) -> bool:
        def _op():
            r = self._route(path)
            if r.kind == "kv":
                return self._kv_write(r.path, data, merge=True)
            if r.kind == "collection":
                LoggingUtil.warning("postgres_storage_system", f"update_data 컬렉션 경로 미지원: {path}")
                return False
            if r.subpath:
                return self._row_subpath_write(r, data, mode="merge")
            return self._row_write(r, data, merge=True)
        return self._safe("데이터 업데이트", _op, False)

    def get_data(self, path: str) -> Optional[Any]:
        def _op():
            r = self._route(path)
            if r.kind == "kv":
                return self._kv_read(r.path)
            if r.kind == "collection":
                return self._collection_read(r)
            value = self._row_read(r)
            if value is None:
                return None
            if r.subpath:
                return self._nested_get(value, r.subpath)
            return value
        return self._safe("데이터 조회", _op, None)

    def delete_data(self, path: str) -> bool:
        def _op():
            r = self._route(path)
            if r.kind == "kv":
                return self._kv_delete(r.path)
            if r.kind == "collection":
                LoggingUtil.warning("postgres_storage_system", f"delete_data 컬렉션 경로 미지원: {path}")
                return False
            if r.subpath:
                return self._row_subpath_delete(r)
            return self._row_delete(r)
        return self._safe("데이터 삭제", _op, False)

    def conditional_update_data(self, path: str, data_to_update: Dict[str, Any],
                                previous_data: Dict[str, Any]) -> bool:
        updates = self._find_data_differences(data_to_update, previous_data)
        if not updates:
            return True
        LoggingUtil.info("postgres_storage_system",
                         f"[conditional_update] path={path} diff_count={len(updates)}")
        for rel_path, value in updates.items():
            full_path = f"{path}/{rel_path}" if path else rel_path
            ok = self.delete_data(full_path) if value is None else self.set_data(full_path, value)
            if ok is False:
                return False
        return True

    def get_children_data(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        data = self.get_data(path)
        if data is None or not isinstance(data, dict):
            return None
        return data

    # =========================================================================
    # 정규화 테이블 — 행 단위
    # =========================================================================
    def _row_read(self, r: _Route) -> Optional[Any]:
        where, params = self._pk_where(r)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT value FROM {r.table} WHERE {where}", params)
            row = cur.fetchone()
            return row[0] if row else None

    def _row_write(self, r: _Route, data: Any, merge: bool) -> bool:
        cols = list(r.pk.keys()) + list(r.insert_cols.keys()) + ["value"]
        vals = list(r.pk.values()) + list(r.insert_cols.values()) + [Jsonb(data)]
        placeholders = ", ".join(["%s"] * len(cols))
        conflict = ", ".join(_TABLE_PK[r.table])
        if merge:
            set_clause = f"value = {r.table}.value || EXCLUDED.value, updated_at = now()"
        else:
            set_clause = "value = EXCLUDED.value, updated_at = now()"
        sql = (
            f"INSERT INTO {r.table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}"
        )
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, vals)
        return True

    def _row_subpath_write(self, r: _Route, data: Any, mode: str) -> bool:
        where, params = self._pk_where(r)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT value FROM {r.table} WHERE {where} FOR UPDATE", params)
            row = cur.fetchone()
            value = row[0] if row and isinstance(row[0], dict) else {}
            if mode == "merge":
                self._nested_merge(value, r.subpath, data)
            else:
                self._nested_set(value, r.subpath, data)
            if row:
                cur.execute(
                    f"UPDATE {r.table} SET value = %s, updated_at = now() WHERE {where}",
                    [Jsonb(value)] + params,
                )
            else:
                cols = list(r.pk.keys()) + list(r.insert_cols.keys()) + ["value"]
                vals = list(r.pk.values()) + list(r.insert_cols.values()) + [Jsonb(value)]
                placeholders = ", ".join(["%s"] * len(cols))
                cur.execute(
                    f"INSERT INTO {r.table} ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
        return True

    def _row_delete(self, r: _Route) -> bool:
        where, params = self._pk_where(r)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {r.table} WHERE {where}", params)
        return True

    def _row_subpath_delete(self, r: _Route) -> bool:
        where, params = self._pk_where(r)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {r.table} SET value = value #- %s, updated_at = now() WHERE {where}",
                [list(r.subpath)] + params,
            )
        return True

    def _collection_read(self, r: _Route) -> Optional[Dict[str, Any]]:
        if r.filters:
            where = " AND ".join(f"{c} = %s" for c in r.filters)
            params = list(r.filters.values())
        else:
            where, params = "TRUE", []
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {r.key_col}, value FROM {r.table} WHERE {where}", params)
            rows = cur.fetchall()
        if not rows:
            return None
        return {row[0]: row[1] for row in rows}

    # =========================================================================
    # kv_store — catch-all
    # =========================================================================
    def _kv_write(self, path: str, data: Any, merge: bool) -> bool:
        if merge:
            set_clause = "value = kv_store.value || EXCLUDED.value, updated_at = now()"
        else:
            set_clause = "value = EXCLUDED.value, updated_at = now()"
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO kv_store (path, value) VALUES (%s, %s) "
                f"ON CONFLICT (path) DO UPDATE SET {set_clause}",
                [path, Jsonb(data)],
            )
        return True

    def _kv_read(self, path: str) -> Optional[Any]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM kv_store WHERE path = %s", [path])
            exact = cur.fetchone()
            cur.execute("SELECT path, value FROM kv_store WHERE path LIKE %s", [path + "/%"])
            children = cur.fetchall()
        if not children:
            return exact[0] if exact else None
        assembled: Dict[str, Any] = {}
        for child_path, value in children:
            rel = child_path[len(path) + 1:].split("/")
            self._nested_set(assembled, rel, value)
        if exact and isinstance(exact[0], dict):
            merged = dict(exact[0])
            merged.update(assembled)
            return merged
        return assembled

    def _kv_delete(self, path: str) -> bool:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM kv_store WHERE path = %s OR path LIKE %s",
                        [path, path + "/%"])
        return True

    # =========================================================================
    # 비동기 / fire-and-forget 래퍼
    # =========================================================================
    async def _run(self, fn: Callable, *args) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, partial(fn, *args))

    async def set_data_async(self, path: str, data: Dict[str, Any]) -> bool:
        return await self._run(self.set_data, path, data)

    async def update_data_async(self, path: str, data: Dict[str, Any]) -> bool:
        return await self._run(self.update_data, path, data)

    async def delete_data_async(self, path: str) -> bool:
        return await self._run(self.delete_data, path)

    async def conditional_update_data_async(self, path: str, data_to_update: Dict[str, Any],
                                            previous_data: Dict[str, Any]) -> bool:
        return await self._run(self.conditional_update_data, path, data_to_update, previous_data)

    async def get_children_data_async(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        return await self._run(self.get_children_data, path)

    def set_data_fire_and_forget(self, path: str, data: Dict[str, Any]) -> None:
        self._executor.submit(self.set_data, path, data)

    def update_data_fire_and_forget(self, path: str, data: Dict[str, Any]) -> None:
        self._executor.submit(self.update_data, path, data)

    def conditional_update_data_fire_and_forget(self, path: str, data_to_update: Dict[str, Any],
                                                previous_data: Dict[str, Any]) -> None:
        self._executor.submit(self.conditional_update_data, path, data_to_update, previous_data)

    def delete_data_fire_and_forget(self, path: str) -> None:
        self._executor.submit(self.delete_data, path)

    # =========================================================================
    # 트랜잭션 — PostgreSQL FOR UPDATE 기반 진짜 원자적 연산
    # (AceBase 는 비원자적 read-modify-write 였음 — §5.1 개선점)
    # =========================================================================
    def transaction(self, path: str, update_function: Callable) -> Any:
        def _op():
            r = self._route(path)
            if r.kind == "collection":
                LoggingUtil.warning("postgres_storage_system", f"transaction 컬렉션 경로 미지원: {path}")
                return None

            with self._pool.connection() as conn, conn.cursor() as cur:
                if r.kind == "kv":
                    cur.execute("SELECT value FROM kv_store WHERE path = %s FOR UPDATE", [r.path])
                    row = cur.fetchone()
                    current = row[0] if row else {}
                    updated = update_function(current if current is not None else {})
                    if updated is None or updated == current:
                        return updated
                    cur.execute(
                        "INSERT INTO kv_store (path, value) VALUES (%s, %s) "
                        "ON CONFLICT (path) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                        [r.path, Jsonb(updated)],
                    )
                    return updated

                # 정규화 테이블 행
                where, params = self._pk_where(r)
                cur.execute(f"SELECT value FROM {r.table} WHERE {where} FOR UPDATE", params)
                row = cur.fetchone()
                value = row[0] if row else None
                current = (self._nested_get(value, r.subpath)
                           if (value is not None and r.subpath) else value)
                updated = update_function(current if current is not None else {})
                if updated is None or updated == current:
                    return updated

                if r.subpath:
                    base = value if isinstance(value, dict) else {}
                    self._nested_set(base, r.subpath, updated)
                    new_value = base
                else:
                    new_value = updated

                if row:
                    cur.execute(
                        f"UPDATE {r.table} SET value = %s, updated_at = now() WHERE {where}",
                        [Jsonb(new_value)] + params,
                    )
                else:
                    cols = list(r.pk.keys()) + list(r.insert_cols.keys()) + ["value"]
                    vals = list(r.pk.values()) + list(r.insert_cols.values()) + [Jsonb(new_value)]
                    placeholders = ", ".join(["%s"] * len(cols))
                    cur.execute(
                        f"INSERT INTO {r.table} ({', '.join(cols)}) VALUES ({placeholders})",
                        vals,
                    )
                return updated
        return self._safe("트랜잭션", _op, None)

    async def transaction_async(self, path: str, update_function: Callable) -> Any:
        return await self._run(self.transaction, path, update_function)

    # =========================================================================
    # LLM Job 큐 — SKIP LOCKED 클레임 (인터페이스 외 추가 메서드)
    # =========================================================================
    def claim_pending_job(self, namespace: str, pod_id: str) -> Optional[Dict[str, Any]]:
        """대기 중인 requested_job 하나를 원자적으로 점유한다 (§4.3)."""
        def _op():
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE requested_jobs SET status = 'processing', "
                    "locked_by = %s, locked_at = now() "
                    "WHERE job_id = ("
                    "  SELECT job_id FROM requested_jobs "
                    "  WHERE namespace = %s AND status = 'pending' "
                    "  ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) "
                    "RETURNING job_id, value",
                    [pod_id, namespace],
                )
                row = cur.fetchone()
                return {"job_id": row[0], "value": row[1]} if row else None
        return self._safe("Job 큐 클레임", _op, None)

    # =========================================================================
    # watch — PostgreSQL 직결 어댑터에서는 미사용 (실시간은 게이트웨이 담당)
    # =========================================================================
    def watch_data(self, path: str, callback: Callable[[Optional[Dict[str, Any]]], None]) -> bool:
        LoggingUtil.warning("postgres_storage_system",
                            f"watch_data 미지원 (실시간은 게이트웨이 담당): {path}")
        return False

    async def watch_data_async(self, path: str,
                               callback: Callable[[Optional[Dict[str, Any]]], None]) -> bool:
        return False

    def unwatch_data(self, path: str) -> bool:
        return False

    async def unwatch_data_async(self, path: str) -> bool:
        return False

    def unwatch_all(self) -> bool:
        return True

    async def unwatch_all_async(self) -> bool:
        return True

    def get_active_watchers(self) -> list[str]:
        return []

    # =========================================================================
    # 데이터 정제 — PostgreSQL JSONB 는 null/[]/{} 를 네이티브 저장 → passthrough
    # =========================================================================
    def sanitize_data_for_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def restore_data_from_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    @property
    def database(self):
        """데이터베이스 참조 객체 (호환성용 — PostgreSQL 어댑터는 self 반환)."""
        return self
