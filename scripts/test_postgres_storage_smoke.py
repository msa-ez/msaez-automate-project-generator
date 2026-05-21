"""PostgresStorageSystem 어댑터 스모크 테스트 (Phase 1 검증)

로컬 dev PostgreSQL(platform/data-gateway/docker-compose.dev.yml) 이 떠 있어야 한다.
실행: venv/bin/python scripts/test_postgres_storage_smoke.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# project_generator 에는 utils <-> systems 순환 import 가 있다(기존 코드).
# utils 패키지를 먼저 완전 초기화하면 회피된다 (실제 앱의 import 순서와 동일).
import project_generator.utils  # noqa: E402, F401

from project_generator.systems.postgres_storage_system import PostgresStorageSystem  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def busy_wait_until(fn, tries=5000):
    """fire-and-forget 결과를 sleep 없이 폴링."""
    for _ in range(tries):
        if fn():
            return True
    return False


def main():
    db = PostgresStorageSystem.initialize(
        host="localhost", port=5432, dbname="msaez", user="msaez", password="msaez_dev"
    )
    NS = "pg_smoke"

    for p in (f"jobs/{NS}/j1", f"jobs/{NS}/j2", f"jobs/{NS}/tx1",
              f"requestedJobs/{NS}/r1", f"requestedJobs/{NS}/r2",
              f"jobs/{NS}/ff1", "kvtest/pg/x1"):
        db.delete_data(p)

    # 1. 코어: set/get
    db.set_data(f"jobs/{NS}/j1", {"state": {"outputs": {}}})
    check("set/get 전체 행", db.get_data(f"jobs/{NS}/j1") == {"state": {"outputs": {}}})

    # 2. 깊은 서브경로
    db.set_data(f"jobs/{NS}/j1/state/outputs/done", True)
    check("깊은 서브경로 set/get", db.get_data(f"jobs/{NS}/j1/state/outputs/done") is True)

    # 3. update merge
    db.update_data(f"jobs/{NS}/j1/state/outputs", {"count": 1})
    out = db.get_data(f"jobs/{NS}/j1/state/outputs")
    check("update merge", out == {"done": True, "count": 1})

    # 4. conditional_update
    old = db.get_data(f"jobs/{NS}/j1")
    new = {"state": {"outputs": {"done": True, "count": 2}}}
    db.conditional_update_data(f"jobs/{NS}/j1", new, old)
    check("conditional_update", db.get_data(f"jobs/{NS}/j1/state/outputs/count") == 2)

    # 5. get_children
    db.set_data(f"jobs/{NS}/j2", {"state": {}})
    children = db.get_children_data(f"jobs/{NS}")
    check("get_children", children is not None and {"j1", "j2"} <= set(children.keys()))

    # 6. delete
    db.delete_data(f"jobs/{NS}/j2")
    check("delete", db.get_data(f"jobs/{NS}/j2") is None)

    # 7. kv_store catch-all
    db.set_data("kvtest/pg/x1", {"v": 1})
    check("kv set/get", db.get_data("kvtest/pg/x1") == {"v": 1})

    # 8. transaction — 원자적 카운터 증가
    db.set_data(f"jobs/{NS}/tx1", {"counter": 0})

    def inc(cur):
        cur = dict(cur)
        cur["counter"] = cur.get("counter", 0) + 1
        return cur

    db.transaction(f"jobs/{NS}/tx1", inc)
    db.transaction(f"jobs/{NS}/tx1", inc)
    result = db.transaction(f"jobs/{NS}/tx1", inc)
    check("transaction 원자적 증가", result.get("counter") == 3
          and db.get_data(f"jobs/{NS}/tx1/counter") == 3)

    # 9. transaction — 서브경로
    db.transaction(f"jobs/{NS}/tx1/state/outputs",
                   lambda c: {"merged": True})
    check("transaction 서브경로", db.get_data(f"jobs/{NS}/tx1/state/outputs") == {"merged": True})

    # 10. fire-and-forget
    db.set_data_fire_and_forget(f"jobs/{NS}/ff1", {"fire": "forget"})
    ok = busy_wait_until(lambda: db.get_data(f"jobs/{NS}/ff1") == {"fire": "forget"})
    check("set_data_fire_and_forget", ok)

    # 11. async 변형
    async def _async_checks():
        await db.set_data_async(f"jobs/{NS}/j1/state/outputs/asyncflag", "Y")
        v = await db._run(db.get_data, f"jobs/{NS}/j1/state/outputs/asyncflag")
        return v == "Y"

    check("async set/get", asyncio.run(_async_checks()))

    # 12. SKIP LOCKED 큐
    db.set_data(f"requestedJobs/{NS}/r1", {"x": 1})
    db.set_data(f"requestedJobs/{NS}/r2", {"x": 2})
    c1 = db.claim_pending_job(NS, "pod-A")
    c2 = db.claim_pending_job(NS, "pod-B")
    c3 = db.claim_pending_job(NS, "pod-C")
    check("SKIP LOCKED 2건 클레임",
          c1 and c2 and {c1["job_id"], c2["job_id"]} == {"r1", "r2"})
    check("SKIP LOCKED 소진 시 None", c3 is None)

    # 13. sanitize/restore passthrough
    check("sanitize passthrough", db.sanitize_data_for_storage({"a": None}) == {"a": None})
    check("database property", db.database is db)

    for p in (f"jobs/{NS}/j1", f"jobs/{NS}/tx1", f"jobs/{NS}/ff1",
              f"requestedJobs/{NS}/r1", f"requestedJobs/{NS}/r2", "kvtest/pg/x1"):
        db.delete_data(p)

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
