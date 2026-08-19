"""Demo: 5 queries against HybridMemoryAgent. Run: `python bonus/demo.py` (exits 0).

Each query exercises a different part of the design:
  1. vector-only lookup            -> Kubernetes memory
  2. profile-driven recommendation -> topic_affinity (Feast / fallback)
  3. fresh activity                -> queries_last_hour (streaming-style feature)
  4. paraphrase                    -> vector/lexical hybrid over "tự động mở rộng"
  5. mixed (episodic + profile)    -> cloud + security memories
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bonus.agent import HybridMemoryAgent  # noqa: E402


def _try_feast_store():
    """Use the lab's real Feast store if NB4 materialized it, else None."""
    try:
        from feast import FeatureStore
        repo = _ROOT / "app" / "feast_repo"
        if (repo / "registry.db").exists():
            return FeatureStore(repo_path=str(repo))
    except Exception:
        pass
    return None


def _seed(agent: HybridMemoryAgent) -> None:
    """Episodic memories of 'things u_001 has read' — 5 topics so that the
    top-3 selection actually discriminates (RRF from NB2 is visible)."""
    agent.remember(
        "Kubernetes: triển khai cluster nhiều node, autoscaling theo CPU "
        "và bộ nhớ, dùng horizontal pod autoscaler để mở rộng pod khi tải cao.",
        "u_001",
    )
    agent.remember(
        "Cloud autoscaling: tự động mở rộng hạ tầng theo lưu lượng, kết hợp "
        "spot instance để giảm chi phí; cân bằng tải giữa nhiều region giúp "
        "giảm độ trễ cho người dùng Việt Nam.",
        "u_001",
    )
    agent.remember(
        "Bảo mật API: OAuth 2.0 và JWT, mã hoá dữ liệu nhạy cảm khi lưu trữ, "
        "chống rò rỉ dữ liệu giữa các tenant trong cùng hệ thống.",
        "u_001",
    )
    agent.remember(
        "Database: tối ưu truy vấn SQL bằng chỉ mục B-tree và phân vùng theo "
        "ngày để tăng tốc báo cáo.",
        "u_001",
    )
    agent.remember(
        "Frontend: cải thiện LCP bằng lazy loading ảnh và code-splitting React.",
        "u_001",
    )


def main() -> int:
    store = _try_feast_store()
    agent = HybridMemoryAgent(user_id="u_001", feature_store=store)
    _seed(agent)
    print(f"profile source: {'feast' if store is not None else 'deterministic-fallback'}")

    demos = [
        ("1. vector-only lookup", "Tôi đã đọc gì về Kubernetes?"),
        ("2. profile-driven", "Recommend đọc gì tiếp cho tôi"),
        ("3. fresh activity", "Tôi đang quan tâm gì gần đây?"),
        ("4. paraphrase", "Tài liệu về tự động mở rộng hạ tầng?"),
        ("5. mixed (episodic + profile)", "Cho tôi summary về cloud security"),
    ]
    for label, q in demos:
        print(f"\n===== {label}: «{q}» =====")
        print(agent.recall(q))

    print("\nDEMO OK — 5 queries done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
