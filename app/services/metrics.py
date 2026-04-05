from __future__ import annotations

import threading

_lock = threading.Lock()
_total_processed = 0
_successful_processed = 0
_failed_processed = 0
_total_processing_time_ms = 0


def record_processing(elapsed_ms: int, success: bool) -> None:
    global _total_processed, _successful_processed, _failed_processed, _total_processing_time_ms
    with _lock:
        _total_processed += 1
        _total_processing_time_ms += max(0, int(elapsed_ms))
        if success:
            _successful_processed += 1
        else:
            _failed_processed += 1


def get_metrics() -> dict:
    with _lock:
        avg = (_total_processing_time_ms / _total_processed) if _total_processed else 0.0
        success_rate = (_successful_processed / _total_processed) if _total_processed else 0.0
        return {
            "total_processed": _total_processed,
            "successful_processed": _successful_processed,
            "failed_processed": _failed_processed,
            "avg_time_ms": round(avg, 2),
            "success_rate": round(success_rate, 4),
        }
