"""OR-Tools monitoring & analytics endpoints."""
import logging
import time as time_module
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from services.ortools_monitoring import ortools_monitor, get_monitoring_dashboard, get_benchmark_report

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v4/monitoring", tags=["OR-Tools Monitoring"])


@router.get("/dashboard")
async def get_dashboard():
    """Comprehensive OR-Tools monitoring dashboard."""
    start = time_module.time()
    data = await get_monitoring_dashboard()
    data["query_time_ms"] = round((time_module.time() - start) * 1000, 2)
    return data


@router.get("/benchmark")
async def get_benchmark_comparison():
    """OR-Tools vs Legacy benchmark comparison."""
    start = time_module.time()
    data = await get_benchmark_report()
    data["query_time_ms"] = round((time_module.time() - start) * 1000, 2)
    return data


@router.get("/alerts")
async def get_active_alerts():
    """Active OR-Tools production alerts."""
    start = time_module.time()

    active_alerts = list(ortools_monitor.active_alerts.values())
    cutoff = datetime.now() - timedelta(hours=24)
    recent_alerts = [a for a in ortools_monitor.alert_history if a["timestamp"] >= cutoff]

    severity_counts = {level: 0 for level in ("HIGH", "MEDIUM", "LOW")}
    for a in active_alerts:
        s = a.get("severity", "LOW")
        severity_counts[s] = severity_counts.get(s, 0) + 1

    if any(a.get("severity") == "HIGH" for a in active_alerts):
        health = "CRITICAL"
    elif active_alerts:
        health = "WARNING"
    else:
        health = "HEALTHY"

    return {
        "timestamp": datetime.now().isoformat(),
        "active_alerts": active_alerts,
        "active_count": len(active_alerts),
        "alert_history_24h": recent_alerts,
        "alerts_24h_count": len(recent_alerts),
        "severity_breakdown": severity_counts,
        "health_status": health,
        "query_time_ms": round((time_module.time() - start) * 1000, 2),
    }


@router.get("/health")
async def get_ortools_health():
    """Quick OR-Tools health check."""
    start = time_module.time()
    summary = await ortools_monitor.get_performance_summary(hours=1)
    health_score = 0

    if not summary or summary.get("overview", {}).get("total_requests", 0) == 0:
        health_status = "NO_DATA"
    else:
        overview = summary["overview"]
        perf = summary["performance"]
        success_score = overview["success_rate"] * 50
        time_score = max(0, 50 - (perf["avg_execution_time_ms"] / 100))
        health_score = min(100, success_score + time_score)

        if health_score >= 90:
            health_status = "EXCELLENT"
        elif health_score >= 75:
            health_status = "GOOD"
        elif health_score >= 50:
            health_status = "DEGRADED"
        else:
            health_status = "CRITICAL"

    if ortools_monitor.active_alerts:
        health_status = "ALERTS_ACTIVE"

    recommendations = []
    if health_status == "CRITICAL":
        recommendations.append("Immediate investigation required")
    elif health_status == "NO_DATA":
        recommendations.append("No recent OR-Tools activity")

    return {
        "timestamp": datetime.now().isoformat(),
        "health_status": health_status,
        "health_score": round(health_score, 1),
        "active_alerts": len(ortools_monitor.active_alerts),
        "recent_metrics": summary.get("overview", {}),
        "performance_indicators": {
            "avg_response_time_ms": summary.get("performance", {}).get("avg_execution_time_ms", 0),
            "success_rate": summary.get("overview", {}).get("success_rate", 0),
            "requests_last_hour": summary.get("overview", {}).get("total_requests", 0),
        },
        "recommendations": recommendations,
        "query_time_ms": round((time_module.time() - start) * 1000, 2),
    }


@router.get("/metrics/summary")
async def get_metrics_summary(hours: int = 24):
    """Detailed OR-Tools metrics summary (1-168 hours)."""
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=400, detail="hours must be 1-168")

    start = time_module.time()
    summary = await ortools_monitor.get_performance_summary(hours=hours)
    summary["query_time_ms"] = round((time_module.time() - start) * 1000, 2)
    return summary
