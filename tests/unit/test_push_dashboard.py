"""Preparing the generated dashboard for a Grafana Cloud stack."""

from push_dashboard import for_cloud


def dashboard():
    return {
        "id": 7,
        "uid": "legal-rag",
        "panels": [
            {
                "title": "p95",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}}],
            },
            {
                "title": "drift runs",
                "datasource": {"type": "grafana-postgresql-datasource", "uid": "drift-db"},
            },
            {"title": "row", "type": "row"},
        ],
    }


def test_datasources_are_mapped_and_local_only_panels_dropped():
    out = for_cloud(dashboard(), {"prometheus": "grafanacloud-prom", "loki": "grafanacloud-logs"})
    titles = [p["title"] for p in out["panels"]]
    assert titles == ["p95", "row"] and out["id"] is None and out["uid"] == "legal-rag"
    assert out["panels"][0]["datasource"]["uid"] == "grafanacloud-prom"
    assert out["panels"][0]["targets"][0]["datasource"]["uid"] == "grafanacloud-prom"


def test_the_source_dashboard_is_not_changed():
    original = dashboard()
    for_cloud(original, {"prometheus": "x"})
    assert original["panels"][0]["datasource"]["uid"] == "prometheus" and original["id"] == 7
