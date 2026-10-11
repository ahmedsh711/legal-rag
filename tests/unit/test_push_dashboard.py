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


class FakeGrafana:
    def __init__(self, datasources):
        self.datasources = datasources

    def get(self, path):
        assert path == "/api/datasources"
        return self

    def raise_for_status(self):
        return self

    def json(self):
        return self.datasources


def test_the_stack_datasources_win_over_internal_ones():
    from push_dashboard import stack_uids

    grafana = FakeGrafana(
        [
            {"type": "loki", "name": "grafanacloud-dreamy-alert-state-history", "uid": "ash"},
            {"type": "prometheus", "name": "grafanacloud-dreamy-usage-insights", "uid": "usage"},
            {"type": "prometheus", "name": "grafanacloud-dreamy-prom", "uid": "grafanacloud-prom"},
            {"type": "loki", "name": "grafanacloud-dreamy-logs", "uid": "grafanacloud-logs"},
        ]
    )
    assert stack_uids(grafana) == {"prometheus": "grafanacloud-prom", "loki": "grafanacloud-logs"}
