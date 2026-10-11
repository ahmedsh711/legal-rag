# Running parts of the stack in the cloud

Everything runs locally by default. Each section below moves one part to a free hosted service; they are independent, so you can do any of them. Keys go in `.env` (git-ignored) or in the hosting service's secret store, never in the repository.

| Part | Service | What changes |
|---|---|---|
| Chat UI | Streamlit Community Cloud | public URL for `ui/app.py`; the API stays on your machine behind an ngrok tunnel |
| Tracing and prompts | Langfuse Cloud | replaces the local Langfuse + ClickHouse stack |
| Data and experiments | DagsHub | DVC remote and MLflow server instead of local MinIO + MLflow |
| Metrics, logs, dashboards | Grafana Cloud | receives metrics and logs from the local Alloy agent |

## Chat UI on Streamlit Community Cloud

The UI is a thin client: it needs the API, which needs about 3 GB of memory for the embedding model and does not fit the free hosting tiers. The API therefore keeps running locally and is reached through an ngrok tunnel with a fixed address.

1. Create a free [ngrok](https://ngrok.com) account, then copy your authtoken and claim the free static domain (Dashboard → Domains).
2. In `.env` set `NGROK_AUTHTOKEN` and `NGROK_DOMAIN` (for example `your-name.ngrok-free.app`). `UI_API_KEY` is already generated; it is the only key the API accepts while it is public.
3. Start the API with the tunnel:
   ```bash
   docker compose --env-file .env -f docker/compose.yaml -f docker/compose.public.yaml \
     --profile core --profile public up -d
   curl -s https://<NGROK_DOMAIN>/health
   ```
4. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and create an app: repository `ahmedsh711/legal-rag`, branch `main`, main file `ui/app.py`, Python 3.12 (Advanced settings). Streamlit installs `ui/requirements.txt` only.
5. In the app's Secrets, add:
   ```toml
   API_URL = "https://<NGROK_DOMAIN>"
   API_KEY = "<the UI_API_KEY value from .env>"
   ```

The public UI works while your machine, Docker and the tunnel are running; the sidebar says when the API is unreachable. Requests are still rate-limited per key, and the `DailyTokenBudget` alert watches spend.

To run the UI locally instead: `uv run --group ui streamlit run ui/app.py` (it uses `API_URL`, default `http://127.0.0.1:8000`).

## Tracing and prompts on Langfuse Cloud

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com), create a project and an API key pair.
2. In `.env` set `LANGFUSE_HOST=https://cloud.langfuse.com` (or the US region URL shown in the project), `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. Restart the API.
3. Recreate the system prompt there and promote it through the quality gate:
   ```bash
   uv run python -m legalrag.observability.prompts push
   uv run --group eval python -m legalrag.eval.smoke          # writes the verdict for this text
   uv run python -m legalrag.observability.prompts promote --version 1 --verdict reports/eval/smoke/verdict.json
   ```
4. The local `llm-observability` profile is no longer needed. In `monitoring/grafana/provisioning/datasources/datasources.yml`, point the log-to-trace link at your cloud project URL.

## Data and experiments on DagsHub

1. Sign up at [dagshub.com](https://dagshub.com) with GitHub and connect the repository (New Repository → Connect a repository). Create a token under User Settings → Tokens.
2. DVC remote (the credentials go to `.dvc/config.local`, which git ignores):
   ```bash
   uv run dvc remote add dagshub s3://dvc
   uv run dvc remote modify dagshub endpointurl https://dagshub.com/<user>/legal-rag.s3
   uv run dvc remote modify dagshub --local access_key_id <token>
   uv run dvc remote modify dagshub --local secret_access_key <token>
   uv run dvc push -r dagshub
   uv run dvc remote default dagshub
   ```
3. MLflow: in `.env` set
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/<user>/legal-rag.mlflow
   MLFLOW_TRACKING_USERNAME=<user>
   MLFLOW_TRACKING_PASSWORD=<token>
   ```
   then copy the production pipeline config from the local server, so the API can load it from the registry (`CONFIG_SOURCE=mlflow`):
   ```bash
   docker compose --env-file .env -f docker/compose.yaml --profile tracking up -d mlflow
   uv run --group eval python -m legalrag.eval.track copy --from http://127.0.0.1:5000
   ```
   New evaluation runs (`legalrag.eval.run --mlflow`) are logged to DagsHub from then on. The local `tracking` profile can be stopped afterwards.

## Metrics, logs and dashboards on Grafana Cloud

1. Create a free stack at [grafana.com](https://grafana.com). In the stack's details page note the Prometheus remote-write URL and user ID, and the Loki URL and user ID. Create an access policy token with `metrics:write`, `logs:write` and `rules:write`, and a service account token (Editor) in the stack's Grafana.
2. In `.env` set `GRAFANA_CLOUD_PROM_URL`, `GRAFANA_CLOUD_PROM_USER`, `GRAFANA_CLOUD_LOKI_URL`, `GRAFANA_CLOUD_LOKI_USER`, `GRAFANA_CLOUD_TOKEN` (the access policy token), `GRAFANA_CLOUD_URL` (`https://<stack>.grafana.net`) and `GRAFANA_CLOUD_SA_TOKEN`.
3. Ship metrics and logs (the local Prometheus, Loki and Grafana keep working):
   ```bash
   docker compose --env-file .env -f docker/compose.yaml -f docker/compose.grafana-cloud.yaml \
     --profile core --profile monitoring up -d alloy
   ```
   Alloy federates only the series the dashboard and alerts use, which keeps the free tier's series limit in reach.
4. Publish the dashboard and the alert rules:
   ```bash
   uv run python monitoring/grafana/push_dashboard.py
   docker run --rm -v "$PWD/monitoring/prometheus/rules:/rules:ro" grafana/mimirtool:3.2.1 \
     rules load --address=<prometheus URL without /api/prom/push> --id=<prometheus user> --key=<token> \
     /rules/recording.yml /rules/alerts.yml
   ```
   Alerts are then evaluated in Grafana Cloud; set a contact point (email or Slack) there.
